from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import os
import threading
import time
import requests
from werkzeug.utils import secure_filename
from transcriber import AudioTranscriber
from database import db
from setup import setup_environment
from ai_reporter import ai_reporter
import openai

app = Flask(__name__)
app.secret_key = 'whisper_transcriber_secret_key'

# Konfigurasi upload
UPLOAD_FOLDER = 'uploads'
TRANSCRIPTS_FOLDER = 'transcripts'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'mp4', 'mkv', 'mov', 'avi', 'wmv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRANSCRIPTS_FOLDER'] = TRANSCRIPTS_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Inisialisasi transcriber
transcriber = AudioTranscriber()

# Store progress for each job
transcription_progress = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    try:
        transcriptions = db.get_all_transcriptions()
        return render_template('index.html', transcriptions=transcriptions)
    except Exception as e:
        flash(f'Error loading transcriptions: {str(e)}')
        return render_template('index.html', transcriptions=[])

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada file yang dipilih')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Buat job ID untuk tracking progress
        job_id = str(int(time.time() * 1000))
        transcription_progress[job_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Memulai proses...',
            'filename': filename,
            'estimated_time': 0,
            'elapsed_time': 0,
            'start_time': time.time()
        }
        
        # Proses transkripsi di background dengan progress callback
        def process_transcription(job_id, filepath, filename):
            try:
                start_time = time.time()
                
                # Progress callback function
                def progress_callback(progress, message):
                    if job_id in transcription_progress:
                        transcription_progress[job_id]['message'] = message
                        if progress is not None:
                            transcription_progress[job_id]['progress'] = progress
                        # Update elapsed time
                        elapsed = time.time() - transcription_progress[job_id]['start_time']
                        transcription_progress[job_id]['elapsed_time'] = elapsed
                        print(f"[{time.strftime('%I:%M:%S %p')}] {message}")
                
                # Update progress awal
                progress_callback(2, 'Memulai proses...')
                
                # Dapatkan durasi file untuk estimasi waktu
                try:
                    duration = transcriber.get_audio_duration(filepath)
                    if duration:
                        # Estimasi: 1 menit audio = 0.7-2.0 menit proses
                        speed_factor = 0.7 if transcriber.gpu_available else 2.0
                        estimated_time = duration * speed_factor / 60  # dalam menit
                        transcription_progress[job_id]['estimated_time'] = estimated_time
                        progress_callback(5, f'Memvalidasi file... (Durasi: {duration/60:.1f} menit)')
                except Exception as e:
                    progress_callback(5, 'Memvalidasi file...')
                    print(f"Warning: Could not get audio duration: {e}")
                
                # Transcribe dengan progress tracking
                try:
                    transcription, duration, word_count = transcriber.transcribe_with_progress(
                        filepath, progress_callback
                    )
                except Exception as transcribe_error:
                    if job_id in transcription_progress:
                        transcription_progress[job_id]['status'] = 'failed'
                        error_message = str(transcribe_error)
                        if 'moov atom not found' in error_message:
                            transcription_progress[job_id]['message'] = '❌ File video korup. Coba upload ulang file yang utuh.'
                        elif 'timeout' in error_message.lower():
                            transcription_progress[job_id]['message'] = '❌ Proses timeout. File terlalu besar atau sistem sibuk.'
                        else:
                            transcription_progress[job_id]['message'] = f'❌ Error: {error_message[:100]}...'
                    print(f"❌ Error transkripsi: {transcribe_error}")
                    return
                
                if transcription and len(transcription.strip()) > 0:
                    # Update elapsed time
                    elapsed = time.time() - start_time
                    transcription_progress[job_id]['elapsed_time'] = elapsed
                    progress_callback(95, 'Menyimpan hasil...')
                    
                    # Simpan transkripsi ke file
                    transcript_filename = os.path.splitext(filename)[0] + '.txt'
                    transcript_filepath = os.path.join(app.config['TRANSCRIPTS_FOLDER'], transcript_filename)
                    
                    with open(transcript_filepath, 'w', encoding='utf-8') as f:
                        f.write(transcription)
                    
                    # Simpan ke database
                    transcription_id = db.add_transcription(
                        filename=transcript_filename,
                        original_file=filename,
                        transcription=transcription,
                        duration=duration,
                        word_count=word_count
                    )
                    
                    # Update final time
                    final_elapsed = time.time() - start_time
                    transcription_progress[job_id]['elapsed_time'] = final_elapsed
                    progress_callback(100, f'✅ Selesai dalam {final_elapsed/60:.1f} menit!')
                    transcription_progress[job_id]['status'] = 'completed'
                    transcription_progress[job_id]['transcription_id'] = transcription_id
                    print(f"✅ Transkripsi selesai: {filename}")
                else:
                    if job_id in transcription_progress:
                        transcription_progress[job_id]['status'] = 'failed'
                        transcription_progress[job_id]['message'] = '❌ Gagal transkripsi - hasil kosong'
                    print(f"❌ Gagal transkripsi: {filename}")
                    
            except Exception as e:
                if job_id in transcription_progress:
                    transcription_progress[job_id]['status'] = 'failed'
                    error_msg = str(e)
                    if 'timeout' in error_msg.lower():
                        transcription_progress[job_id]['message'] = '❌ Timeout - File terlalu besar'
                    elif 'permission' in error_msg.lower():
                        transcription_progress[job_id]['message'] = '❌ Permission denied - Cek hak akses file'
                    else:
                        transcription_progress[job_id]['message'] = f'❌ Error sistem: {error_msg[:100]}...'
                print(f"❌ Error transkripsi: {e}")
        
        # Jalankan di thread terpisah
        thread = threading.Thread(target=process_transcription, args=(job_id, filepath, filename))
        thread.start()
        
        # Redirect ke halaman progress
        return redirect(url_for('progress', job_id=job_id))
    
    flash('Format file tidak didukung')
    return redirect(url_for('index'))

@app.route('/progress/<job_id>')
def progress(job_id):
    if job_id not in transcription_progress:
        flash('Job tidak ditemukan')
        return redirect(url_for('index'))
    return render_template('progress.html', job_id=job_id)

@app.route('/progress_status/<job_id>')
def progress_status(job_id):
    if job_id in transcription_progress:
        # Update elapsed time
        if 'start_time' in transcription_progress[job_id]:
            elapsed = time.time() - transcription_progress[job_id]['start_time']
            transcription_progress[job_id]['elapsed_time'] = elapsed
        
        return jsonify(transcription_progress[job_id])
    return jsonify({'status': 'not_found', 'progress': 0, 'message': 'Job tidak ditemukan'})

@app.route('/transcript/<int:transcript_id>')
def view_transcript(transcript_id):
    try:
        transcription = db.get_transcription(transcript_id)
        if transcription:
            return render_template('transcript.html', transcription=transcription)
        flash('Transkripsi tidak ditemukan')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error loading transcript: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download/<int:transcript_id>')
def download_transcript(transcript_id):
    try:
        transcription = db.get_transcription(transcript_id)
        if transcription:
            transcript_filename = transcription[1]  # filename
            transcript_filepath = os.path.join(app.config['TRANSCRIPTS_FOLDER'], transcript_filename)
            if os.path.exists(transcript_filepath):
                return send_file(transcript_filepath, as_attachment=True)
        flash('File tidak ditemukan')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error downloading transcript: {str(e)}')
        return redirect(url_for('index'))

@app.route('/delete/<int:transcript_id>')
def delete_transcript(transcript_id):
    try:
        transcription = db.get_transcription(transcript_id)
        if transcription:
            # Hapus file
            transcript_filename = transcription[1]
            transcript_filepath = os.path.join(app.config['TRANSCRIPTS_FOLDER'], transcript_filename)
            if os.path.exists(transcript_filepath):
                os.remove(transcript_filepath)
            
            # Hapus dari database
            db.delete_transcription(transcript_id)
            flash('Transkripsi berhasil dihapus')
        else:
            flash('Transkripsi tidak ditemukan')
        
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error deleting transcript: {str(e)}')
        return redirect(url_for('index'))

@app.route('/setup')
def setup_page():
    """Halaman setup"""
    return render_template('setup.html')

@app.route('/run_setup')
def run_setup():
    """Jalankan setup"""
    try:
        gpu_available = setup_environment()
        return jsonify({
            'status': 'success',
            'gpu_available': gpu_available,
            'message': 'Setup berhasil!'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error setup: {str(e)}'
        })

# AI Report Routes
@app.route('/ai-settings')
def ai_settings():
    """Halaman pengaturan AI"""
    try:
        api_key = db.get_api_key('openrouter')
        return render_template('ai_settings.html', api_key=bool(api_key))
    except Exception as e:
        flash(f'Error loading AI settings: {str(e)}')
        return render_template('ai_settings.html', api_key=False)

@app.route('/save-api-key', methods=['POST'])
def save_api_key():
    """Simpan API key"""
    try:
        api_key = request.form.get('api_key')
        if api_key:
            db.save_api_key('openrouter', api_key)
            ai_reporter.set_api_key(api_key)
            flash('API key berhasil disimpan!')
        else:
            flash('API key tidak boleh kosong!')
        return redirect(url_for('ai_settings'))
    except Exception as e:
        flash(f'Error saving API key: {str(e)}')
        return redirect(url_for('ai_settings'))

@app.route('/generate-report/<int:transcript_id>')
def generate_report_page(transcript_id):
    """Halaman generate laporan"""
    try:
        transcription = db.get_transcription(transcript_id)
        if not transcription:
            flash('Transkripsi tidak ditemukan')
            return redirect(url_for('index'))
        
        api_key = db.get_api_key('openrouter')
        return render_template('generate_report.html', 
                             transcription=transcription, 
                             api_key_set=bool(api_key))
    except Exception as e:
        flash(f'Error loading report page: {str(e)}')
        return redirect(url_for('index'))

@app.route('/create-report', methods=['POST'])
def create_report():
    """Buat laporan AI"""
    try:
        transcript_id = int(request.form.get('transcript_id', 0))
        report_type = request.form.get('report_type', '').strip()
        custom_prompt = request.form.get('custom_prompt', '').strip()
        model_id = request.form.get('model_id', '').strip()
        analysis_type = request.form.get('analysis_type', 'general')
        
        # Validasi input
        if transcript_id <= 0:
            return jsonify({'status': 'error', 'message': 'ID transkripsi tidak valid'})
        
        if not report_type:
            return jsonify({'status': 'error', 'message': 'Tipe laporan harus dipilih'})
        
        # Validasi model_id
        if not model_id:
            model_id = 'mistralai/mistral-7b-instruct:free'
        
        # Dapatkan transkripsi
        transcription = db.get_transcription(transcript_id)
        if not transcription:
            return jsonify({'status': 'error', 'message': 'Transkripsi tidak ditemukan'})
        
        transcription_text = transcription[3]  # Kolom transcription
        
        # Cek API key
        api_key = db.get_api_key('openrouter')
        if not api_key:
            return jsonify({'status': 'error', 'message': 'API key belum diatur. Silakan atur di halaman AI Settings.'})
        
        ai_reporter.set_api_key(api_key)
        
        # Generate laporan berdasarkan tipe
        if report_type == 'summary':
            report_content = ai_reporter.generate_summary(transcription_text, model_id)
            report_title = f"Ringkasan - {transcription[2]}"
            report_db_type = 'summary'
        elif report_type == 'analysis':
            report_content = ai_reporter.generate_analysis(transcription_text, analysis_type, model_id)
            report_title = f"Analisis - {transcription[2]}"
            report_db_type = 'analysis'
        elif report_type == 'custom':
            if not custom_prompt:
                return jsonify({'status': 'error', 'message': 'Prompt kustom tidak boleh kosong!'})
            report_content = ai_reporter.generate_custom_report(transcription_text, custom_prompt, model_id)
            report_title = f"Laporan Kustom - {transcription[2]}"
            report_db_type = 'custom'
        else:
            return jsonify({'status': 'error', 'message': 'Tipe laporan tidak valid'})
        
        # Validasi hasil
        if not report_content or len(report_content.strip()) == 0:
            return jsonify({'status': 'error', 'message': 'Gagal menghasilkan laporan - hasil kosong'})
        
        # Simpan laporan ke database
        report_id = db.save_ai_report(transcript_id, report_title, report_content, report_db_type)
        
        return jsonify({
            'status': 'success', 
            'message': 'Laporan berhasil dibuat!',
            'report_id': report_id
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})

@app.route('/reports')
def view_reports():
    """Lihat semua laporan"""
    try:
        reports = db.get_ai_reports()
        return render_template('reports.html', reports=reports)
    except Exception as e:
        flash(f'Error loading reports: {str(e)}')
        return render_template('reports.html', reports=[])

@app.route('/report/<int:report_id>')
def view_report(report_id):
    """Lihat detail laporan"""
    try:
        report = db.get_ai_report(report_id)
        if not report:
            flash('Laporan tidak ditemukan')
            return redirect(url_for('view_reports'))
        return render_template('view_report.html', report=report)
    except Exception as e:
        flash(f'Error loading report: {str(e)}')
        return redirect(url_for('view_reports'))

@app.route('/download-report/<int:report_id>/<format>')
def download_report(report_id, format):
    """Download laporan dalam format DOCX atau PDF"""
    try:
        report = db.get_ai_report(report_id)
        if not report:
            flash('Laporan tidak ditemukan')
            return redirect(url_for('view_reports'))
        
        report_title = report[2]  # report_title
        report_content = report[3]  # report_content
        
        if format == 'docx':
            filename = f"{report_title.replace(' ', '_')}.docx"
            filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
            ai_reporter.create_docx_report(report_title, report_content, filepath)
        elif format == 'pdf':
            filename = f"{report_title.replace(' ', '_')}.pdf"
            filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
            ai_reporter.create_pdf_report(report_title, report_content, filepath)
        else:
            flash('Format tidak didukung')
            return redirect(url_for('view_report', report_id=report_id))
        
        return send_file(filepath, as_attachment=True)
        
    except Exception as e:
        flash(f'Error download: {str(e)}')
        return redirect(url_for('view_report', report_id=report_id))

# AI Model Routes
@app.route('/ai-models')
def ai_models():
    """Halaman daftar model AI"""
    try:
        # Set API key jika tersedia
        api_key = db.get_api_key('openrouter')
        if api_key:
            ai_reporter.set_api_key(api_key)
        
        # Dapatkan model yang tersedia
        available_models = ai_reporter.get_available_models()
        user_models = ai_reporter.get_user_models()
        
        return render_template('ai_models.html', 
                             models=available_models,
                             user_models=user_models,
                             api_key_set=bool(api_key))
    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(url_for('ai_settings'))

@app.route('/test-model', methods=['POST'])
def test_model():
    """Test model AI"""
    try:
        model_id = request.form.get('model_id', '').strip()
        test_prompt = "Berikan ringkasan singkat tentang pentingnya AI dalam transkripsi audio dalam bahasa Indonesia."
        
        if not model_id:
            return jsonify({'status': 'error', 'message': 'Model tidak dipilih'})
        
        # Set API key
        api_key = db.get_api_key('openrouter')
        if not api_key:
            return jsonify({'status': 'error', 'message': 'API key belum diatur'})
        
        ai_reporter.set_api_key(api_key)
        
        # Record start time
        start_time = time.time()
        
        # Test dengan prompt sederhana
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=[
                {"role": "user", "content": test_prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        end_time = time.time()
        response_time = int((end_time - start_time) * 1000)  # in milliseconds
        
        result = response.choices[0].message.content.strip()
        
        return jsonify({
            'status': 'success',
            'message': 'Model berhasil diuji!',
            'result': result,
            'response_time': response_time
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error: {str(e)}'})

@app.route('/get-models')
def get_models():
    """API endpoint untuk mendapatkan daftar model"""
    try:
        # Set API key jika tersedia
        api_key = db.get_api_key('openrouter')
        if api_key:
            ai_reporter.set_api_key(api_key)
        
        # Dapatkan model yang tersedia
        available_models = ai_reporter.get_available_models()
        user_models = ai_reporter.get_user_models()
        
        return jsonify({
            'status': 'success',
            'models': available_models,
            'user_models': user_models
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)