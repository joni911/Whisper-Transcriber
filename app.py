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
import re
import html
from typing import Tuple

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



def format_report_content_for_document(content: str) -> Tuple[str, str]:
    """
    Memformat konten laporan untuk dokumen DOCX/PDF dengan menangani markdown dan komponen khusus.
    
    Args:
        content (str): Konten laporan mentah
        
    Returns:
        Tuple[str, str]: Tuple berisi (formatted_content, html_content) untuk keperluan berbeda
    """
    
    # Normalisasi line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    
    # Simpan bagian think terlebih dahulu
    think_sections = []
    
    def extract_think_section(match):
        think_content = match.group(1).strip()
        think_sections.append(think_content)
        return f"<<<THINK_PLACEHOLDER_{len(think_sections)-1}>>>"
    
    # Ekstrak semua bagian <think>
    content = re.sub(r'<think>(.*?)</think>', extract_think_section, content, flags=re.DOTALL)
    
    # Bersihkan karakter yang tidak diinginkan tapi pertahankan struktur
    content = re.sub(r'[^\S\n]+$', '', content, flags=re.MULTILINE)  # hapus spasi di akhir baris
    content = re.sub(r'\n{3,}', '\n\n', content)  # maksimal 2 baris kosong berturut-turut
    
    # Konversi heading markdown ke format yang konsisten
    # H1
    content = re.sub(r'^#\s+(.+)', r'# \1', content, flags=re.MULTILINE)
    # H2
    content = re.sub(r'^##\s+(.+)', r'## \1', content, flags=re.MULTILINE)
    # H3
    content = re.sub(r'^###\s+(.+)', r'### \1', content, flags=re.MULTILINE)
    # H4
    content = re.sub(r'^####\s+(.+)', r'#### \1', content, flags=re.MULTILINE)
    # H5
    content = re.sub(r'^#####\s+(.+)', r'##### \1', content, flags=re.MULTILINE)
    # H6
    content = re.sub(r'^######\s+(.+)', r'###### \1', content, flags=re.MULTILINE)
    
    # Konversi bold dan italic
    content = re.sub(r'\*\*(.*?)\*\*', r'**\1**', content)  # bold
    content = re.sub(r'\*(.*?)\*', r'*\1*', content)        # italic
    
    # Konversi list
    content = re.sub(r'^\s*-\s+(.+)', r'- \1', content, flags=re.MULTILINE)  # unordered list
    content = re.sub(r'^\s*\d+\.\s+(.+)', r'1. \1', content, flags=re.MULTILINE)  # ordered list
    
    # Persiapkan versi HTML untuk dokumen yang mendukung HTML
    html_content = content
    
    # Konversi untuk HTML (jika diperlukan)
    # Heading
    html_content = re.sub(r'^######\s+(.+)', r'<h6>\1</h6>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^#####\s+(.+)', r'<h5>\1</h5>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^####\s+(.+)', r'<h4>\1</h4>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^###\s+(.+)', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^##\s+(.+)', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^#\s+(.+)', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
    
    # Bold dan italic
    html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
    html_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_content)
    
    # List
    html_content = re.sub(r'^-\s+(.+)', r'<li>\1</li>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'(<li>.*?</li>\s*)+', r'<ul>\n\g<0></ul>\n', html_content, flags=re.DOTALL)
    
    html_content = re.sub(r'^\d+\.\s+(.+)', r'<li>\1</li>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'(<li>.*?</li>\s*)+', r'<ol>\n\g<0></ol>\n', html_content, flags=re.DOTALL)
    
    # Masukkan kembali bagian think dengan format yang sesuai
    for i, think_content in enumerate(think_sections):
        # Format untuk konten teks biasa
        think_placeholder = f"<<<THINK_PLACEHOLDER_{i}>>>"
        formatted_think = f"\n[Proses Berpikir]\n{think_content}\n"
        content = content.replace(think_placeholder, formatted_think)
        
        # Format untuk HTML
        html_think = f'<div class="think-box"><h3>💭 Proses Berpikir</h3><p>{html.escape(think_content)}</p></div>'
        html_content = html_content.replace(think_placeholder, html_think)
    
    # Pastikan ada baris baru di akhir
    if not content.endswith('\n'):
        content += '\n'
    
    if not html_content.endswith('\n'):
        html_content += '\n'
    
    return content, html_content

def clean_for_docx(content: str) -> str:
    """
    Membersihkan konten untuk format DOCX.
    
    Args:
        content (str): Konten yang akan dibersihkan
        
    Returns:
        str: Konten yang telah dibersihkan untuk DOCX
    """
    # Gunakan fungsi format utama
    plain_content, _ = format_report_content_for_document(content)
    
    # Tambahkan pembersihan khusus untuk DOCX
    # Ganti karakter yang bermasalah di DOCX
    replacements = {
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2018': "'",  # left single quotation mark
        '\u2019': "'",  # right single quotation mark
        '\u201c': '"',  # left double quotation mark
        '\u201d': '"',  # right double quotation mark
        '\u2026': '...', # horizontal ellipsis
    }
    
    for unicode_char, replacement in replacements.items():
        plain_content = plain_content.replace(unicode_char, replacement)
    
    # Hapus karakter kontrol yang tidak diizinkan di XML/DOCX
    # Izinkan hanya karakter yang valid dalam XML
    cleaned_content = ""
    for char in plain_content:
        if ord(char) == 0x09 or ord(char) == 0x0A or ord(char) == 0x0D or \
           (0x20 <= ord(char) <= 0xD7FF) or \
           (0xE000 <= ord(char) <= 0xFFFD) or \
           (0x10000 <= ord(char) <= 0x10FFFF):
            cleaned_content += char
        else:
            # Ganti karakter tidak valid dengan spasi
            cleaned_content += ' '
    
    return cleaned_content.strip()

def clean_for_pdf(content: str) -> str:
    """
    Membersihkan konten untuk format PDF.
    
    Args:
        content (str): Konten yang akan dibersihkan
        
    Returns:
        str: Konten yang telah dibersihkan untuk PDF
    """
    # Gunakan fungsi format utama
    plain_content, _ = format_report_content_for_document(content)
    
    # Untuk PDF, kita bisa mempertahankan lebih banyak karakter Unicode
    # Tapi tetap perlu membersihkan karakter kontrol
    cleaned_content = ""
    for char in plain_content:
        if ord(char) >= 32 or char in '\n\r\t':
            cleaned_content += char
        else:
            # Ganti karakter kontrol dengan spasi
            cleaned_content += ' '
    
    return cleaned_content.strip()

# Contoh penggunaan dalam fungsi download_report
# app.py (fungsi download_report yang dimodifikasi)
@app.route('/download-report/<int:report_id>/<format>')
def download_report(report_id, format):
    """Download laporan dalam format DOCX atau PDF"""
    try:
        report = db.get_ai_report(report_id)
        if not report:
            flash('Laporan tidak ditemukan')
            return redirect(url_for('view_reports'))
        
        report_title = report[2]  # report_title
        report_content = report[3]  # report_content original
        
        if format == 'docx':
            filename = f"{report_title.replace(' ', '_')}.docx"
            filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
            # Bersihkan dan format konten untuk DOCX menggunakan mdformat
            cleaned_content = clean_for_docx(report_content)
            ai_reporter.create_docx_report(report_title, cleaned_content, filepath)
        elif format == 'pdf':
            filename = f"{report_title.replace(' ', '_')}.pdf"
            filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
            # Bersihkan dan format konten untuk PDF menggunakan mdformat
            cleaned_content = clean_for_pdf(report_content)
            ai_reporter.create_pdf_report(report_title, cleaned_content, filepath)
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