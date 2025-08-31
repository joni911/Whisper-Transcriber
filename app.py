from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
import os
import threading
import time
from werkzeug.utils import secure_filename
from transcriber import AudioTranscriber
from database import db
from setup import setup_environment

app = Flask(__name__)
app.secret_key = 'whisper_transcriber_secret_key'

# Konfigurasi upload
UPLOAD_FOLDER = 'uploads'
TRANSCRIPTS_FOLDER = 'transcripts'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'mp4', 'mkv', 'mov', 'avi', 'wmv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TRANSCRIPTS_FOLDER'] = TRANSCRIPTS_FOLDER

# Buat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)

# Inisialisasi transcriber
transcriber = AudioTranscriber()

# Store progress for each job
transcription_progress = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    transcriptions = db.get_all_transcriptions()
    return render_template('index.html', transcriptions=transcriptions)

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
                
                # Update progress awal
                progress_callback(5, 'Mempersiapkan proses...')
                
                # Dapatkan durasi file untuk estimasi waktu
                duration = transcriber.get_audio_duration(filepath)
                if duration:
                    # Estimasi: 1 menit audio = 0.7-2.0 menit proses
                    speed_factor = 0.7 if transcriber.gpu_available else 2.0
                    estimated_time = duration * speed_factor / 60  # dalam menit
                    transcription_progress[job_id]['estimated_time'] = estimated_time
                
                # Transcribe dengan progress tracking
                transcription, duration, word_count = transcriber.transcribe_with_progress(
                    filepath, progress_callback
                )
                
                if transcription:
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
                    db.add_transcription(
                        filename=transcript_filename,
                        original_file=filename,
                        transcription=transcription,
                        duration=duration,
                        word_count=word_count
                    )
                    
                    # Update final time
                    final_elapsed = time.time() - start_time
                    transcription_progress[job_id]['elapsed_time'] = final_elapsed
                    progress_callback(100, f'Selesai dalam {final_elapsed/60:.1f} menit!')
                    transcription_progress[job_id]['status'] = 'completed'
                    print(f"✅ Transkripsi selesai: {filename}")
                else:
                    transcription_progress[job_id]['status'] = 'failed'
                    transcription_progress[job_id]['message'] = 'Gagal transkripsi'
                    print(f"❌ Gagal transkripsi: {filename}")
                    
            except Exception as e:
                transcription_progress[job_id]['status'] = 'failed'
                transcription_progress[job_id]['message'] = f'Error: {str(e)}'
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
    transcription = db.get_transcription(transcript_id)
    if transcription:
        return render_template('transcript.html', transcription=transcription)
    flash('Transkripsi tidak ditemukan')
    return redirect(url_for('index'))

@app.route('/download/<int:transcript_id>')
def download_transcript(transcript_id):
    transcription = db.get_transcription(transcript_id)
    if transcription:
        transcript_filename = transcription[1]  # filename
        transcript_filepath = os.path.join(app.config['TRANSCRIPTS_FOLDER'], transcript_filename)
        if os.path.exists(transcript_filepath):
            return send_file(transcript_filepath, as_attachment=True)
    flash('File tidak ditemukan')
    return redirect(url_for('index'))

@app.route('/delete/<int:transcript_id>')
def delete_transcript(transcript_id):
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)