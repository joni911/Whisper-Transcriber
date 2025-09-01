import whisper
import subprocess
import os
import torch
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self):
        self.gpu_available = self.check_gpu()
        self.model = None
        self.model_size = self.determine_model_size()
    
    def check_gpu(self):
        """Cek ketersediaan GPU"""
        try:
            return torch.cuda.is_available()
        except:
            return False
    
    def determine_model_size(self):
        """Tentukan ukuran model berdasarkan GPU"""
        if self.gpu_available:
            print("🖥️  GPU tersedia - menggunakan model 'small'")
            return "small"
        else:
            print("🖥️  GPU tidak tersedia - menggunakan model 'base'")
            return "base"
    
    def load_model(self):
        """Load model Whisper"""
        if self.model is None:
            device = "cuda" if self.gpu_available else "cpu"
            print(f"📥 Memuat model Whisper ({self.model_size})...")
            self.model = whisper.load_model(self.model_size, device=device)
            print("✅ Model berhasil dimuat")
        return self.model
    
    def get_audio_duration(self, audio_file):
        """Dapatkan durasi audio"""
        try:
            command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                      '-of', 'default=noprint_wrappers=1:nokey=1', audio_file]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            print(f"⚠️  Error getting duration: {e}")
            pass
        return None
    
    def extract_audio(self, input_file, output_file):
        """Ekstrak audio dari file video"""
        try:
            print(f"🎵 Mengekstrak audio dari {os.path.basename(input_file)}...")
            command = [
                'ffmpeg', '-i', input_file,
                '-q:a', '0', '-map', 'a',
                '-ar', '16000', '-ac', '1',
                output_file, '-y'
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                if file_size > 0:
                    print(f"✅ Audio berhasil diekstrak ({file_size} bytes)")
                    return True
                else:
                    print("❌ File ekstrak kosong")
                    return False
            else:
                print(f"❌ Gagal mengekstrak audio: {result.stderr[:200]}...")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Timeout saat mengekstrak audio")
            return False
        except Exception as e:
            print(f"❌ Error ekstraksi: {e}")
            return False
    
    def split_audio_to_chunks(self, audio_file, chunk_duration=30):
        """Split audio ke chunk kecil"""
        try:
            # Gunakan ffmpeg untuk split audio
            base_name = os.path.splitext(audio_file)[0]
            chunks = []
            
            # Dapatkan durasi total
            duration = self.get_audio_duration(audio_file)
            if not duration or duration <= 0:
                print("⚠️  Tidak bisa mendapatkan durasi, menggunakan file penuh")
                return [audio_file]  # Return original jika tidak bisa dapat durasi
            
            print(f"📊 Durasi audio: {duration/60:.1f} menit")
            
            # Split ke chunk
            start_time = 0
            chunk_index = 0
            
            while start_time < duration:
                end_time = min(start_time + chunk_duration, duration)
                chunk_file = f"{base_name}_chunk_{chunk_index:04d}.wav"
                
                command = [
                    'ffmpeg', '-i', audio_file,
                    '-ss', str(start_time), '-t', str(end_time - start_time),
                    '-ar', '16000', '-ac', '1', '-y',
                    chunk_file
                ]
                
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and os.path.exists(chunk_file):
                    file_size = os.path.getsize(chunk_file)
                    if file_size > 0:  # Cek apakah file tidak kosong
                        chunks.append(chunk_file)
                        print(f"✅ Chunk {chunk_index+1} dibuat ({file_size} bytes)")
                    else:
                        print(f"⚠️  Chunk {chunk_index} kosong, dilewati")
                        try:
                            os.remove(chunk_file)
                        except:
                            pass
                else:
                    print(f"⚠️  Gagal membuat chunk {chunk_index}")
                
                start_time = end_time
                chunk_index += 1
            
            if not chunks:
                print("⚠️  Tidak ada chunk yang berhasil dibuat, menggunakan file asli")
                return [audio_file]
            
            print(f"✅ Berhasil membuat {len(chunks)} chunk")
            return chunks
        except subprocess.TimeoutExpired:
            print("❌ Timeout saat membuat chunk")
            return [audio_file]
        except Exception as e:
            print(f"❌ Error splitting audio: {e}")
            return [audio_file]
    
    def transcribe_with_progress(self, input_file, progress_callback=None):
        """Transcribe dengan progress tracking"""
        temp_files_to_cleanup = []
        
        try:
            # Cek ekstensi file
            _, ext = os.path.splitext(input_file.lower())
            
            if ext in ['.wav', '.mp3', '.flac', '.m4a']:
                audio_file = input_file
                print("🎵 Menggunakan file audio langsung")
                if progress_callback:
                    progress_callback(5, "Memvalidasi file audio...")
            else:
                # Ekstrak audio dari video
                audio_file = os.path.splitext(input_file)[0] + '_extracted.wav'
                temp_files_to_cleanup.append(audio_file)
                if progress_callback:
                    progress_callback(5, "Mengekstrak audio dari video...")
                
                try:
                    if not self.extract_audio(input_file, audio_file):
                        raise Exception("Gagal mengekstrak audio")
                except Exception as extract_error:
                    if progress_callback:
                        progress_callback(None, f"Gagal ekstraksi: {str(extract_error)}")
                    raise extract_error
            
            # Validasi file audio hasil ekstraksi
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
                raise Exception("File audio hasil ekstraksi tidak valid")
            
            # Dapatkan durasi untuk estimasi
            if progress_callback:
                progress_callback(10, "Menghitung durasi audio...")
            
            duration = self.get_audio_duration(audio_file)
            
            # Split audio ke chunk untuk progress tracking
            if progress_callback:
                progress_callback(15, "Mempersiapkan chunk audio...")
            
            chunks = self.split_audio_to_chunks(audio_file, chunk_duration=60)  # 60 detik per chunk untuk lebih cepat
            temp_files_to_cleanup.extend(chunks)
            total_chunks = len(chunks)
            
            if progress_callback:
                progress_callback(20, f"Mempersiapkan {total_chunks} segmen...")
            
            # Load model
            if progress_callback:
                progress_callback(25, "Memuat model Whisper...")
            
            model = self.load_model()
            
            # Transcribe setiap chunk
            transcriptions = []
            total_word_count = 0
            
            print(f"🔄 Memulai transkripsi {total_chunks} segmen...")
            
            for i, chunk_file in enumerate(chunks):
                if progress_callback:
                    progress = 30 + int((i / total_chunks) * 60)  # 30% - 90%
                    progress_callback(progress, f"Memproses segmen {i+1}/{total_chunks}...")
                
                try:
                    # Cek apakah chunk valid
                    if not os.path.exists(chunk_file) or os.path.getsize(chunk_file) == 0:
                        print(f"⚠️  Chunk {i} tidak valid, dilewati")
                        continue
                    
                    # Transcribe chunk dengan timeout
                    print(f"🔊 Memproses chunk {i+1}/{total_chunks}...")
                    
                    # Set timeout berdasarkan durasi chunk
                    chunk_duration = self.get_audio_duration(chunk_file) or 60
                    timeout_seconds = max(60, int(chunk_duration * 3))  # Minimal 60 detik
                    
                    result = model.transcribe(
                        chunk_file,
                        language="id",
                        task="transcribe",
                        fp16=self.gpu_available,
                        verbose=False
                    )
                    
                    chunk_text = result["text"]
                    transcriptions.append(chunk_text)
                    total_word_count += len(chunk_text.split())
                    
                    print(f"✅ Chunk {i+1} selesai ({len(chunk_text)} karakter)")
                    
                except Exception as chunk_error:
                    print(f"❌ Error transcribing chunk {i}: {chunk_error}")
                    # Jangan stop proses, lanjut ke chunk berikutnya
                    continue
            
            # Gabungkan semua transkripsi
            if progress_callback:
                progress_callback(95, "Menggabungkan hasil...")
            
            final_transcription = " ".join(transcriptions)
            
            print("✅ Transkripsi selesai")
            return final_transcription, duration, total_word_count
            
        except Exception as e:
            logger.error(f"Error dalam transkripsi: {e}")
            print(f"❌ Error transkripsi: {e}")
            raise e
        finally:
            # Cleanup file temporary dengan error handling yang lebih baik
            self.cleanup_temp_files(temp_files_to_cleanup, input_file)
    
    def cleanup_temp_files(self, temp_files, original_file):
        """Cleanup file temporary dengan aman"""
        try:
            print(f"🧹 Membersihkan {len(temp_files)} file temporary...")
            for temp_file in temp_files:
                # Jangan hapus file asli
                if temp_file and temp_file != original_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        print(f"🗑️  File temporary dihapus: {os.path.basename(temp_file)}")
                    except Exception as e:
                        print(f"⚠️  Gagal menghapus {temp_file}: {e}")
            print("✅ Cleanup selesai")
        except Exception as e:
            print(f"⚠️  Error dalam cleanup: {e}")