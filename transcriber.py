import whisper
import subprocess
import os
import torch
from pathlib import Path
import logging

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
    
    # Tambahkan method ini di class AudioTranscriber (jika belum ada)
    def get_audio_duration(self, audio_file):
        """Dapatkan durasi audio"""
        try:
            command = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                    '-of', 'default=noprint_wrappers=1:nokey=1', audio_file]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
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
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_file):
                print(f"✅ Audio berhasil diekstrak")
                return True
            else:
                print(f"❌ Gagal mengekstrak audio: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error ekstraksi: {e}")
            return False
    
    def transcribe(self, input_file):
        """Transcribe file audio/video"""
        try:
            # Cek ekstensi file
            _, ext = os.path.splitext(input_file.lower())
            
            if ext in ['.wav', '.mp3', '.flac', '.m4a']:
                audio_file = input_file
                print("🎵 Menggunakan file audio langsung")
            else:
                # Ekstrak audio dari video
                audio_file = os.path.splitext(input_file)[0] + '_extracted.wav'
                if not self.extract_audio(input_file, audio_file):
                    return None, None, None
            
            # Dapatkan durasi
            duration = self.get_audio_duration(audio_file)
            
            # Load model
            model = self.load_model()
            
            # Transcribe
            print("🔄 Memulai transkripsi...")
            result = model.transcribe(
                audio_file,
                language="id",
                task="transcribe",
                fp16=self.gpu_available
            )
            
            transcription = result["text"]
            word_count = len(transcription.split())
            
            print("✅ Transkripsi selesai")
            return transcription, duration, word_count
            
        except Exception as e:
            logger.error(f"Error dalam transkripsi: {e}")
            print(f"❌ Error transkripsi: {e}")
            return None, None, None