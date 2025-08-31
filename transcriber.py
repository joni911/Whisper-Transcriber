import whisper
import subprocess
import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import logging
import json

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
    
    def split_audio_to_chunks(self, audio_file, chunk_duration=30):
        """Split audio ke chunk kecil"""
        try:
            # Gunakan ffmpeg untuk split audio
            base_name = os.path.splitext(audio_file)[0]
            chunks = []
            
            # Dapatkan durasi total
            duration = self.get_audio_duration(audio_file)
            if not duration:
                return [audio_file]  # Return original jika tidak bisa dapat durasi
            
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
                
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0 and os.path.exists(chunk_file):
                    chunks.append(chunk_file)
                
                start_time = end_time
                chunk_index += 1
            
            return chunks
        except Exception as e:
            print(f"❌ Error splitting audio: {e}")
            return [audio_file]
    
    def transcribe_with_progress(self, input_file, progress_callback=None):
        """Transcribe dengan progress tracking"""
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
            
            # Dapatkan durasi untuk estimasi
            duration = self.get_audio_duration(audio_file)
            
            # Split audio ke chunk untuk progress tracking
            print("🔄 Memecah audio ke chunk...")
            chunks = self.split_audio_to_chunks(audio_file, chunk_duration=30)  # 30 detik per chunk
            total_chunks = len(chunks)
            
            if progress_callback:
                progress_callback(5, f"Mempersiapkan {total_chunks} segmen...")
            
            # Load model
            model = self.load_model()
            
            # Transcribe setiap chunk
            transcriptions = []
            total_word_count = 0
            
            print(f"🔄 Memulai transkripsi {total_chunks} segmen...")
            
            for i, chunk_file in enumerate(chunks):
                if progress_callback:
                    progress = 10 + int((i / total_chunks) * 80)  # 10% - 90%
                    progress_callback(progress, f"Memproses segmen {i+1}/{total_chunks}...")
                
                try:
                    # Transcribe chunk
                    result = model.transcribe(
                        chunk_file,
                        language="id",
                        task="transcribe",
                        fp16=self.gpu_available
                    )
                    
                    chunk_text = result["text"]
                    transcriptions.append(chunk_text)
                    total_word_count += len(chunk_text.split())
                    
                    # Hapus chunk temporary
                    if chunk_file != audio_file:  # Jangan hapus file asli
                        try:
                            os.remove(chunk_file)
                        except:
                            pass
                
                except Exception as chunk_error:
                    print(f"❌ Error transcribing chunk {i}: {chunk_error}")
                    if progress_callback:
                        progress_callback(None, f"Error segmen {i+1}: {str(chunk_error)}")
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
            return None, None, None