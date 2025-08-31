import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_ffmpeg():
    """Cek apakah FFmpeg tersedia"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        print("✅ FFmpeg ditemukan di sistem")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg tidak ditemukan")
        return False

def install_ffmpeg_windows():
    """Install FFmpeg di Windows menggunakan Chocolatey"""
    try:
        print("📦 Menginstall FFmpeg menggunakan Chocolatey...")
        subprocess.run(['choco', 'install', 'ffmpeg', '-y'], check=True)
        print("✅ FFmpeg berhasil diinstall")
        return True
    except:
        print("❌ Gagal menginstall FFmpeg dengan Chocolatey")
        print("💡 Alternatif: Download manual dari https://www.gyan.dev/ffmpeg/builds/")
        return False

def install_ffmpeg_linux():
    """Install FFmpeg di Linux"""
    try:
        print("📦 Menginstall FFmpeg...")
        # Ubuntu/Debian
        subprocess.run(['sudo', 'apt', 'update'], check=True)
        subprocess.run(['sudo', 'apt', 'install', 'ffmpeg', '-y'], check=True)
        print("✅ FFmpeg berhasil diinstall")
        return True
    except:
        try:
            # Fedora/RHEL
            subprocess.run(['sudo', 'dnf', 'install', 'ffmpeg', '-y'], check=True)
            print("✅ FFmpeg berhasil diinstall")
            return True
        except:
            print("❌ Gagal menginstall FFmpeg")
            return False

def install_ffmpeg_mac():
    """Install FFmpeg di macOS"""
    try:
        print("📦 Menginstall FFmpeg menggunakan Homebrew...")
        subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
        print("✅ FFmpeg berhasil diinstall")
        return True
    except:
        print("❌ Gagal menginstall FFmpeg dengan Homebrew")
        return False

def check_gpu():
    """Cek apakah GPU tersedia"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU ditemukan: {gpu_name}")
            return True
        else:
            print("⚠️  GPU tidak ditemukan, menggunakan CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch tidak ditemukan")
        return False

def install_whisper():
    """Install Whisper dan dependensi"""
    try:
        print("📦 Menginstall Whisper...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'openai-whisper'], check=True)
        print("✅ Whisper berhasil diinstall")
        return True
    except:
        print("❌ Gagal menginstall Whisper")
        return False

def setup_environment():
    """Setup lengkap environment"""
    print("🔧 Setup Environment Whisper Transcriber")
    print("=" * 50)
    
    # Cek dan install FFmpeg
    if not check_ffmpeg():
        system = platform.system().lower()
        if system == "windows":
            install_ffmpeg_windows()
        elif system == "linux":
            install_ffmpeg_linux()
        elif system == "darwin":  # macOS
            install_ffmpeg_mac()
        else:
            print(f"❌ Sistem {system} tidak didukung untuk auto-install FFmpeg")
    
    # Cek GPU
    gpu_available = check_gpu()
    
    # Install Whisper jika belum ada
    try:
        import whisper
        print("✅ Whisper sudah terinstall")
    except ImportError:
        install_whisper()
    
    print("\n✅ Setup selesai!")
    if gpu_available:
        print("💡 GPU tersedia - akan menggunakan model 'small'")
    else:
        print("💡 GPU tidak tersedia - akan menggunakan model 'base'")
    
    return gpu_available

if __name__ == "__main__":
    setup_environment()