import sqlite3
from datetime import datetime
import os

class TranscriptionDB:
    def __init__(self, db_path="transcriptions.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Inisialisasi database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_file TEXT NOT NULL,
                transcription TEXT,
                duration REAL,
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'completed'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_transcription(self, filename, original_file, transcription, duration=None, word_count=None):
        """Tambah transkripsi ke database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO transcriptions 
            (filename, original_file, transcription, duration, word_count, status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (filename, original_file, transcription, duration, word_count, 'completed'))
        
        transcription_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return transcription_id
    
    def get_all_transcriptions(self):
        """Dapatkan semua transkripsi"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, filename, original_file, duration, word_count, created_at, status
            FROM transcriptions 
            ORDER BY created_at DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_transcription(self, transcription_id):
        """Dapatkan transkripsi berdasarkan ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM transcriptions WHERE id = ?
        ''', (transcription_id,))
        
        result = cursor.fetchone()
        conn.close()
        return result
    
    def delete_transcription(self, transcription_id):
        """Hapus transkripsi"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM transcriptions WHERE id = ?', (transcription_id,))
        
        conn.commit()
        conn.close()
        return cursor.rowcount > 0

# Inisialisasi database saat import
db = TranscriptionDB()