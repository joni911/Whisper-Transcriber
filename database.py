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
        
        # Tabel transkripsi
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
        
        # Tabel API keys
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT NOT NULL,
                api_key TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabel laporan AI
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcription_id INTEGER,
                report_title TEXT,
                report_content TEXT,
                report_type TEXT,  -- 'summary', 'analysis', 'custom'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (transcription_id) REFERENCES transcriptions (id)
            )
        ''')
        
        # Tabel user models (untuk tracking model yang digunakan)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                model_name TEXT,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    
    # API Key Management
    def save_api_key(self, service, api_key):
        """Simpan API key"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Hapus key lama untuk service yang sama
        cursor.execute('DELETE FROM api_keys WHERE service = ?', (service,))
        
        # Simpan key baru
        cursor.execute('''
            INSERT INTO api_keys (service, api_key)
            VALUES (?, ?)
        ''', (service, api_key))
        
        conn.commit()
        conn.close()
    
    def get_api_key(self, service):
        """Dapatkan API key"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT api_key FROM api_keys WHERE service = ?
            ORDER BY created_at DESC LIMIT 1
        ''', (service,))
        
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    # AI Report Management
    def save_ai_report(self, transcription_id, report_title, report_content, report_type):
        """Simpan laporan AI"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ai_reports 
            (transcription_id, report_title, report_content, report_type)
            VALUES (?, ?, ?, ?)
        ''', (transcription_id, report_title, report_content, report_type))
        
        report_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return report_id
    
    def get_ai_reports(self, transcription_id=None):
        """Dapatkan laporan AI"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if transcription_id:
            cursor.execute('''
                SELECT ar.*, t.filename as transcription_filename
                FROM ai_reports ar
                JOIN transcriptions t ON ar.transcription_id = t.id
                WHERE ar.transcription_id = ?
                ORDER BY ar.created_at DESC
            ''', (transcription_id,))
        else:
            cursor.execute('''
                SELECT ar.*, t.filename as transcription_filename
                FROM ai_reports ar
                JOIN transcriptions t ON ar.transcription_id = t.id
                ORDER BY ar.created_at DESC
            ''')
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_ai_report(self, report_id):
        """Dapatkan laporan AI berdasarkan ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ar.*, t.filename as transcription_filename, t.transcription
            FROM ai_reports ar
            JOIN transcriptions t ON ar.transcription_id = t.id
            WHERE ar.id = ?
        ''', (report_id,))
        
        result = cursor.fetchone()
        conn.close()
        return result
    
    # User Model Management
    def save_user_model(self, model_id, model_name=None):
        """Simpan model yang digunakan user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Hapus entri lama untuk model yang sama
            cursor.execute('DELETE FROM user_models WHERE model_id = ?', (model_id,))
            
            # Simpan model baru
            cursor.execute('''
                INSERT INTO user_models (model_id, model_name)
                VALUES (?, ?)
            ''', (model_id, model_name))
            
            conn.commit()
            conn.close()
        except:
            pass
    
    def get_user_models(self):
        """Dapatkan model yang pernah digunakan user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT DISTINCT model_id, model_name 
                FROM user_models 
                ORDER BY last_used DESC 
                LIMIT 10
            ''')
            
            user_models = cursor.fetchall()
            conn.close()
            
            return [{'id': model[0], 'name': model[1] or model[0]} for model in user_models]
        except:
            return []

# Inisialisasi database saat import
db = TranscriptionDB()