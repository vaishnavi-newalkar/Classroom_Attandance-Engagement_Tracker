"""
SQLite Database Schema and Initialization
Optimized for real-time performance
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

DATABASE_PATH = Path(__file__).parent.parent.parent / "classroom_ai.db"


class DatabaseManager:
    """Async SQLite database manager with connection pooling"""
    
    def __init__(self, db_path: str = str(DATABASE_PATH)):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Students table with embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                enrollment_id TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                email TEXT,
                department TEXT,
                year INTEGER,
                section TEXT,
                embeddings BLOB NOT NULL,
                enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Create index on enrollment_id for fast lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_students_enrollment 
            ON students(enrollment_id)
        """)
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT NOT NULL,
                course_name TEXT,
                classroom TEXT,
                scheduled_start TIMESTAMP,
                scheduled_end TIMESTAMP,
                actual_start TIMESTAMP,
                actual_end TIMESTAMP,
                total_enrolled INTEGER DEFAULT 0,
                total_present INTEGER DEFAULT 0,
                average_engagement REAL,
                status TEXT DEFAULT 'scheduled',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Attendance table (optimized with composite index)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence_score REAL,
                method TEXT DEFAULT 'auto',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                UNIQUE(session_id, student_id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_attendance_session 
            ON attendance(session_id)
        """)
        
        # Engagement logs (time-series data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS engagement_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id INTEGER,
                track_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                engagement_score REAL,
                attention_score REAL,
                engagement_state TEXT,
                posture_state TEXT,
                gaze_direction TEXT,
                phone_detected BOOLEAN DEFAULT 0,
                is_sleeping BOOLEAN DEFAULT 0,
                is_drowsy BOOLEAN DEFAULT 0,
                yawn_count INTEGER DEFAULT 0,
                phone_time_sec REAL DEFAULT 0,
                head_visible BOOLEAN DEFAULT 1,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Migrate existing databases: add new columns if they don't exist
        new_columns = [
            ("attention_score",  "REAL"),
            ("engagement_state", "TEXT"),
            ("is_sleeping",      "BOOLEAN DEFAULT 0"),
            ("is_drowsy",        "BOOLEAN DEFAULT 0"),
            ("yawn_count",       "INTEGER DEFAULT 0"),
            ("phone_time_sec",   "REAL DEFAULT 0"),
        ]
        for col_name, col_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE engagement_logs ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass  # Column already exists — safe to ignore
        
        # Session summary (denormalized for fast reporting)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_summary (
                summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER UNIQUE NOT NULL,
                student_id INTEGER NOT NULL,
                avg_engagement REAL,
                attentive_percent REAL,
                distracted_percent REAL,
                phone_usage_count INTEGER DEFAULT 0,
                total_frames INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                FOREIGN KEY (student_id) REFERENCES students(student_id),
                UNIQUE(session_id, student_id)
            )
        """)
        
        # Tracking cache (in-memory style table for active sessions)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracking_cache (
                cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                student_id INTEGER,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                UNIQUE(session_id, track_id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database initialized at {self.db_path}")
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    @asynccontextmanager
    async def get_async_connection(self):
        """Async context manager for database operations"""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            conn.close()
    
    # Student Operations
    def add_student(self, enrollment_id: str, full_name: str, embeddings: List[np.ndarray], 
                   email: str = None, department: str = None, year: int = None, 
                   section: str = None) -> int:
        """
        Add student with face embeddings
        
        Args:
            enrollment_id: Student enrollment ID
            full_name: Student name
            embeddings: List of face embedding vectors (512-dim)
            
        Returns:
            student_id
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Convert embeddings to JSON for storage
        embeddings_blob = json.dumps([emb.tolist() for emb in embeddings])
        
        cursor.execute("""
            INSERT INTO students (enrollment_id, full_name, email, department, year, section, embeddings)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (enrollment_id, full_name, email, department, year, section, embeddings_blob))
        
        student_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return student_id
    
    def get_all_students(self) -> List[Dict]:
        """Get all active students"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT student_id, enrollment_id, full_name, email, department, year, section, enrolled_at
            FROM students
            WHERE is_active = 1
        """)
        
        students = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return students
    
    def get_student_embeddings(self, student_id: int) -> List[np.ndarray]:
        """Get student face embeddings"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT embeddings FROM students WHERE student_id = ?", (student_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            embeddings_json = json.loads(row['embeddings'])
            return [np.array(emb, dtype=np.float32) for emb in embeddings_json]
        return []
    
    def get_all_embeddings(self) -> Dict[int, List[np.ndarray]]:
        """Get all student embeddings (for FAISS index building)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT student_id, embeddings 
            FROM students 
            WHERE is_active = 1
        """)
        
        embeddings_dict = {}
        for row in cursor.fetchall():
            student_id = row['student_id']
            embeddings_json = json.loads(row['embeddings'])
            embeddings_dict[student_id] = [np.array(emb, dtype=np.float32) for emb in embeddings_json]
        
        conn.close()
        return embeddings_dict
    
    # Session Operations
    def create_session(self, course_code: str, course_name: str, classroom: str,
                      scheduled_start: datetime, scheduled_end: datetime) -> int:
        """Create new session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sessions (course_code, course_name, classroom, scheduled_start, scheduled_end)
            VALUES (?, ?, ?, ?, ?)
        """, (course_code, course_name, classroom, scheduled_start, scheduled_end))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def start_session(self, session_id: int):
        """Mark session as started"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions 
            SET actual_start = CURRENT_TIMESTAMP, status = 'live'
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    def end_session(self, session_id: int):
        """Mark session as ended"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE sessions 
            SET actual_end = CURRENT_TIMESTAMP, status = 'completed'
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
    
    # Attendance Operations
    def mark_attendance(self, session_id: int, student_id: int, confidence: float, method: str = 'auto'):
        """Mark attendance (idempotent - won't duplicate)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO attendance (session_id, student_id, confidence_score, method)
            VALUES (?, ?, ?, ?)
        """, (session_id, student_id, confidence, method))
        
        conn.commit()
        conn.close()
    
    def get_session_attendance(self, session_id: int) -> List[Dict]:
        """Get attendance for a session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.*, s.enrollment_id, s.full_name
            FROM attendance a
            JOIN students s ON a.student_id = s.student_id
            WHERE a.session_id = ?
            ORDER BY a.marked_at
        """, (session_id,))
        
        attendance = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return attendance
    
    # Engagement Operations (Batch Insert for Performance)
    def log_engagement_batch(self, logs: List[Dict]):
        """Batch insert engagement logs with rich behavioural signals"""
        if not logs:
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT INTO engagement_logs 
            (session_id, student_id, track_id,
             engagement_score, attention_score, engagement_state,
             posture_state, gaze_direction,
             phone_detected, is_sleeping, is_drowsy, yawn_count, phone_time_sec,
             head_visible, bbox_x, bbox_y, bbox_w, bbox_h)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (
                log['session_id'],
                log.get('student_id'),
                log.get('track_id'),
                log.get('engagement_score', 0),
                log.get('attention_score'),
                log.get('engagement_state'),
                log.get('posture_state'),
                log.get('gaze_direction'),
                int(bool(log.get('phone_detected', False))),
                int(bool(log.get('is_sleeping', False))),
                int(bool(log.get('is_drowsy', False))),
                log.get('yawn_count', 0),
                log.get('phone_time_sec', 0.0),
                int(bool(log.get('head_visible', True))),
                log.get('bbox_x'),
                log.get('bbox_y'),
                log.get('bbox_w'),
                log.get('bbox_h'),
            )
            for log in logs
        ])
        
        conn.commit()
        conn.close()
    
    # Tracking Cache Operations
    def update_tracking_cache(self, session_id: int, track_id: int, student_id: int, confidence: float):
        """Update tracking cache for active session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO tracking_cache (session_id, track_id, student_id, confidence, last_seen)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (session_id, track_id, student_id, confidence))
        
        conn.commit()
        conn.close()
    
    def get_tracking_cache(self, session_id: int) -> Dict[int, int]:
        """Get track_id -> student_id mapping"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT track_id, student_id 
            FROM tracking_cache 
            WHERE session_id = ? AND student_id IS NOT NULL
        """, (session_id,))
        
        mapping = {row['track_id']: row['student_id'] for row in cursor.fetchall()}
        conn.close()
        
        return mapping
    
    def clear_session_cache(self, session_id: int):
        """Clear tracking cache for session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM tracking_cache WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()


# Global database instance
db = DatabaseManager()
