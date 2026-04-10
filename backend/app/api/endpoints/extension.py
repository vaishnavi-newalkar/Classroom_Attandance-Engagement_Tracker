from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import json
from datetime import datetime
from groq import Groq
from typing import List, Optional
from dotenv import load_dotenv

from app.db.database import db

load_dotenv()
router = APIRouter()

class AttendancePayload(BaseModel):
    session_id: int
    enrollment_id: str

class TranscriptPayload(BaseModel):
    transcript: str

class EngagementPayload(BaseModel):
    session_id: int
    enrollment_id: str
    action: str  # "switched_tab", "answered_correct", "answered_wrong"

@router.post("/attendance")
async def mark_ext_attendance(payload: AttendancePayload):
    """Marks a student present from the Chrome Extension"""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT student_id FROM students WHERE enrollment_id = ?", (payload.enrollment_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Student not found")
        
    student_id = row["student_id"]
    
    # Mark attendance method='online'
    cursor.execute("""
        INSERT OR IGNORE INTO attendance (session_id, student_id, confidence_score, method)
        VALUES (?, ?, 1.0, 'online')
    """, (payload.session_id, student_id))
    
    cursor.execute("""
        UPDATE sessions
        SET total_present = (SELECT COUNT(*) FROM attendance WHERE session_id = ?)
        WHERE session_id = ?
    """, (payload.session_id, payload.session_id))
    
    conn.commit()
    conn.close()
    return {"success": True, "student_id": student_id, "message": "Online attendance tracked"}


@router.post("/generate-quiz")
async def generate_quiz(payload: TranscriptPayload):
    """Takes recent Google Meet transcript, calls Groq LLM, returns a MCQ"""
    if not payload.transcript or len(payload.transcript) < 20:
        return {"success": False, "message": "Transcript too short"}
        
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "dummy_key":
        # Fallback question if they don't have Groq key setup yet
        return {
            "success": True,
            "question": "The professor was just talking about:",
            "options": ["The material just discussed", "Something unrelated", "A joke", "Unknown"],
            "correct_index": 0
        }
        
    client = Groq(api_key=api_key)
    prompt = f"""
    You are an AI teaching assistant. I will provide you with a live, 60-second transcript snippet from an online class. 
    (Note: The transcript may contain multiple speakers, including students asking questions or casual noise).
    
    Your task is to isolate the TEACHER'S primary lecture material and generate ONE technical or conceptual multiple-choice question strictly based on the academic subject discussed. 
    Ignore casual student interruptions. Formulate a direct, academic question testing the factual concepts just explained by the instructor.
    
    Provide exactly 4 options. Only one option should be correct.
    
    Format your response purely as a JSON object:
    {{
        "question": "The actual academic question...",
        "options": ["opt1", "opt2", "opt3", "opt4"],
        "correct_index": 0
    }}
    
    Transcript: "{payload.transcript}"
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        data["success"] = True
        return data
    except Exception as e:
        print("Groq Error:", e)
        return {"success": False, "message": "LLM generation failed."}


@router.post("/log-engagement")
async def log_engagement(payload: EngagementPayload):
    """Applies distracted/attentive penalties directly to the session summary."""
    conn = db.get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT student_id FROM students WHERE enrollment_id = ?", (payload.enrollment_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Student not found")
        
    student_id = row["student_id"]
    
    # Get existing session summary or create heavily mocked
    cursor.execute("""
        INSERT OR IGNORE INTO session_summary (session_id, student_id, avg_engagement, attentive_percent, distracted_percent)
        VALUES (?, ?, 100.0, 100.0, 0.0)
    """, (payload.session_id, student_id))
    
    if payload.action == "switched_tab" or payload.action == "answered_wrong":
        cursor.execute("""
            UPDATE session_summary
            SET distracted_percent = distracted_percent + 10.0,
                attentive_percent = attentive_percent - 10.0
            WHERE session_id = ? AND student_id = ?
        """, (payload.session_id, student_id))
    
    elif payload.action == "answered_correct":
        cursor.execute("""
            UPDATE session_summary
            SET attentive_percent = attentive_percent + 5.0,
                distracted_percent = MAX(0.0, distracted_percent - 5.0)
            WHERE session_id = ? AND student_id = ?
        """, (payload.session_id, student_id))
        
    conn.commit()
    conn.close()
    return {"success": True}
