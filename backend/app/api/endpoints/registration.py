"""
Student Registration API
Handles student enrollment with face capture
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import List
import numpy as np
import cv2
import base64
import json
from datetime import datetime

from app.db.database import db
from app.services.ml.face_recognition import face_recognizer

router = APIRouter()


@router.post("/register")
async def register_student(
    enrollment_id: str = Form(...),
    full_name: str = Form(...),
    email: str = Form(None),
    department: str = Form(None),
    year: int = Form(None),
    section: str = Form(None),
    face_images: List[str] = Form(...)  # Base64 encoded images
):
    """
    Register new student with face images
    
    Args:
        enrollment_id: Student enrollment ID
        full_name: Student name
        face_images: List of base64 encoded face images (min 3)
        
    Returns:
        Student registration confirmation
    """
    try:
        # Validate inputs
        if len(face_images) < 3:
            raise HTTPException(400, "Minimum 3 face images required")
        
        # Decode and process face images
        embeddings = []
        
        for idx, img_base64 in enumerate(face_images):
            try:
                # Decode base64
                img_data = base64.b64decode(img_base64.split(',')[1] if ',' in img_base64 else img_base64)
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Detect face
                faces = face_recognizer.detect_faces(img)
                
                if not faces:
                    raise HTTPException(400, f"No face detected in image {idx+1}")
                
                if len(faces) > 1:
                    raise HTTPException(400, f"Multiple faces detected in image {idx+1}")
                
                # Extract embedding
                embedding = face_recognizer.extract_embedding(img, faces[0].bbox)
                embeddings.append(embedding)
                
            except Exception as e:
                raise HTTPException(400, f"Error processing image {idx+1}: {str(e)}")
        
        # Store student in database
        student_id = db.add_student(
            enrollment_id=enrollment_id,
            full_name=full_name,
            embeddings=embeddings,
            email=email,
            department=department,
            year=year,
            section=section
        )
        
        # Rebuild FAISS index
        all_embeddings = db.get_all_embeddings()
        face_recognizer.build_index(all_embeddings)
        
        return {
            "success": True,
            "message": "Student registered successfully",
            "student_id": student_id,
            "enrollment_id": enrollment_id,
            "embeddings_count": len(embeddings)
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, f"Registration failed: {str(e)}")


@router.get("/students")
async def get_all_students():
    """Get list of all registered students"""
    try:
        students = db.get_all_students()
        return {
            "success": True,
            "count": len(students),
            "students": students
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch students: {str(e)}")


@router.get("/students/{student_id}")
async def get_student(student_id: int):
    """Get student details"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT student_id, enrollment_id, full_name, email, department, year, section, enrolled_at
            FROM students
            WHERE student_id = ? AND is_active = 1
        """, (student_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(404, "Student not found")
        
        return {
            "success": True,
            "student": dict(row)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch student: {str(e)}")


@router.get("/students/{student_id}/attendance")
async def get_student_attendance(student_id: int):
    """Get attendance history for a student"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Total sessions
        cursor.execute("SELECT COUNT(*) as count FROM sessions")
        total_sessions = cursor.fetchone()['count']
        
        cursor.execute("""
            SELECT a.marked_at, a.confidence_score, a.method,
                   s.course_name, s.course_code as class_id, s.scheduled_start as date
            FROM attendance a
            JOIN sessions s ON a.session_id = s.session_id
            WHERE a.student_id = ?
            ORDER BY a.marked_at DESC
        """, (student_id,))
        
        records = []
        for row in cursor.fetchall():
            records.append({
                "status": "present",
                "subject": row["course_name"] or "Unknown Subject",
                "class_id": row["class_id"] or "Unknown Class",
                "date": row["date"] or row["marked_at"],
                "confidence": row["confidence_score"]
            })
            
        conn.close()
        
        return {
            "success": True,
            "student_id": student_id,
            "total_sessions": total_sessions,
            "present": len(records),
            "records": records
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch attendance: {str(e)}")


@router.delete("/students/{student_id}")
async def delete_student(student_id: int):
    """Soft delete student (mark as inactive)"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE students
            SET is_active = 0
            WHERE student_id = ?
        """, (student_id,))
        
        if cursor.rowcount == 0:
            raise HTTPException(404, "Student not found")
        
        conn.commit()
        conn.close()
        
        # Rebuild index
        all_embeddings = db.get_all_embeddings()
        face_recognizer.build_index(all_embeddings)
        
        return {
            "success": True,
            "message": "Student deleted successfully"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, f"Failed to delete student: {str(e)}")


@router.post("/students/{student_id}/add-faces")
async def add_face_images(
    student_id: int,
    face_images: List[str] = Form(...)
):
    """Add additional face images to existing student"""
    try:
        # Get existing embeddings
        existing_embeddings = db.get_student_embeddings(student_id)
        
        if not existing_embeddings:
            raise HTTPException(404, "Student not found")
        
        # Process new images
        new_embeddings = []
        
        for idx, img_base64 in enumerate(face_images):
            img_data = base64.b64decode(img_base64.split(',')[1] if ',' in img_base64 else img_base64)
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            faces = face_recognizer.detect_faces(img)
            if not faces or len(faces) > 1:
                continue
            
            embedding = face_recognizer.extract_embedding(img, faces[0].bbox)
            new_embeddings.append(embedding)
        
        # Update database
        all_embeddings = existing_embeddings + new_embeddings
        
        conn = db.get_connection()
        cursor = conn.cursor()
        
        embeddings_blob = json.dumps([emb.tolist() for emb in all_embeddings])
        cursor.execute("""
            UPDATE students
            SET embeddings = ?
            WHERE student_id = ?
        """, (embeddings_blob, student_id))
        
        conn.commit()
        conn.close()
        
        # Rebuild index
        all_student_embeddings = db.get_all_embeddings()
        face_recognizer.build_index(all_student_embeddings)
        
        return {
            "success": True,
            "message": "Face images added successfully",
            "new_embeddings": len(new_embeddings),
            "total_embeddings": len(all_embeddings)
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, f"Failed to add faces: {str(e)}")


@router.get("/stats")
async def get_registration_stats():
    """Get registration statistics"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Total students
        cursor.execute("SELECT COUNT(*) as count FROM students WHERE is_active = 1")
        total_students = cursor.fetchone()['count']
        
        # By department
        cursor.execute("""
            SELECT department, COUNT(*) as count
            FROM students
            WHERE is_active = 1
            GROUP BY department
        """)
        by_department = [dict(row) for row in cursor.fetchall()]
        
        # Recent registrations
        cursor.execute("""
            SELECT student_id, enrollment_id, full_name, enrolled_at
            FROM students
            WHERE is_active = 1
            ORDER BY enrolled_at DESC
            LIMIT 10
        """)
        recent = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "success": True,
            "total_students": total_students,
            "by_department": by_department,
            "recent_registrations": recent
        }
    
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch stats: {str(e)}")
