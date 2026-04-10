"""
Attendance Processing API
Handles image and video uploads for offline attendance marking via face recognition.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import List, Optional
import numpy as np
import cv2
import uuid
import base64
import json
import asyncio
from datetime import datetime

from app.db.database import db
from app.services.ml.face_recognition import face_recognizer

router = APIRouter()

# In-memory job store for video processing
_jobs: dict = {}

# The face recognizer's identify_face() already filters at threshold=0.60.
# We do NOT add a second gate here — that was causing registered students to be dropped.
MIN_CONFIDENCE = 0.60  # kept for logging only


# ── Helper: get enrolled student IDs for a session ────────────────────────────
def _get_enrolled_student_ids(session_id: int) -> set:
    """
    Returns set of student_ids who belong to the session's class/section.
    Falls back to ALL active students if no section data is found (safety net).
    """
    conn = db.get_connection()
    cursor = conn.cursor()

    # Get the course_code (=class identifier like 'CS-3A') for this session
    cursor.execute("SELECT course_code FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return set()

    course_code = row["course_code"]

    # Find students whose section matches the course_code
    cursor.execute(
        "SELECT student_id FROM students WHERE section = ? AND is_active = 1",
        (course_code,)
    )
    enrolled = {r["student_id"] for r in cursor.fetchall()}

    if not enrolled:
        # If no section data exists, fall back to all registered students
        # (prevents breaking demo when section column is empty)
        cursor.execute("SELECT student_id FROM students WHERE is_active = 1")
        enrolled = {r["student_id"] for r in cursor.fetchall()}
        print(f"⚠️  No section data for course {course_code} — using all registered students")
    else:
        print(f"✅ Session {session_id} ({course_code}): {len(enrolled)} enrolled students")

    conn.close()
    return enrolled


# ── Helper ─────────────────────────────────────────────────────────────────────

def _recognize_from_frame(frame: np.ndarray) -> List[dict]:
    """Run face recognition on a frame; return all matches approved by the model."""
    faces = face_recognizer.recognize_faces(frame)
    results = []
    for face in faces:
        # face.student_id is None when identify_face() rejected the match (below model threshold)
        if face.student_id is not None:
            conf = float(face.match_confidence or 0.0)
            results.append({"student_id": face.student_id, "confidence": conf})
            print(f"  ✓ Face matched: student_id={face.student_id} conf={conf:.3f}")
    if not results:
        print("  → No faces recognised in this frame/image")
    return results


def _decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Enhance a low-quality image to improve face detection and recognition.
    Steps:
      1. Upscale if too small (InsightFace needs at least 640px shortest side)
      2. Denoise with fast Non-Local Means
      3. CLAHE contrast enhancement on luminance channel
      4. Unsharp mask sharpening
    """
    if frame is None:
        return frame

    h, w = frame.shape[:2]
    # -- 1. Upscale small images --
    min_side = min(h, w)
    if min_side < 640:
        scale = 640.0 / min_side
        new_w, new_h = int(w * scale), int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"  ↑ Upscaled {w}x{h} → {new_w}x{new_h}")

    # -- 2. Denoise (fast, small filter to avoid over-smoothing) --
    frame = cv2.fastNlMeansDenoisingColored(frame, None, h=6, hColor=6,
                                            templateWindowSize=7, searchWindowSize=21)

    # -- 3. CLAHE contrast enhancement on Y channel (luminance) --
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # -- 4. Unsharp mask sharpening --
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=2)
    frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

    return frame


# ── Process Images ──────────────────────────────────────────────────────────────

@router.post("/process-images")
async def process_images(
    session_id: int = Form(...),
    images: List[UploadFile] = File(...)
):
    """
    Upload one or more classroom photos.
    Runs face recognition on each image and marks attendance once per student.
    """
    if not images:
        raise HTTPException(400, "No images provided")

    # Only consider students enrolled in this session's class
    enrolled_ids = _get_enrolled_student_ids(session_id)
    if not enrolled_ids:
        return {"success": False, "detail": "Session not found or no students registered",
                "session_id": session_id, "images_processed": 0, "students_identified": 0, "attendance": []}

    recognized = {}   # student_id -> best confidence

    for img_file in images:
        data = await img_file.read()
        frame = _decode_image_bytes(data)
        if frame is None:
            continue
        frame = _preprocess_frame(frame)   # enhance before recognition
        for hit in _recognize_from_frame(frame):
            sid = hit["student_id"]
            conf = hit["confidence"]
            # Only keep hits that belong to this session's class
            if sid not in enrolled_ids:
                print(f"  ↳ Skipped student {sid}: not enrolled in this session's class")
                continue
            if sid not in recognized or recognized[sid] < conf:
                recognized[sid] = conf

    marked = []
    for student_id, confidence in recognized.items():
        db.mark_attendance(session_id, student_id, confidence, method="photo")
        marked.append({"student_id": student_id, "confidence": confidence})

    # Get names for response
    conn = db.get_connection()
    cursor = conn.cursor()
    named = []
    for m in marked:
        cursor.execute(
            "SELECT enrollment_id, full_name FROM students WHERE student_id = ?",
            (m["student_id"],)
        )
        row = cursor.fetchone()
        named.append({
            "student_id": m["student_id"],
            "enrollment_id": row["enrollment_id"] if row else str(m["student_id"]),
            "full_name": row["full_name"] if row else "Unknown",
            "confidence": m["confidence"]
        })
    conn.close()

    return {
        "success": True,
        "session_id": session_id,
        "images_processed": len(images),
        "students_identified": len(named),
        "attendance": named
    }


# ── Process Base64 Snapshots (from browser cam) ────────────────────────────────

@router.post("/process-snapshots")
async def process_snapshots(payload: dict):
    """
    Accept base64-encoded snapshots captured by the teacher's browser camera.
    Marks attendance for all recognized faces.
    """
    session_id = payload.get("session_id")
    snapshots: List[str] = payload.get("snapshots", [])

    if not session_id:
        raise HTTPException(400, "session_id required")
    if not snapshots:
        raise HTTPException(400, "No snapshots provided")

    enrolled_ids = _get_enrolled_student_ids(session_id)
    if not enrolled_ids:
        raise HTTPException(404, "Session not found or no students registered")

    recognized = {}

    for b64 in snapshots:
        try:
            header, encoded = b64.split(",", 1) if "," in b64 else ("", b64)
            img_data = base64.b64decode(encoded)
            frame = _decode_image_bytes(img_data)
            if frame is None:
                continue
            frame = _preprocess_frame(frame)   # enhance before recognition
            for hit in _recognize_from_frame(frame):
                sid = hit["student_id"]
                conf = hit["confidence"]
                if sid not in enrolled_ids:
                    continue
                if sid not in recognized or recognized[sid] < conf:
                    recognized[sid] = conf
        except Exception as e:
            print(f"Snapshot decode error: {e}")
            continue

    marked = []
    for student_id, confidence in recognized.items():
        db.mark_attendance(session_id, student_id, confidence, method="snapshot")
        marked.append({"student_id": student_id, "confidence": confidence})

    # Enrich with names
    conn = db.get_connection()
    cursor = conn.cursor()
    named = []
    for m in marked:
        cursor.execute(
            "SELECT enrollment_id, full_name FROM students WHERE student_id = ?",
            (m["student_id"],)
        )
        row = cursor.fetchone()
        named.append({
            "student_id": m["student_id"],
            "enrollment_id": row["enrollment_id"] if row else str(m["student_id"]),
            "full_name": row["full_name"] if row else "Unknown",
            "confidence": m["confidence"]
        })
    conn.close()

    return {
        "success": True,
        "session_id": session_id,
        "snapshots_processed": len(snapshots),
        "students_identified": len(named),
        "attendance": named
    }


# ── Process Video ───────────────────────────────────────────────────────────────

def _process_video_background(job_id: str, session_id: int, video_path: str):
    """Background thread: process video, mark attendance."""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        sample_interval = max(1, int(fps * 2))

        _jobs[job_id].update({"total_frames": total_frames, "processed_frames": 0})

        # Get enrolled students for this session's class
        enrolled_ids = _get_enrolled_student_ids(session_id)
        if not enrolled_ids:
            _jobs[job_id].update({"status": "error", "error": "Session not found or no students registered"})
            cap.release()
            return

        recognized = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                preprocessed = _preprocess_frame(frame)  # enhance each sampled frame
                for hit in _recognize_from_frame(preprocessed):
                    sid = hit["student_id"]
                    conf = hit["confidence"]
                    # Only count students enrolled in this class
                    if sid not in enrolled_ids:
                        continue
                    if sid not in recognized or recognized[sid] < conf:
                        recognized[sid] = conf
                progress = int((frame_idx / max(total_frames, 1)) * 100)
                _jobs[job_id]["processed_frames"] = frame_idx
                _jobs[job_id]["progress"] = progress
                _jobs[job_id]["students_identified"] = list(recognized.keys())
            frame_idx += 1

        cap.release()

        # Mark attendance
        named = []
        for student_id, confidence in recognized.items():
            db.mark_attendance(session_id, student_id, confidence, method="video")
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT enrollment_id, full_name FROM students WHERE student_id = ?",
                (student_id,)
            )
            row = cursor.fetchone()
            conn.close()
            named.append({
                "student_id": student_id,
                "enrollment_id": row["enrollment_id"] if row else str(student_id),
                "full_name": row["full_name"] if row else "Unknown",
                "confidence": confidence
            })

        _jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "processed_frames": total_frames,
            "attendance": named,
            "students_identified": [n["enrollment_id"] for n in named]
        })
        print(f"✅ Video job {job_id} done: {len(named)} students marked")

    except Exception as e:
        _jobs[job_id].update({"status": "error", "error": str(e)})
        print(f"❌ Video job {job_id} error: {e}")


@router.post("/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    session_id: int = Form(...),
    file: UploadFile = File(...)
):
    """Upload a classroom video for background processing + attendance marking."""
    import os, threading

    if not file.filename:
        raise HTTPException(400, "No file provided")

    # Save uploaded file
    os.makedirs("uploads", exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    save_path = f"uploads/video_{uuid.uuid4().hex}{ext}"
    data = await file.read()
    with open(save_path, "wb") as f:
        f.write(data)

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "processing",
        "progress": 0,
        "total_frames": 0,
        "processed_frames": 0,
        "students_identified": [],
        "attendance": [],
        "created_at": datetime.utcnow().isoformat()
    }

    # Run in background thread (not async because cv2 is blocking)
    t = threading.Thread(
        target=_process_video_background,
        args=(job_id, session_id, save_path),
        daemon=True
    )
    t.start()

    return {
        "success": True,
        "job_id": job_id,
        "message": "Video uploaded, processing started"
    }


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Poll the status of a video processing job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job
