"""
Enhanced Classroom AI System - Main Application
Production-ready SIH demo version
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

# Import routers
from app.api.endpoints import registration, sessions, attendance

# Import services for initialization
from app.db.database import db
from app.services.ml.face_recognition import face_recognizer
from app.services.ml.engagement_detector import engagement_detector
from app.services.tracking.byte_tracker import tracker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("=" * 60)
    print("🚀 Starting Classroom AI System...")
    print("=" * 60)
    
    # Initialize database
    db._init_db()
    print("✅ Database initialized")
    
    # Load ML models
    try:
        print("📦 Loading ML models...")
        
        # Load face recognition models
        face_recognizer.load_models()
        
        # Build FAISS index from existing students
        all_embeddings = db.get_all_embeddings()
        face_recognizer.build_index(all_embeddings)
        print(f"✅ FAISS index built with {len(all_embeddings)} students")
        
        # Load engagement detector
        engagement_detector.load_models()
        
        print("✅ All models loaded successfully")
    except Exception as e:
        print(f"⚠️  Model loading error (will use fallbacks): {e}")
    
    print("=" * 60)
    print("✅ Classroom AI System Ready!")
    print("📖 API Docs: http://localhost:8000/docs")
    print("=" * 60)
    
    yield
    
    # Shutdown
    print("\n🛑 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Classroom AI System - Enhanced",
    version="2.0.0",
    description="Real-time Attendance & Engagement Tracking with AI",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(registration.router, prefix="/api/registration", tags=["Registration"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])
app.include_router(attendance.router, prefix="/api/attendance", tags=["Attendance"])

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Mount static files (optional)
try:
    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
except:
    pass


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Classroom AI System API - Enhanced Edition",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Student Registration with Webcam",
            "Real-time Face Recognition",
            "Multi-Object Tracking (ByteTrack)",
            "YOLO Pose Engagement Detection",
            "Live Dashboard with WebSocket",
            "Optimized for CPU (30-50 students)"
        ],
        "endpoints": {
            "docs": "/docs",
            "registration": "/api/registration",
            "sessions": "/api/sessions"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        student_count = cursor.fetchone()[0]
        conn.close()
        
        # Check models
        models_loaded = (
            face_recognizer.face_detector is not None and
            engagement_detector.model is not None
        )
        
        return {
            "status": "healthy",
            "database": "connected",
            "models": "loaded" if models_loaded else "partial",
            "students_enrolled": student_count
        }
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")


@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Student count
        cursor.execute("SELECT COUNT(*) as count FROM students WHERE is_active = 1")
        total_students = cursor.fetchone()['count']
        
        # Session count
        cursor.execute("SELECT COUNT(*) as count FROM sessions")
        total_sessions = cursor.fetchone()['count']
        
        # Recent session
        cursor.execute("""
            SELECT course_name, classroom, scheduled_start, status
            FROM sessions
            ORDER BY scheduled_start DESC
            LIMIT 1
        """)
        recent_session = cursor.fetchone()
        
        conn.close()
        
        # Active sessions
        from app.services.video.video_processor import active_processors
        active_count = len(active_processors)
        
        return {
            "success": True,
            "total_students": total_students,
            "total_sessions": total_sessions,
            "active_sessions": active_count,
            "recent_session": dict(recent_session) if recent_session else None
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
