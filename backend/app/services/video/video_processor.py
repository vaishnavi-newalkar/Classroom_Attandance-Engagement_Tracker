"""
Real-Time Video Processing Pipeline
Optimized for 30-50 students with low latency
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
import asyncio
from queue import Queue
from threading import Thread
import time
from collections import deque
import traceback

from app.services.ml.face_recognition import face_recognizer
from app.services.ml.engagement_detector import engagement_detector
from app.services.ml.temporal_tracker import TemporalTrackerManager
from app.services.video.video_source import WebcamSource, ESP32Source
from app.services.tracking.byte_tracker import tracker
from app.db.database import db


class VideoProcessor:
    """
    Optimized video processing pipeline
    Features:
    - Async frame processing
    - Frame skipping for performance
    - Batch database writes
    - Caching for attendance
    """
    
    def __init__(self, session_id: int, frame_skip: int = 2):
        self.session_id = session_id
        self.frame_skip = frame_skip
        
        # Processing state
        self.is_running = False
        self.frame_count = 0
        self.fps = 0
        
        # Queues for async processing
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=10)
        
        # Attendance cache (avoid duplicate marking)
        self.attendance_marked = set()
        self.max_unidentified = 0
        
        # Engagement logs buffer (batch writes)
        self.engagement_buffer = []
        self.buffer_size = 20  # Write every 20 entries
        
        # Performance tracking
        self.processing_times = deque(maxlen=30)
        
        # Student names cache
        self.student_names = {}
        try:
            students = db.get_all_students()
            for s in students:
                self.student_names[s['student_id']] = s['full_name']
        except Exception as e:
            print(f"Failed to load names: {e}")
        
        # Per-student temporal tracker (EMA attention score, sleep, phone debounce)
        self.temporal_tracker = TemporalTrackerManager()
        
        print(f"📹 VideoProcessor initialized for session {session_id}")
    
    def start(self, camera_source: str = "0"):
        """
        Start video processing
        
        Args:
            camera_source: Camera index or video path
        """
        self.is_running = True
        self.active_source_type = 'webcam'

        
        # Start capture thread
        capture_thread = Thread(target=self._capture_frames, args=(camera_source,), daemon=True)
        capture_thread.start()
        
        # Start processing thread
        process_thread = Thread(target=self._process_frames, daemon=True)
        process_thread.start()
        
        print(f"▶️  Video processing started (source: {camera_source})")
    
    def stop(self):
        """Stop video processing"""
        self.is_running = False
        
        # Flush remaining engagement logs
        if self.engagement_buffer:
            db.log_engagement_batch(self.engagement_buffer)
            self.engagement_buffer = []
        
        # Clear tracking cache
        db.clear_session_cache(self.session_id)
        
        print(f"⏹️  Video processing stopped")
    
    def _capture_frames(self, camera_source: str):
        """
        Frame capture thread
        Runs independently to avoid blocking
        """
        # Instantiate correct video source logic wrapper
        if str(camera_source).startswith("http"):
            source = ESP32Source(camera_source)
        else:
            source = WebcamSource(camera_source)
            
        self.active_source_type = source.get_source_type()
        
        if not source.isOpened():
            print(f"❌ Failed to open camera: {camera_source}")
            self.is_running = False
            return
        
        print(f"📷 Camera opened: {camera_source} ({self.active_source_type})")
        
        frame_idx = 0
        
        while self.is_running:
            ret, frame = source.read()
            
            if not ret:
                # Network cameras might temporarily drop setup backoff in source.read()
                time.sleep(0.05)
                continue
            
            # Frame skipping for performance
            if frame_idx % self.frame_skip == 0:
                # Resize for faster processing
                frame_resized = cv2.resize(frame, (640, 480))
                
                # Keep only the absolute latest frame in queue to eliminate visual lag
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                
                try:
                    self.frame_queue.put_nowait(frame_resized)
                except:
                    pass
            
            frame_idx += 1
        
        source.release()
        print("📷 Camera released")
    
    def _process_frames(self):
        """
        Frame processing thread
        Main processing pipeline
        """
        while self.is_running:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue
            
            # Get frame
            frame = self.frame_queue.get()
            start_time = time.time()
            
            try:
                # Process frame
                result = self._process_single_frame(frame)
                
                # Add to result queue
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Update FPS
                if len(self.processing_times) > 0:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    self.fps = 1.0 / avg_time if avg_time > 0 else 0
                
            except Exception as e:
                print(f"❌ Frame processing error: {e}")
                err_str = traceback.format_exc()
                print(err_str)
                with open('traceback.log', 'a', encoding='utf-8') as tf:
                    tf.write(err_str + '\n')
                continue
    
    def _process_single_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame through complete pipeline
        
        Pipeline:
        1. Face detection & recognition
        2. Engagement detection (YOLO Pose)
        3. Multi-object tracking
        4. Attendance marking
        5. Engagement logging
        
        Returns:
            Processing result dict
        """
        self.frame_count += 1
        
        # Step 1: Face Recognition
        faces = face_recognizer.recognize_faces(frame)
        
        # Step 2: Engagement Detection (hybrid — pass temporal profiles for EMA + sleep checks)
        temporal_profiles = {
            tid: self.temporal_tracker.get_or_create_profile(tid)
            for tid in self.temporal_tracker.profiles  # pre-existing profiles
        }
        # Pass full dict to engine so new track_ids also get created on the fly
        engagement_results = engagement_detector.analyze_frame(frame, temporal_profiles=self.temporal_tracker.profiles)
        
        # Step 3: Prepare detections for tracking
        detections = []
        for eng in engagement_results:
            det = {
                'bbox': eng.bbox,
                'confidence': eng.confidence,
            }
            
            # Match with face detection
            matched_face = self._match_face_to_bbox(faces, eng.bbox)
            if matched_face:
                det['student_id'] = matched_face.student_id
                det['embedding'] = matched_face.embedding
            
            detections.append(det)
        
        # Step 4: Update tracker
        active_tracks = tracker.update(detections)
        
        # Step 5: Merge tracking with engagement data
        tracked_engagement = []
        for track in active_tracks:
            # Find corresponding engagement data
            eng_data = self._find_engagement_by_bbox(engagement_results, track.bbox)
            
            if eng_data:
                # Update with tracking info
                eng_data.track_id = track.track_id
                eng_data.student_id = track.student_id
                
                # Ensure temporal profile exists under the real track_id from tracker
                profile = self.temporal_tracker.get_or_create_profile(track.track_id)
                if track.student_id:
                    profile.student_id = track.student_id
                
                tracked_engagement.append(eng_data)
                
                # Mark attendance (first time only)
                if track.student_id and track.student_id not in self.attendance_marked:
                    db.mark_attendance(
                        self.session_id, 
                        track.student_id, 
                        track.confidence,
                        method='auto'
                    )
                    self.attendance_marked.add(track.student_id)
                    print(f"✅ Attendance: Student {track.student_id}")
                
                # Update tracking cache
                if track.student_id:
                    db.update_tracking_cache(
                        self.session_id, 
                        track.track_id, 
                        track.student_id,
                        track.confidence
                    )
                
                # Buffer engagement log — now includes rich behavioural signals
                profile = self.temporal_tracker.get_or_create_profile(track.track_id)
                log_entry = {
                    'session_id': self.session_id,
                    'student_id': track.student_id,
                    'track_id': track.track_id,
                    'engagement_score': eng_data.engagement_score,
                    'attention_score': eng_data.attention_score,
                    'engagement_state': eng_data.engagement_state,
                    'posture_state': eng_data.posture_state,
                    'gaze_direction': eng_data.gaze_direction,
                    'phone_detected': eng_data.phone_detected,
                    'is_sleeping': eng_data.is_sleeping,
                    'is_drowsy': eng_data.is_drowsy,
                    'yawn_count': eng_data.yawn_count,
                    'phone_time_sec': eng_data.phone_time_sec,
                    'head_visible': eng_data.head_visible,
                    'bbox_x': eng_data.bbox[0],
                    'bbox_y': eng_data.bbox[1],
                    'bbox_w': eng_data.bbox[2],
                    'bbox_h': eng_data.bbox[3]
                }
                self.engagement_buffer.append(log_entry)
        
        # Batch write engagement logs
        if len(self.engagement_buffer) >= self.buffer_size:
            db.log_engagement_batch(self.engagement_buffer)
            self.engagement_buffer = []
            
        current_unidentified = sum(1 for eng in tracked_engagement if eng.student_id is None)
        if current_unidentified > self.max_unidentified:
            self.max_unidentified = current_unidentified
            try:
                conn = db.get_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE sessions SET unidentified_count = ? WHERE session_id = ?", (self.max_unidentified, self.session_id))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Failed to update session unidentified_count: {e}")
        
        # Compute class metrics — use smoothed attention score
        class_avg_engagement = 0.0
        if tracked_engagement:
            class_avg_engagement = sum(e.engagement_score for e in tracked_engagement) / len(tracked_engagement)
        
        return {
            'frame_count': self.frame_count,
            'fps': self.fps,
            'students_detected': len(tracked_engagement),
            'attendance_count': len(self.attendance_marked),
            'class_engagement': class_avg_engagement,
            'engagement_data': tracked_engagement,
            'annotated_frame': self._annotate_frame(frame, tracked_engagement)
        }
    
    def _match_face_to_bbox(self, faces, person_bbox: List[int]):
        """Match face detection to person bbox"""
        px, py, pw, ph = person_bbox
        
        best_match = None
        best_iou = 0.0
        
        for face in faces:
            fx, fy, fw, fh = face.bbox
            
            # Compute IoU
            x_left = max(px, fx)
            y_top = max(py, fy)
            x_right = min(px + pw, fx + fw)
            y_bottom = min(py + ph, fy + fh)
            
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                face_area = fw * fh
                iou = intersection / face_area if face_area > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = face
        
        return best_match if best_iou > 0.3 else None
    
    def _find_engagement_by_bbox(self, engagement_results, bbox: List[int]):
        """Find engagement data by bbox (approximate match)"""
        for eng in engagement_results:
            # Simple bbox matching
            if self._bbox_overlap(eng.bbox, bbox) > 0.5:
                return eng
        return None
    
    def _bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute bbox overlap ratio"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        
        return intersection / area1 if area1 > 0 else 0.0
    
    def _annotate_frame(self, frame: np.ndarray, engagement_data: List) -> np.ndarray:
        """
        Annotate frame with visualizations
        
        Draws:
        - Bounding boxes
        - Student names/IDs
        - Engagement scores
        - Alerts
        """
        annotated = frame.copy()
        
        for eng in engagement_data:
            x, y, w, h = eng.bbox
            state = getattr(eng, 'engagement_state', None)
            score = eng.engagement_score

            # Colour-coded by state
            if eng.is_sleeping:
                color = (128, 0, 128)    # Purple — sleeping
            elif eng.is_drowsy:
                color = (0, 100, 255)    # Orange — drowsy
            elif state == 'attentive':
                color = (0, 220, 80)     # Green — attentive
            elif state == 'mildly_distracted':
                color = (0, 230, 230)    # Cyan — mild
            elif state == 'highly_distracted':
                color = (0, 0, 255)      # Red — highly distracted
            else:
                color = (180, 180, 180)  # Grey — unknown

            # Draw bbox
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            
            # Build label
            label_parts = []
            
            if eng.student_id:
                name = self.student_names.get(eng.student_id, f"ID:{eng.student_id}")
                short_name = name.split()[0] if isinstance(name, str) else name
                label_parts.append(short_name)
            else:
                label_parts.append(f"T:{eng.track_id}")
            
            attn = getattr(eng, 'attention_score', score * 100)
            label_parts.append(f"A:{attn:.0f}%")

            if eng.is_sleeping:          label_parts.append("💤")
            elif eng.is_drowsy:          label_parts.append("😪")
            if eng.phone_detected:       label_parts.append("📱")
            if getattr(eng, 'is_yawning', False): label_parts.append("🥱")

            label = " ".join(label_parts)
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y-20), (x+text_w, y), color, -1)
            cv2.putText(annotated, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
        
        # Draw FPS
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                   
        # Draw ESP32 monitoring indicator
        if getattr(self, 'active_source_type', 'webcam') == 'esp32':
            cv2.putText(annotated, "🔴 Monitoring via ESP32-CAM", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated
    
    def get_latest_result(self) -> Optional[Dict]:
        """Get latest processing result (non-blocking)"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None


# Global processor instances (session_id -> processor)
active_processors: Dict[int, VideoProcessor] = {}
