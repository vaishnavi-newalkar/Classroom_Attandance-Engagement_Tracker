"""
ByteTrack Integration for Multi-Object Tracking
Maintains consistent student IDs across frames
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from filterpy.kalman import KalmanFilter
import lap

@dataclass
class Track:
    """Object track with history"""
    track_id: int
    bbox: List[int]  # [x, y, w, h]
    confidence: float
    
    # Identity
    student_id: int = None
    face_embedding: np.ndarray = None
    
    # State
    state: str = 'active'  # active, lost, terminated
    frames_since_update: int = 0
    age: int = 0
    hits: int = 0
    
    # Kalman filter
    kf: KalmanFilter = None


class ByteTracker:
    """
    Simplified ByteTrack implementation
    Optimized for classroom scenarios
    """
    
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30, 
                 match_thresh: float = 0.8):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
        
        print(f"✅ ByteTracker initialized (thresh={track_thresh}, buffer={track_buffer})")
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of dicts with keys: bbox, confidence, (optional) student_id, embedding
            
        Returns:
            List of active Track objects
        """
        self.frame_count += 1
        
        # Separate by confidence
        high_conf = [d for d in detections if d['confidence'] >= self.track_thresh]
        low_conf = [d for d in detections if d['confidence'] < self.track_thresh]
        
        # Get current tracks
        active_tracks = [t for t in self.tracks if t.state == 'active']
        lost_tracks = [t for t in self.tracks if t.state == 'lost']
        
        # First association: high confidence with active tracks
        matched, unmatched_tracks, unmatched_dets = self._match(
            active_tracks, high_conf, threshold=self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            track = active_tracks[track_idx]
            det = high_conf[det_idx]
            self._update_track(track, det)
        
        # Second association: unmatched active tracks with low confidence
        unmatched_track_objs = [active_tracks[i] for i in unmatched_tracks]
        matched2, unmatched_tracks2, _ = self._match(
            unmatched_track_objs, low_conf, threshold=0.5
        )
        
        for track_idx, det_idx in matched2:
            track = unmatched_track_objs[track_idx]
            det = low_conf[det_idx]
            self._update_track(track, det)
        
        # Mark unmatched tracks as lost
        for idx in unmatched_tracks2:
            track = unmatched_track_objs[idx]
            track.state = 'lost'
            track.frames_since_update += 1
        
        # Third association: lost tracks with remaining high conf detections
        remaining_dets = [high_conf[i] for i in unmatched_dets]
        matched3, _, unmatched_dets3 = self._match(
            lost_tracks, remaining_dets, threshold=0.7
        )
        
        for track_idx, det_idx in matched3:
            track = lost_tracks[track_idx]
            det = remaining_dets[det_idx]
            self._update_track(track, det)
            track.state = 'active'
        
        # Create new tracks for unmatched detections
        for idx in unmatched_dets3:
            det = remaining_dets[idx]
            new_track = self._create_track(det)
            self.tracks.append(new_track)
        
        # Update all tracks
        for track in self.tracks:
            track.age += 1
            if track.state == 'lost' and track.frames_since_update > self.track_buffer:
                track.state = 'terminated'
        
        # Remove terminated
        self.tracks = [t for t in self.tracks if t.state != 'terminated']
        
        return [t for t in self.tracks if t.state == 'active']
    
    def _match(self, tracks: List[Track], detections: List[Dict], threshold: float) -> Tuple:
        """
        Match tracks to detections using IoU + embedding similarity
        
        Returns:
            (matched_pairs, unmatched_track_indices, unmatched_det_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # IoU-based cost
                iou = self._compute_iou(track.bbox, det['bbox'])
                
                # Embedding similarity (if available)
                emb_sim = 0.0
                if 'embedding' in det and track.face_embedding is not None:
                    emb_sim = np.dot(det['embedding'], track.face_embedding)
                
                # Combined similarity (higher is better)
                similarity = 0.7 * iou + 0.3 * emb_sim
                cost_matrix[i, j] = similarity
        
        # Convert to minimization
        cost_matrix = 1 - cost_matrix
        
        # Hungarian algorithm
        row_ind, col_ind = lap.lapjv(cost_matrix, extend_cost=True)[:2]
        
        # Filter by threshold
        matched = []
        unmatched_tracks = []
        
        for i in range(len(tracks)):
            if col_ind[i] >= 0:
                similarity = 1 - cost_matrix[i, col_ind[i]]
                if similarity >= threshold:
                    matched.append((i, col_ind[i]))
                else:
                    unmatched_tracks.append(i)
            else:
                unmatched_tracks.append(i)
        
        matched_det_indices = [m[1] for m in matched]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_det_indices]
        
        return matched, unmatched_tracks, unmatched_dets
    
    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute IoU between two bboxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_track(self, detection: Dict) -> Track:
        """Create new track from detection"""
        track = Track(
            track_id=self.next_track_id,
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            student_id=detection.get('student_id'),
            face_embedding=detection.get('embedding')
        )
        
        track.kf = self._init_kalman(detection['bbox'])
        track.hits = 1
        
        self.next_track_id += 1
        return track
    
    def _update_track(self, track: Track, detection: Dict):
        """Update track with new detection"""
        track.bbox = detection['bbox']
        track.confidence = detection['confidence']
        track.frames_since_update = 0
        track.hits += 1
        
        # Update Kalman filter
        if track.kf:
            self._update_kalman(track.kf, detection['bbox'])
        
        # Update student ID if detected
        if 'student_id' in detection and detection['student_id']:
            if track.student_id is None:
                track.student_id = detection['student_id']
                print(f"🎯 Track {track.track_id} → Student {track.student_id}")
        
        # Update embedding
        if 'embedding' in detection:
            track.face_embedding = detection['embedding']
    
    def _init_kalman(self, bbox: List[int]) -> KalmanFilter:
        """Initialize Kalman filter for motion prediction"""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        s = w * h
        r = w / h if h > 0 else 1.0
        
        kf.x = np.array([[cx], [cy], [s], [r], [0], [0], [0]])
        kf.H = np.eye(4, 7)
        kf.Q[4:, 4:] *= 0.01
        kf.R[2:, 2:] *= 10
        
        return kf
    
    def _update_kalman(self, kf: KalmanFilter, bbox: List[int]):
        """Update Kalman filter"""
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        s = w * h
        r = w / h if h > 0 else 1.0
        
        kf.predict()
        kf.update(np.array([[cx], [cy], [s], [r]]))
    
    def get_track(self, track_id: int) -> Track:
        """Get track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None
    
    def get_active_tracks(self) -> List[Track]:
        """Get all active tracks"""
        return [t for t in self.tracks if t.state == 'active']
    
    def reset(self):
        """Reset tracker"""
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        print("🔄 Tracker reset")


# Global tracker instance
tracker = ByteTracker()
