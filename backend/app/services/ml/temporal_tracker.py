import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

@dataclass
class StudentTemporalProfile:
    track_id: int
    student_id: Optional[int] = None
    
    # Sliding Windows (using deque for O(1) ops)
    # Storing tuples of (timestamp, value) where needed, or just values if uniform FPS
    attention_history: deque = field(default_factory=lambda: deque(maxlen=150)) # 5 seconds @ 30fps
    pose_history: deque = field(default_factory=lambda: deque(maxlen=150))      # stores (y_nose, y_shoulder)
    bbox_center_history: deque = field(default_factory=lambda: deque(maxlen=150)) # stores (cx, cy)
    phone_history: deque = field(default_factory=lambda: deque(maxlen=90))      # binary 0/1 for last 3 seconds
    gaze_down_history: deque = field(default_factory=lambda: deque(maxlen=150)) # binary 0/1 
    
    # State flags
    is_sleeping: bool = False
    is_drowsy: bool = False
    is_using_phone: bool = False
    
    # Event Counters
    total_yawns: int = 0
    total_phone_time_sec: float = 0.0
    total_sleep_time_sec: float = 0.0
    
    # Smoothed outputs
    smoothed_attention: float = 100.0
    
    # Lock timestamps to prevent event spam (e.g., yawn cool-down)
    last_yawn_timestamp: float = 0.0
    
    def update_attention(self, raw_score: float, alpha: float = 0.1):
        """Exponential Moving Average (EMA) for Attention Score"""
        if len(self.attention_history) == 0:
            self.smoothed_attention = raw_score
        else:
            self.smoothed_attention = (alpha * raw_score) + ((1 - alpha) * self.smoothed_attention)
        
        self.attention_history.append(self.smoothed_attention)
        return self.smoothed_attention
        
    def evaluate_phone_usage(self, phone_detected_in_frame: bool, fps: float = 30.0) -> bool:
        """Temporal logic: Phone must be present in >70% of frames over a small window"""
        self.phone_history.append(1 if phone_detected_in_frame else 0)
        
        # We need a filled buffer to make a confident decision
        if len(self.phone_history) < 15: # minimum 0.5 sec threshold to start flagging
            return self.is_using_phone
            
        ratio = sum(self.phone_history) / len(self.phone_history)
        
        if ratio > 0.65:
            # Turned ON
            if not self.is_using_phone:
                self.is_using_phone = True
            # Accumulate time
            self.total_phone_time_sec += (1.0 / fps)
        elif ratio < 0.2:
            # Turned OFF clearly
            self.is_using_phone = False
            
        return self.is_using_phone

    def check_static_sleep(self) -> bool:
        """Calculate bounding box variance to definitively determine if they are asleep rather than just looking down writing"""
        # Need at least 3 seconds of data (approx 90 frames)
        if len(self.bbox_center_history) < 90 or len(self.gaze_down_history) < 90:
            self.is_sleeping = False
            return False
            
        # 1. Have they been looking down constantly? (ratio > 0.9)
        down_ratio = sum(self.gaze_down_history) / len(self.gaze_down_history)
        if down_ratio < 0.85:
            self.is_sleeping = False
            return False
            
        # 2. Are they perfectly still? Compute variance of centers
        recent_centers = list(self.bbox_center_history)
        xs = [pt[0] for pt in recent_centers]
        ys = [pt[1] for pt in recent_centers]
        
        var_x = np.var(xs)
        var_y = np.var(ys)
        
        # Tightly bounded variance threshold
        if var_x < 5.0 and var_y < 5.0:
            self.is_sleeping = True
        else:
            self.is_sleeping = False
            
        return self.is_sleeping

    def check_head_nod_drowsiness(self) -> bool:
        """Velocity checking of Nose-to-Shoulder distance"""
        if len(self.pose_history) < 30: # Need at least 1 second
            return self.is_drowsy
            
        # Extract nose distances
        recent_poses = list(self.pose_history)
        distances = []
        for p in recent_poses:
            y_nose, y_shoulder = p
            if y_nose is not None and y_shoulder is not None:
                # Assuming top-left origin, shoulder Y > nose Y normally.
                # Distance represents how 'tall' the posture is
                dist = abs(y_shoulder - y_nose)
                distances.append(dist)
                
        if len(distances) < 15:
            return self.is_drowsy
            
        # Compute first derivative (velocity of distance change)
        velocities = np.diff(distances)
        
        # Look for a sharp compression (negative velocity) followed by expansion (positive)
        # We simplify by checking variance and extreme min/max shifts within 1 sec
        last_sec = distances[-30:] # Last 1 sec
        max_shift = max(last_sec) - min(last_sec)
        
        # If head drops drastically (> 30% of average length) rapidly
        avg_dist = np.mean(distances)
        if avg_dist > 0 and (max_shift / avg_dist) > 0.4:
            # We see a drastic nod!
            self.is_drowsy = True
        else:
            # Slowly decay drowsiness if not nodding
            self.is_drowsy = False
            
        return self.is_drowsy
        

class TemporalTrackerManager:
    """Manages profiles for all tracked students over time."""
    def __init__(self):
        self.profiles: Dict[int, StudentTemporalProfile] = {}
        
    def get_or_create_profile(self, track_id: int) -> StudentTemporalProfile:
        if track_id not in self.profiles:
            self.profiles[track_id] = StudentTemporalProfile(track_id=track_id)
        return self.profiles[track_id]
        
    def clean_stale_tracks(self, active_track_ids: List[int]):
        """Remove memory of track_ids that have been gone for a long time. 
        (Ideally, logic should tie them to student_id so history persists across tracking loss)."""
        # For this hackathon scope, we keep them in memory to prevent data loss if tracker blinks.
        pass
