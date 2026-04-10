"""
YOLO-Based Engagement Detector
Uses YOLOv8 Pose for posture and attention detection
Optimized for CPU performance
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class EngagementData:
    """Engagement detection result"""
    bbox: List[int]  # [x, y, w, h]
    track_id: int
    student_id: int = None
    
    # Engagement metrics
    engagement_score: float = 0.5
    posture_state: str = "unknown"  # upright, slouched, laying
    gaze_direction: str = "unknown"  # front, left, right, down
    phone_detected: bool = False
    head_visible: bool = True
    
    # Raw data
    keypoints: np.ndarray = None
    confidence: float = 0.0


class YOLOEngagementDetector:
    """
    YOLO-based engagement detection system
    Combines: Person detection + Pose estimation + Phone detection
    """
    
    def __init__(self, model_path: str = "models/yolov8n.pt", 
                 pose_model_path: str = "models/yolov8n-pose.pt"):
        self.model = None
        self.pose_model = None
        self.model_path = model_path
        self.pose_model_path = pose_model_path
        
        print("🚀 Initializing YOLO Engagement Detector...")
    
    def load_models(self):
        """Load YOLO models (using nano for CPU speed)"""
        try:
            # Person/Phone detection model
            self.model = YOLO(self.model_path)
            self.model.fuse()  # Fuse layers for speed
            print(f"✅ Loaded YOLO detection model: {self.model_path}")
            
            # Pose estimation model
            self.pose_model = YOLO(self.pose_model_path)
            self.pose_model.fuse()
            print(f"✅ Loaded YOLO pose model: {self.pose_model_path}")
            
        except Exception as e:
            print(f"❌ Error loading YOLO models: {e}")
            raise
    
    def detect_persons_and_phones(self, frame: np.ndarray, conf: float = 0.25) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect persons and phones in frame (lowered confidence threshold for nano model phone sensitivity)
        
        Args:
            frame: Input BGR image
            conf: Confidence threshold
            
        Returns:
            (persons, phones) - lists of detections
        """
        # Run detection
        results = self.model(frame, conf=conf, classes=[0, 67], verbose=False)  # person=0, cell phone=67
        
        persons = []
        phones = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                    'confidence': confidence,
                    'class': cls
                }
                
                if cls == 0:  # Person
                    persons.append(detection)
                elif cls == 67:  # Cell phone
                    phones.append(detection)
        
        return persons, phones
    
    def detect_pose(self, frame: np.ndarray, person_bbox: List[int] = None) -> np.ndarray:
        """
        Detect pose keypoints
        
        Args:
            frame: Input image
            person_bbox: Optional person bounding box for ROI [x, y, w, h]
            
        Returns:
            Keypoints array (17, 3) - [x, y, confidence] or None
        """
        # If bbox provided, crop to ROI
        if person_bbox:
            x, y, w, h = person_bbox
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame
            x1, y1 = 0, 0
        
        # Run pose estimation
        results = self.pose_model(roi, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return None
        
        # Extract keypoints
        kpts = results[0].keypoints
        if kpts.xy.shape[0] == 0:
            return None
        
        keypoints_xy = kpts.xy[0].cpu().numpy()
        keypoints_conf = kpts.conf[0].cpu().numpy()
        
        # Adjust coordinates to full frame
        keypoints_xy[:, 0] += x1
        keypoints_xy[:, 1] += y1
        
        # Combine [x, y, confidence]
        keypoints = np.hstack([keypoints_xy, keypoints_conf.reshape(-1, 1)])
        
        return keypoints
    
    def analyze_posture(self, keypoints: np.ndarray) -> Tuple[str, float]:
        """
        Analyze posture from keypoints
        
        Returns:
            (posture_state, confidence)
            posture_state: 'upright', 'slouched', 'laying'
        """
        if keypoints is None or len(keypoints) < 17:
            return 'unknown', 0.0
        
        # COCO keypoint indices
        NOSE = 0
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        
        # Check visibility
        if keypoints[NOSE][2] < 0.3:
            return 'unknown', 0.0
            
        # Check head tilt for "sleeping" (resting head on arms/desk) even if shoulders are missing
        LEFT_EAR = 3
        RIGHT_EAR = 4
        if keypoints[LEFT_EAR][2] > 0.3 and keypoints[RIGHT_EAR][2] > 0.3:
            ear_y_diff = abs(keypoints[LEFT_EAR][1] - keypoints[RIGHT_EAR][1])
            ear_x_diff = abs(keypoints[LEFT_EAR][0] - keypoints[RIGHT_EAR][0])
            # If head is heavily tilted sideways (y diff is large compared to x diff)
            if ear_x_diff > 0 and (ear_y_diff / ear_x_diff) > 0.6:
                return 'laying', 0.85
                
        # If shoulders are missing (cut off by frame), but face is visible, fallback to upright
        if keypoints[LEFT_SHOULDER][2] < 0.2 and keypoints[RIGHT_SHOULDER][2] < 0.2:
            return 'upright', 0.5
        
        # Calculate centers
        shoulder_center_y = (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2
        hip_center_y = (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2 if keypoints[LEFT_HIP][2] > 0.3 else shoulder_center_y + 100
        
        # Spine length (vertical distance)
        spine_length = abs(shoulder_center_y - hip_center_y)
        
        # Head-shoulder distance
        head_shoulder_dist = abs(keypoints[NOSE][1] - shoulder_center_y)
        
        # Determine posture
        if spine_length < 50:  # Very compressed spine
            return 'laying', 0.8
        
        if head_shoulder_dist < spine_length * 0.3:  # Head too low
            return 'slouched', 0.7
        
        return 'upright', 0.85
    
    def analyze_gaze(self, keypoints: np.ndarray) -> str:
        """
        Estimate gaze direction from head orientation
        
        Returns:
            'front', 'left', 'right', 'down'
        """
        if keypoints is None or len(keypoints) < 17:
            return 'unknown'
        
        NOSE = 0
        LEFT_EAR = 3
        RIGHT_EAR = 4
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        
        # Check visibility
        if keypoints[NOSE][2] < 0.3:
            return 'unknown'
        
        # Shoulder center
        shoulder_center_x = (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2
        
        # Horizontal offset of nose
        nose_offset = keypoints[NOSE][0] - shoulder_center_x
        
        # Ear visibility
        left_ear_visible = keypoints[LEFT_EAR][2] > 0.3
        right_ear_visible = keypoints[RIGHT_EAR][2] > 0.3
        
        # Determine gaze
        if abs(nose_offset) < 30 and left_ear_visible and right_ear_visible:
            return 'front'
        elif nose_offset < -40 or (left_ear_visible and not right_ear_visible):
            return 'left'
        elif nose_offset > 40 or (right_ear_visible and not left_ear_visible):
            return 'right'
        
        # Check if looking down (nose lower than ears)
        nose_y = keypoints[NOSE][1]
        eyes_y = (keypoints[1][1] + keypoints[2][1]) / 2 if (keypoints[1][2] > 0.3 and keypoints[2][2] > 0.3) else keypoints[NOSE][1] - 20
        ears_y = (keypoints[LEFT_EAR][1] + keypoints[RIGHT_EAR][1]) / 2 if (left_ear_visible and right_ear_visible) else eyes_y
        
        # If nose is pushed down significantly compared to the average eye/ear line
        if nose_y > ears_y + 40:
            return 'down'
            
        return 'front'
    
    def check_phone_usage(self, person_bbox: List[int], phones: List[Dict]) -> bool:
        """
        Check if person is using phone (proximity-based)
        
        Args:
            person_bbox: Person bounding box [x, y, w, h]
            phones: List of phone detections
            
        Returns:
            True if phone detected near person
        """
        if not phones:
            return False
        
        px, py, pw, ph = person_bbox
        person_center_x = px + pw / 2
        person_center_y = py + ph / 2
        
        # Check proximity to any phone
        for phone in phones:
            phone_x, phone_y, phone_w, phone_h = phone['bbox']
            phone_center_x = phone_x + phone_w / 2
            phone_center_y = phone_y + phone_h / 2
            
            # Distance threshold
            dist = np.sqrt((person_center_x - phone_center_x)**2 + (person_center_y - phone_center_y)**2)
            
            # Phone within person's general workspace (relaxed to allow phones held up)
            if dist < (pw * 1.5):
                return True
        
        return False
    
    def compute_engagement_score(self, posture: str, gaze: str, phone_detected: bool) -> float:
        """
        Compute engagement score
        
        Returns:
            Score between 0.0 and 1.0
        """
        # Base scores
        posture_scores = {
            'upright': 1.0,
            'slouched': 0.5,
            'laying': 0.0,
            'unknown': 0.6
        }
        
        gaze_scores = {
            'front': 1.0,
            'down': 0.4,
            'left': 0.3,
            'right': 0.3,
            'unknown': 0.5
        }
        
        # Weighted combination
        score = (
            0.4 * posture_scores.get(posture, 0.5) +
            0.4 * gaze_scores.get(gaze, 0.5) +
            0.2 * (0.0 if phone_detected else 1.0)
        )
        
        return max(0.0, min(1.0, score))
    
    def analyze_frame(self, frame: np.ndarray) -> List[EngagementData]:
        """
        Complete engagement analysis pipeline
        
        Args:
            frame: Input BGR image
            
        Returns:
            List of EngagementData objects
        """
        # Detect persons and phones
        persons, phones = self.detect_persons_and_phones(frame)
        
        engagement_results = []
        
        for idx, person in enumerate(persons):
            bbox = person['bbox']
            
            # Detect pose
            keypoints = self.detect_pose(frame, bbox)
            
            # Analyze posture and gaze
            posture, posture_conf = self.analyze_posture(keypoints)
            gaze = self.analyze_gaze(keypoints)
            
            # Check phone usage
            phone_detected = self.check_phone_usage(bbox, phones)
            
            # Compute engagement score
            engagement_score = self.compute_engagement_score(posture, gaze, phone_detected)
            
            # Create engagement data
            engagement = EngagementData(
                bbox=bbox,
                track_id=idx,  # Will be replaced by actual tracking ID
                engagement_score=engagement_score,
                posture_state=posture,
                gaze_direction=gaze,
                phone_detected=phone_detected,
                head_visible=keypoints is not None,
                keypoints=keypoints,
                confidence=person['confidence']
            )
            
            engagement_results.append(engagement)
        
        return engagement_results


# Global instance
engagement_detector = YOLOEngagementDetector()
