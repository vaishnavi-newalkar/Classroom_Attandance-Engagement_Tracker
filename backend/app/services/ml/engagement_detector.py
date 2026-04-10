"""
YOLO-Based Engagement Detector (Hybrid Pipeline)
Phase 1: Pure YOLO kinematics for all students (0 overhead)
Phase 2: Conditional MediaPipe FaceMesh (only for large, high-res face crops)

Produces a rich per-student behavioral profile including:
- Continuous attention score (0–100, EMA-smoothed)
- Posture state (upright, slouched, laying)
- Gaze direction (front, left, right, down)
- Phone usage (temporal debounced)
- Drowsiness (head-nod kinematics + optional EAR confirmation)
- Yawning (conditional MAR check, throttled to ~1 FPS)
- Sleep (bounding-box-variance + optional EAR confirmation)
"""

import time
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from ultralytics import YOLO

# Conditional import — graceful fallback to YOLO-only if mediapipe is missing
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("⚠️  MediaPipe not found — running YOLO-only pipeline.")

# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EngagementData:
    """Rich engagement detection result per student per frame."""
    bbox: List[int]          # [x, y, w, h]
    track_id: int
    student_id: Optional[int] = None

    # ── Core Engagement ──
    engagement_score:  float = 0.5   # 0.0–1.0 (used externally)
    attention_score:   float = 50.0  # 0–100, EMA-smoothed
    engagement_state:  str   = "unknown"  # attentive | mildly_distracted | highly_distracted | sleeping

    # ── Per-Signal Outputs ──
    posture_state:    str  = "unknown"   # upright | slouched | laying
    gaze_direction:   str  = "unknown"   # front | left | right | down
    phone_detected:   bool = False
    head_visible:     bool = True

    # ── Behavioural Events ──
    is_sleeping:  bool = False
    is_drowsy:    bool = False
    is_yawning:   bool = False
    yawn_count:   int  = 0
    phone_time_sec: float = 0.0

    # ── Raw Data ──
    keypoints:    Optional[np.ndarray] = None
    confidence:   float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid Engagement Detector
# ──────────────────────────────────────────────────────────────────────────────

class YOLOEngagementDetector:
    """
    Hybrid YOLO + (conditional) MediaPipe engagement system.
    """

    # MediaPipe face-mesh landmark IDs we care about
    _LEFT_EYE_TOP    = 159
    _LEFT_EYE_BOT    = 145
    _LEFT_EYE_LEFT   = 33
    _LEFT_EYE_RIGHT  = 133
    _RIGHT_EYE_TOP   = 386
    _RIGHT_EYE_BOT   = 374
    _RIGHT_EYE_LEFT  = 362
    _RIGHT_EYE_RIGHT = 263
    _MOUTH_TOP       = 13
    _MOUTH_BOT       = 14
    _MOUTH_LEFT      = 61
    _MOUTH_RIGHT     = 291

    # Minimum face crop area (px²) to run MediaPipe — below this resolution is too blurry
    _MIN_FACE_AREA_FOR_MP = 1000  # ~32×32 px — works for medium-to-close students
    _MP_INTERVAL = 0.3  # seconds between MediaPipe checks per student (was 1.0s)

    def __init__(self,
                 model_path: str = "models/yolov8n.pt",
                 pose_model_path: str = "models/yolov8n-pose.pt"):
        self.model = None
        self.pose_model = None
        self.model_path = model_path
        self.pose_model_path = pose_model_path

        # MediaPipe instance — shared, lazy-initialised
        self._mp_face_mesh = None
        self._mp_hands = None

        # Per-track-id MediaPipe throttle counters
        # {track_id: last_mp_check_time}
        self._mp_last_check: Dict[int, float] = {}

        print("🚀 Initialising Hybrid Engagement Detector...")

    # ──────────────────────────────────────────────────────────────────────────
    # Model Loading
    # ──────────────────────────────────────────────────────────────────────────

    def load_models(self):
        """Load YOLO models (nano for CPU speed). MediaPipe is lazy-loaded."""
        try:
            self.model = YOLO(self.model_path)
            self.model.fuse()
            print(f"✅ YOLO detection model: {self.model_path}")

            self.pose_model = YOLO(self.pose_model_path)
            self.pose_model.fuse()
            print(f"✅ YOLO pose model: {self.pose_model_path}")

        except Exception as e:
            print(f"❌ Error loading YOLO models: {e}")
            raise

    def _get_face_mesh(self):
        """Lazy-initialise a MediaPipe FaceMesh for single-face crops."""
        if self._mp_face_mesh is None and _MP_AVAILABLE:
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,   # used on still crops — no tracking state needed
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.4,
            )
        return self._mp_face_mesh

    # ──────────────────────────────────────────────────────────────────────────
    # YOLO Detection Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def detect_persons_and_phones(self, frame: np.ndarray,
                                   conf: float = 0.15) -> Tuple[List[Dict], List[Dict]]:
        # Lowered conf to 0.15 to heavily recall cell phones (class 67), even if blurry or held tight.
        results = self.model(frame, conf=conf, classes=[0, 67], verbose=False)
        persons, phones = [], []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls  = int(box.cls[0])
                det  = {
                    'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    'confidence': float(box.conf[0]),
                    'class': cls
                }
                (persons if cls == 0 else phones).append(det)
        return persons, phones

    def detect_pose(self, frame: np.ndarray,
                    person_bbox: Optional[List[int]] = None) -> Optional[np.ndarray]:
        """Returns (17, 3) keypoints [x, y, confidence] or None."""
        if person_bbox:
            x, y, w, h = person_bbox
            pad = int(max(w, h) * 0.1)
            x1 = max(0, x - pad); y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame; x1 = y1 = 0

        results = self.pose_model(roi, verbose=False)
        if not results or results[0].keypoints is None:
            return None

        kpts = results[0].keypoints
        if kpts.xy.shape[0] == 0:
            return None

        kxy  = kpts.xy[0].cpu().numpy()
        kconf = kpts.conf[0].cpu().numpy()
        kxy[:, 0] += x1
        kxy[:, 1] += y1
        return np.hstack([kxy, kconf.reshape(-1, 1)])

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 1: YOLO Kinematic Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_posture(self, keypoints: np.ndarray) -> Tuple[str, float]:
        """Classify posture: upright | slouched | laying"""
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 5, 6
        LEFT_HIP, RIGHT_HIP = 11, 12
        LEFT_EAR, RIGHT_EAR = 3, 4

        if keypoints is None or len(keypoints) < 17:
            return "unknown", 0.0
        if keypoints[NOSE][2] < 0.3:
            return "unknown", 0.0

        # Head tilt → laying (extreme sideways ear Y-axis separation)
        if keypoints[LEFT_EAR][2] > 0.3 and keypoints[RIGHT_EAR][2] > 0.3:
            ear_y = abs(keypoints[LEFT_EAR][1] - keypoints[RIGHT_EAR][1])
            ear_x = abs(keypoints[LEFT_EAR][0] - keypoints[RIGHT_EAR][0])
            # require ratio > 1.0 (45 degree tilt or more) to trigger laying
            if ear_x > 0 and (ear_y / ear_x) > 1.0:
                return "laying", 0.9

        if keypoints[LEFT_SHOULDER][2] < 0.2 and keypoints[RIGHT_SHOULDER][2] < 0.2:
            return "upright", 0.5   # shoulders not visible but face is — assume seated

        shoulder_cy = (keypoints[LEFT_SHOULDER][1] + keypoints[RIGHT_SHOULDER][1]) / 2
        hip_cy = (
            (keypoints[LEFT_HIP][1] + keypoints[RIGHT_HIP][1]) / 2
            if keypoints[LEFT_HIP][2] > 0.3 and keypoints[RIGHT_HIP][2] > 0.3
            else shoulder_cy + 100
        )
        spine_len = abs(shoulder_cy - hip_cy)
        head_gap  = abs(keypoints[NOSE][1] - shoulder_cy)

        if spine_len < 50:
            return "laying", 0.8
        if head_gap < spine_len * 0.3:
            return "slouched", 0.75
        return "upright", 0.9

    def analyze_gaze(self, keypoints: np.ndarray) -> str:
        """Estimate gaze: front | left | right | down"""
        NOSE, LEFT_EAR, RIGHT_EAR = 0, 3, 4
        LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

        if keypoints is None or len(keypoints) < 17:
            return "unknown"
        if keypoints[NOSE][2] < 0.3:
            return "unknown"

        l_ear = keypoints[LEFT_EAR][2]  > 0.3
        r_ear = keypoints[RIGHT_EAR][2] > 0.3

        # Lateral: use relative nose–shoulder offset
        if keypoints[LEFT_SHOULDER][2] > 0.2 and keypoints[RIGHT_SHOULDER][2] > 0.2:
            sc_x = (keypoints[LEFT_SHOULDER][0] + keypoints[RIGHT_SHOULDER][0]) / 2
            nose_off = keypoints[NOSE][0] - sc_x
            if   nose_off < -40 or (l_ear and not r_ear): return "left"
            elif nose_off >  40 or (r_ear and not l_ear): return "right"

        # Downward: nose significantly below ear baseline
        if l_ear and r_ear:
            ears_y = (keypoints[LEFT_EAR][1] + keypoints[RIGHT_EAR][1]) / 2
            if keypoints[NOSE][1] > ears_y + 40:
                return "down"

        return "front"

    def check_phone_usage(self, person_bbox: List[int],
                          phones: List[Dict]) -> bool:
        """Proximity-based phone association."""
        if not phones:
            return False
        px, py, pw, ph = person_bbox
        pcx, pcy = px + pw / 2, py + ph / 2
        for phone in phones:
            qx, qy, qw, qh = phone['bbox']
            qcx, qcy = qx + qw / 2, qy + qh / 2
            if np.hypot(pcx - qcx, pcy - qcy) < pw * 1.5:
                return True
        return False

    def compute_raw_attention(self, posture: str, gaze: str,
                               phone_detected: bool,
                               is_sleeping: bool, is_drowsy: bool) -> float:
        """
        Compute raw attention score (0–100).

        Weights:
          Posture     40
          Gaze        40
          Phone       -30 penalty
          Sleeping    -60 penalty
          Drowsy      -20 penalty
        """
        posture_scores = {"upright": 40.0, "slouched": 20.0, "laying": 0.0, "unknown": 24.0}
        gaze_scores    = {"front": 40.0, "down": 16.0, "left": 12.0, "right": 12.0, "unknown": 20.0}

        score = posture_scores.get(posture, 24.0) + gaze_scores.get(gaze, 20.0)

        if phone_detected: score -= 30.0
        if is_sleeping:    score -= 60.0
        if is_drowsy:      score -= 20.0

        return max(0.0, min(100.0, score))

    @staticmethod
    def attention_state(score: float) -> str:
        if   score >= 80: return "attentive"
        elif score >= 55: return "mildly_distracted"
        elif score >= 30: return "highly_distracted"
        else:             return "sleeping"

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 2: Conditional MediaPipe Checks
    # ──────────────────────────────────────────────────────────────────────────

    def _face_crop_from_bbox(self, frame: np.ndarray, person_bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract face region from the top 40% of the person bounding box.
        This is far more reliable than relying on keypoint precision in
        wide-angle classroom shots, and works even when nose/ear confidence is low.
        """
        x, y, w, h = person_bbox
        face_h = max(40, int(h * 0.42))  # top 42% of bbox = head region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + face_h)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _face_crop_from_keypoints(self, frame: np.ndarray,
                                   keypoints: np.ndarray,
                                   person_bbox: List[int]) -> Optional[np.ndarray]:
        """Extract face crop from YOLO nose/ear keypoints, padded generously."""
        NOSE, L_EAR, R_EAR, L_EYE, R_EYE = 0, 3, 4, 1, 2

        pts = []
        for idx in [NOSE, L_EAR, R_EAR, L_EYE, R_EYE]:
            if keypoints[idx][2] > 0.3:
                pts.append((int(keypoints[idx][0]), int(keypoints[idx][1])))

        if len(pts) < 2:
            return None

        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        pad_x = max(30, (max(xs) - min(xs)) // 2)
        pad_y = max(40, (max(ys) - min(ys)) // 2)

        x1 = max(0, min(xs) - pad_x)
        y1 = max(0, min(ys) - pad_y)
        x2 = min(frame.shape[1], max(xs) + pad_x)
        y2 = min(frame.shape[0], max(ys) + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _compute_ear(self, lm, eye_top: int, eye_bot: int,
                     eye_left: int, eye_right: int) -> float:
        """Eye Aspect Ratio from MediaPipe landmarks."""
        h = abs(lm[eye_top].y - lm[eye_bot].y)
        w = abs(lm[eye_left].x - lm[eye_right].x) + 1e-6
        return h / w

    def _compute_mar(self, lm) -> float:
        """Mouth Aspect Ratio from MediaPipe landmarks."""
        v = abs(lm[self._MOUTH_TOP].y - lm[self._MOUTH_BOT].y)
        h = abs(lm[self._MOUTH_LEFT].x - lm[self._MOUTH_RIGHT].x) + 1e-6
        return v / h

    def _run_mediapipe_check(self, frame: np.ndarray,
                              keypoints: Optional[np.ndarray],
                              person_bbox: List[int],
                              posture: str) -> Dict:
        """
        Run MediaPipe on a single high-res face crop.
        Returns dict with: ear, mar, confirmed_eyes_closed, is_yawning
        """
        result = {"ear": None, "mar": None, "confirmed_sleep": False, "is_yawning": False}

        face_mesh = self._get_face_mesh()
        if face_mesh is None:
            return result

        # Prefer simple bbox crop (more reliable in classroom shots)
        crop = self._face_crop_from_bbox(frame, person_bbox)
        if crop is None:
            return result
        
        face_area = crop.shape[0] * crop.shape[1]
        if face_area < self._MIN_FACE_AREA_FOR_MP:
            return result

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_result = face_mesh.process(rgb)

        if not mp_result.multi_face_landmarks:
            return result

        lm = mp_result.multi_face_landmarks[0].landmark

        # EAR — average of both eyes
        left_ear  = self._compute_ear(lm, self._LEFT_EYE_TOP, self._LEFT_EYE_BOT,
                                       self._LEFT_EYE_LEFT, self._LEFT_EYE_RIGHT)
        right_ear = self._compute_ear(lm, self._RIGHT_EYE_TOP, self._RIGHT_EYE_BOT,
                                       self._RIGHT_EYE_LEFT, self._RIGHT_EYE_RIGHT)
        ear = (left_ear + right_ear) / 2.0
        result["ear"] = ear

        # Confirmed closed eyes → boost sleep confidence
        if ear < 0.18 and posture in ("laying", "slouched"):
            result["confirmed_sleep"] = True

        # MAR — Yawning (lowered threshold from 0.55 to 0.35 for better sensitivity)
        mar = self._compute_mar(lm)
        result["mar"] = mar
        if mar > 0.35:
            result["is_yawning"] = True

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Main Analysis Entry Point
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_frame(self, frame: np.ndarray,
                      temporal_profiles: Optional[Dict] = None) -> List[EngagementData]:
        """
        Complete hybrid engagement analysis pipeline.

        Args:
            frame:             BGR image
            temporal_profiles: optional dict {track_id -> StudentTemporalProfile}
                               — if supplied, MediaPipe throttling + EMA is applied
        Returns:
            List[EngagementData]
        """
        persons, phones = self.detect_persons_and_phones(frame)
        results = []
        now = time.time()

        for idx, person in enumerate(persons):
            bbox      = person['bbox']
            track_id  = idx  # Replaced by real tracker ID from caller

            # ── Phase 1: YOLO kinematics ──────────────────────────────────
            keypoints = self.detect_pose(frame, bbox)
            posture, _ = self.analyze_posture(keypoints)
            gaze       = self.analyze_gaze(keypoints)
            phone_raw  = self.check_phone_usage(bbox, phones)

            # ── Temporal state from profile (if provided) ─────────────────
            profile = None
            if temporal_profiles is not None:
                profile = temporal_profiles.get(track_id)

            # ── Phase 1: Sleep/Drowsiness via YOLO kinematics ────────────
            # Direct posture-based sleep — no need for 90 frames of history
            is_sleeping = (posture == "laying")
            is_drowsy   = False
            is_yawning  = False
            yawn_count  = 0
            phone_time  = 0.0

            if profile is not None:
                # Feed bbox centre for variance tracking (refines sleep over time)
                bx, by, bw, bh = bbox
                profile.bbox_center_history.append((bx + bw / 2, by + bh / 2))

                # Feed pose data for nod detection
                nose_y     = keypoints[0][1] if keypoints is not None and keypoints[0][2] > 0.3 else None
                sh_y_l     = keypoints[5][1] if keypoints is not None and keypoints[5][2] > 0.2 else None
                sh_y_r     = keypoints[6][1] if keypoints is not None and keypoints[6][2] > 0.2 else None
                shoulder_y = (sh_y_l + sh_y_r) / 2 if sh_y_l and sh_y_r else (sh_y_l or sh_y_r)
                profile.pose_history.append((nose_y, shoulder_y))

                # Gaze-down flag for sleep detection
                profile.gaze_down_history.append(1 if gaze == "down" else 0)

                # Phone (temporal debounce)
                profile.evaluate_phone_usage(phone_raw)
                phone_time = profile.total_phone_time_sec

                # Refine sleep with variance check (overrides direct posture if they’re just looking down and writing)
                # Ensure they aren't marked sleeping if they are actively using a phone!
                sleep_from_variance = profile.check_static_sleep()
                if sleep_from_variance and not profile.is_using_phone:
                    is_sleeping = True
                
                # If they are actively using a phone, force cancel sleep
                if profile.is_using_phone or phone_raw:
                    is_sleeping = False
                    
                is_drowsy   = profile.check_head_nod_drowsiness()
                yawn_count  = profile.total_yawns
                phone_time  = profile.total_phone_time_sec

            # ── Phase 2: Conditional MediaPipe ────────────────────────────
            bx, by, bw, bh = bbox
            face_area_est = bw * int(bh * 0.42)  # estimated face sub-region
            run_mp    = (
                _MP_AVAILABLE
                and face_area_est >= self._MIN_FACE_AREA_FOR_MP
                and (now - self._mp_last_check.get(track_id, 0)) >= self._MP_INTERVAL
            )

            if run_mp:
                self._mp_last_check[track_id] = now
                mp_out = self._run_mediapipe_check(frame, keypoints, bbox, posture)

                # Override sleep with EAR confirmation
                if mp_out["confirmed_sleep"]:
                    is_sleeping = True
                elif profile and is_sleeping and mp_out["ear"] is not None and mp_out["ear"] > 0.22:
                    # Eyes clearly open → probably reading/writing, not sleeping
                    is_sleeping = False

                # Yawn detection (with cooldown)
                if mp_out["is_yawning"]:
                    is_yawning = True
                    if profile and (now - profile.last_yawn_timestamp > 5.0):
                        profile.total_yawns += 1
                        profile.last_yawn_timestamp = now
                    if profile:
                        yawn_count = profile.total_yawns

            # ── Attention Score ───────────────────────────────────────────
            raw_attention = self.compute_raw_attention(posture, gaze, phone_raw or (profile.is_using_phone if profile else False), is_sleeping, is_drowsy)

            if profile is not None:
                smoothed = profile.update_attention(raw_attention)
            else:
                smoothed = raw_attention

            state = self.attention_state(smoothed)

            # ── Build EngagementData ──────────────────────────────────────
            eng = EngagementData(
                bbox              = bbox,
                track_id          = track_id,
                engagement_score  = round(smoothed / 100.0, 4),  # 0–1 for DB compatibility
                attention_score   = round(smoothed, 2),
                engagement_state  = state,
                posture_state     = posture,
                gaze_direction    = gaze,
                phone_detected    = profile.is_using_phone if profile else phone_raw,
                head_visible      = keypoints is not None,
                is_sleeping       = is_sleeping,
                is_drowsy         = is_drowsy,
                is_yawning        = is_yawning,
                yawn_count        = yawn_count,
                phone_time_sec    = phone_time,
                keypoints         = keypoints,
                confidence        = person['confidence'],
            )
            results.append(eng)

        return results


# Global instance
engagement_detector = YOLOEngagementDetector()
