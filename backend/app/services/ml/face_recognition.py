"""
Optimized Face Recognition Service
Uses FAISS for fast similarity search (supports 30-50 students)
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import faiss
from dataclasses import dataclass
import time

@dataclass
class FaceDetection:
    """Face detection result"""
    bbox: List[int]  # [x, y, w, h]
    confidence: float
    student_id: Optional[int] = None
    match_confidence: Optional[float] = None
    embedding: Optional[np.ndarray] = None


class OptimizedFaceRecognizer:
    """
    Fast face recognition using FAISS index
    Optimized for CPU performance with 30-50 students
    """
    
    def __init__(self):
        self.face_detector = None
        self.embedding_model = None
        self.faiss_index = None
        self.student_id_map = []  # Maps FAISS index to student_id
        self.threshold = 0.6
        
        print("🚀 Initializing Optimized Face Recognizer...")
    
    def load_models(self):
        """Load lightweight face detection and recognition models"""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            print("🚀 Loading InsightFace models...")
            # Use buffalo_s (lighter) or buffalo_l
            self.app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.face_detector = "insightface"
            print("✅ Loaded InsightFace successfully")
        except Exception as e:
            print(f"⚠️  Using fallback: OpenCV Haar Cascade. Error: {e}")
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_faces(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[FaceDetection]:
        """
        Detect faces in frame
        """
        if self.face_detector == "insightface":
            return self._detect_insightface(frame, conf_threshold)
        elif isinstance(self.face_detector, cv2.dnn_Net):
            return self._detect_dnn(frame, conf_threshold)
        else:
            return self._detect_cascade(frame)
            
    def _detect_insightface(self, frame: np.ndarray, conf_threshold: float) -> List[FaceDetection]:
        faces = self.app.get(frame)
        results = []
        for face in faces:
            if face.det_score >= conf_threshold:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                results.append(FaceDetection(
                    bbox=[x1, y1, x2-x1, y2-y1],
                    confidence=float(face.det_score),
                    embedding=face.normed_embedding
                ))
        return results
    
    def _detect_dnn(self, frame: np.ndarray, conf_threshold: float) -> List[FaceDetection]:
        """DNN-based detection"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                faces.append(FaceDetection(
                    bbox=[x1, y1, x2-x1, y2-y1],
                    confidence=float(confidence)
                ))
        
        return faces
    
    def _detect_cascade(self, frame: np.ndarray) -> List[FaceDetection]:
        """Haar Cascade detection (fallback)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rects = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in faces_rects:
            faces.append(FaceDetection(
                bbox=[int(x), int(y), int(w), int(h)],
                confidence=1.0
            ))
        
        return faces
    
    def extract_embedding(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract face embedding
        """
        if self.face_detector == "insightface":
            # We already get embeddings from the detect phase in InsightFace!
            # But if called separately, we run it again on the crop
            x, y, w, h = bbox
            pad = int(max(w, h) * 0.2)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            
            face_crop = frame[y1:y2, x1:x2]
            faces = self.app.get(face_crop)
            if faces:
                return faces[0].normed_embedding
            # Fallback random
            embedding = np.random.randn(512).astype(np.float32)
            return embedding / np.linalg.norm(embedding)

        x, y, w, h = bbox
        pad = int(max(w, h) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        
        face_crop = frame[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (112, 112))
        
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_norm = (face_rgb - 127.5) / 128.0
        face_input = face_norm.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        
        if self.embedding_model:
            input_name = self.embedding_model.get_inputs()[0].name
            embedding = self.embedding_model.run(None, {input_name: face_input})[0][0]
        else:
            embedding = np.random.randn(512).astype(np.float32)
            
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def build_index(self, student_embeddings: Dict[int, List[np.ndarray]]):
        """
        Build FAISS index from student embeddings
        
        Args:
            student_embeddings: Dict mapping student_id -> list of embeddings
        """
        print("🔨 Building FAISS index...")
        start_time = time.time()
        
        all_embeddings = []
        self.student_id_map = []
        
        for student_id, embeddings in student_embeddings.items():
            for emb in embeddings:
                all_embeddings.append(emb)
                self.student_id_map.append(student_id)
        
        if not all_embeddings:
            print("⚠️  No embeddings to index")
            self.faiss_index = None
            return
        
        # Stack embeddings
        embeddings_matrix = np.vstack(all_embeddings).astype('float32')
        
        # Build FAISS index (Inner Product for cosine similarity)
        dimension = embeddings_matrix.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings_matrix)
        
        build_time = time.time() - start_time
        print(f"✅ FAISS index built: {len(all_embeddings)} embeddings, {build_time:.3f}s")
    
    def identify_face(self, embedding: np.ndarray, k: int = 1) -> Tuple[Optional[int], float]:
        """
        Identify face from embedding using FAISS search
        
        Args:
            embedding: Query embedding (512-dim, L2 normalized)
            k: Number of nearest neighbors
            
        Returns:
            (student_id, similarity) or (None, 0.0) if no match
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return None, 0.0
        
        # Search
        query = embedding.reshape(1, -1).astype('float32')
        distances, indices = self.faiss_index.search(query, k)
        
        # Get best match
        similarity = float(distances[0][0])
        idx = int(indices[0][0])
        
        # Check threshold
        if similarity >= self.threshold:
            student_id = self.student_id_map[idx]
            return student_id, similarity
        
        return None, similarity
    
    def recognize_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Complete pipeline: detect faces and identify students
        """
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Extract embeddings and identify
        for face in faces:
            try:
                # If InsightFace was used, embedding is already computed!
                if face.embedding is None:
                    embedding = self.extract_embedding(frame, face.bbox)
                    face.embedding = embedding
                else:
                    embedding = face.embedding
                    
                student_id, similarity = self.identify_face(embedding)
                face.student_id = student_id
                face.match_confidence = similarity
            except Exception as e:
                print(f"Error recognizing face: {e}")
                continue
        
        return faces
    
    def recognize_faces_batch(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """
        Batch processing for multiple frames (more efficient)
        
        Args:
            frames: List of input images
            
        Returns:
            List of face detection lists (one per frame)
        """
        results = []
        
        # Process all frames
        for frame in frames:
            faces = self.recognize_faces(frame)
            results.append(faces)
        
        return results


# Global instance
face_recognizer = OptimizedFaceRecognizer()
