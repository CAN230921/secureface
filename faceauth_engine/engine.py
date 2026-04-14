from __future__ import annotations

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, EngineConfig
from .database import FaceDatabase
from .detector import FaceDetector
from .embedder import MobileFaceNet
from .liveness import LivenessDetector


class FaceRecognitionSystem:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.detector = FaceDetector(config)
        self.liveness_detector = LivenessDetector(config) if config.liveness_check else None
        self.extractor = MobileFaceNet(config)
        self.db = FaceDatabase(config)

    @staticmethod
    def _check_face_quality(face_img: np.ndarray) -> float:
        gray = cv2.cvtColor(((face_img + 1) * 127.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        clarity = min(laplacian_var / 300, 1.0)
        mean_brightness = float(np.mean(gray))
        brightness = 1.0 - abs(mean_brightness - 127) / 127
        contrast = min(float(np.std(gray)) / 40, 1.0)
        return clarity * 0.5 + brightness * 0.3 + contrast * 0.2

    @staticmethod
    def _analyze_features(embeddings: list[np.ndarray], outlier_threshold: float) -> np.ndarray | None:
        if len(embeddings) == 0:
            return None
        if len(embeddings) == 1:
            return embeddings[0]

        emb_array = np.array(embeddings)
        sim_matrix = np.dot(emb_array, emb_array.T)

        avg_sims = np.mean(sim_matrix, axis=1)
        center_idx = int(np.argmax(avg_sims))
        center_sim = sim_matrix[center_idx]

        valid_mask = center_sim > outlier_threshold
        valid_embeddings = emb_array[valid_mask]
        if len(valid_embeddings) == 0:
            median_emb = np.median(emb_array, axis=0)
            norm = np.linalg.norm(median_emb)
            return median_emb / norm if norm > 0 else median_emb

        weights = center_sim[valid_mask]
        weights = weights / np.sum(weights)
        weighted_emb = np.average(valid_embeddings, axis=0, weights=weights)

        norm = np.linalg.norm(weighted_emb)
        return weighted_emb / norm if norm > 0 else weighted_emb

    def enroll(self, name: str, cam: int | None = None) -> bool:
        if not self.extractor.is_loaded():
            return False

        cap = cv2.VideoCapture(self.config.camera_index if cam is None else cam)
        if not cap.isOpened():
            return False

        target_samples = self.config.frames_per_angle * len(self.config.enroll_states)
        embeddings: list[np.ndarray] = []

        try:
            max_frames = max(target_samples * 4, target_samples)
            for _ in range(max_frames):
                if len(embeddings) >= target_samples:
                    break

                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                faces = self.detector.detect(frame)
                face = self.detector.select_largest(faces)
                if face is None:
                    continue
                if face.confidence < self.config.face_detection_confidence:
                    continue

                aligned = self.detector.align_face(frame, face)
                if aligned is None:
                    continue

                quality_score = self._check_face_quality(aligned)
                if quality_score < self.config.min_quality_threshold:
                    continue

                if self.config.liveness_check and self.liveness_detector is not None:
                    is_live, _ = self.liveness_detector.check(aligned)
                    if not is_live:
                        continue

                emb = self.extractor.extract(aligned)
                if emb is not None:
                    embeddings.append(emb)

            if len(embeddings) < int(target_samples * 0.4):
                return False

            final_emb = self._analyze_features(embeddings, self.config.outlier_threshold)
            if final_emb is None:
                return False

            return self.db.add(name, final_emb)
        finally:
            cap.release()

    def authenticate(self, cam: int | None = None) -> str:
        if not self.extractor.is_loaded():
            return "ERROR"
        if not self.db.faces:
            return "UNKNOWN"

        cap = cv2.VideoCapture(self.config.camera_index if cam is None else cam)
        if not cap.isOpened():
            return "ERROR"

        saw_face = False
        best_score = -1.0

        try:
            for _ in range(self.config.auth_frames):
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                faces = self.detector.detect(frame)
                face = self.detector.select_largest(faces)
                if face is None:
                    continue
                saw_face = True

                aligned = self.detector.align_face(frame, face)
                if aligned is None:
                    continue

                if self.config.liveness_check and self.liveness_detector is not None:
                    is_live, _ = self.liveness_detector.check(aligned)
                    if not is_live:
                        return "NOT_LIVE"

                emb = self.extractor.extract(aligned)
                if emb is None:
                    continue

                user, score = self.db.verify(emb, self.config.verification_threshold)
                if user is not None:
                    best_score = max(best_score, score)

            if best_score >= self.config.verification_threshold:
                return "PASS"
            if saw_face:
                return "UNKNOWN"
            return "UNKNOWN"
        except Exception:
            return "ERROR"
        finally:
            cap.release()


class FaceAuthEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self.system = FaceRecognitionSystem(self.config)

    def enroll(self, name: str) -> bool:
        try:
            return self.system.enroll(name)
        except Exception:
            return False

    def authenticate(self) -> str:
        try:
            result = self.system.authenticate()
        except Exception:
            return "ERROR"

        return result if result in {"PASS", "UNKNOWN", "NOT_LIVE", "ERROR"} else "ERROR"
