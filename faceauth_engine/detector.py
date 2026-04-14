from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from .config import EngineConfig


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class FaceDetector:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.net = None
        self.cascade = None
        self._load_model()

    def _load_model(self) -> None:
        if os.path.exists(self.config.prototxt) and os.path.exists(self.config.caffemodel):
            self.net = cv2.dnn.readNetFromCaffe(self.config.prototxt, self.config.caffemodel)
        if self.net is None:
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        return self._detect_dnn(frame) if self.net is not None else self._detect_haar(frame)

    def _detect_dnn(self, frame: np.ndarray) -> list[FaceBox]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces: list[FaceBox] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence <= self.config.face_detection_confidence:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            width = x2 - x1
            height = y2 - y1
            if width < 100 or height < 100:
                continue
            aspect_ratio = width / height if height > 0 else 0.0
            if not (0.5 < aspect_ratio < 2.0):
                continue
            faces.append(FaceBox(x1, y1, x2, y2, confidence))
        return faces

    def _detect_haar(self, frame: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(gray, 1.05, 3, minSize=(100, 100))
        return [FaceBox(int(x), int(y), int(x + w), int(y + h), 0.8) for (x, y, w, h) in boxes]

    @staticmethod
    def select_largest(faces: list[FaceBox]) -> FaceBox | None:
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)

    def align_face(self, frame: np.ndarray, box: FaceBox) -> np.ndarray | None:
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
        x2, y2 = min(w, int(box.x2)), min(h, int(box.y2))
        if x2 <= x1 or y2 <= y1:
            return None

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        face = cv2.resize(face, (self.config.face_size, self.config.face_size))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5
        return face
