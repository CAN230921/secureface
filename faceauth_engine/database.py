from __future__ import annotations

import os
import pickle

import numpy as np

from .config import EngineConfig


class FaceDatabase:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.path = config.face_db_path
        self.faces: dict[str, np.ndarray] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            self.faces = {}
            return

        with open(self.path, "rb") as f:
            data = pickle.load(f)

        db_version = data.get("version", "legacy_v0")
        if db_version != self.config.db_version:
            self.faces = {}
            return

        faces = data.get("faces", {})
        self.faces = {k: np.asarray(v, dtype=np.float32) for k, v in faces.items()}

    def save(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(
                {
                    "version": self.config.db_version,
                    "feature_dim": self.config.expected_feature_dim,
                    "faces": self.faces,
                },
                f,
            )

    def add(self, name: str, emb: np.ndarray) -> bool:
        if emb.shape[0] != self.config.expected_feature_dim:
            return False
        self.faces[name] = emb
        self.save()
        return True

    def verify(self, emb: np.ndarray, threshold: float | None = None) -> tuple[str | None, float]:
        if not self.faces:
            return None, 0.0

        threshold = threshold or self.config.verification_threshold
        best, score = None, -1.0

        for name, stored in self.faces.items():
            sim = float(np.dot(emb, stored))
            if sim > score and sim > threshold:
                score, best = sim, name

        return best, score
