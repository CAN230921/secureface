from __future__ import annotations

import importlib
import importlib.util
import os

import cv2
import numpy as np

from .config import EngineConfig


class LivenessDetector:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.config.liveness_model):
            return
        if not importlib.util.find_spec("onnxruntime"):
            return

        ort = importlib.import_module("onnxruntime")
        self.session = ort.InferenceSession(self.config.liveness_model, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def is_loaded(self) -> bool:
        return self.session is not None

    def check(self, face_img_112: np.ndarray) -> tuple[bool, float]:
        if self.session is None:
            quality_score = self._quality_check(face_img_112)
            return quality_score > 0.7, quality_score

        input_img = cv2.resize(face_img_112, (self.config.liveness_size, self.config.liveness_size))
        input_img = ((input_img + 1) * 127.5).astype(np.float32)
        input_tensor = np.transpose(input_img, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        output = outputs[0][0]

        live_score = float(output[1]) if len(output) == 2 else float(output[0])
        live_score = float(np.clip(live_score, 0.0, 1.0))
        is_live = live_score > self.config.liveness_threshold
        return is_live, live_score

    @staticmethod
    def _quality_check(face_img: np.ndarray) -> float:
        gray = cv2.cvtColor(((face_img + 1) * 127.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        clarity_score = min(laplacian_var / 500, 1.0)
        mean_brightness = float(np.mean(gray))
        brightness_score = 1.0 - abs(mean_brightness - 127) / 127
        std_score = min(float(np.std(gray)) / 50, 1.0)
        return clarity_score * 0.5 + brightness_score * 0.3 + std_score * 0.2
