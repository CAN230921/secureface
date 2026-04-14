from __future__ import annotations

import importlib
import importlib.util
import os

import numpy as np

from .config import EngineConfig


class MobileFaceNet:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.config.mobilefacenet_model):
            return
        if not importlib.util.find_spec("onnxruntime"):
            return

        ort = importlib.import_module("onnxruntime")
        self.session = ort.InferenceSession(self.config.mobilefacenet_model, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def is_loaded(self) -> bool:
        return self.session is not None

    def extract(self, face_img: np.ndarray) -> np.ndarray | None:
        if self.session is None:
            return None

        if face_img is None or face_img.shape != (112, 112, 3):
            return None

        if self.input_shape and len(self.input_shape) == 4 and self.input_shape[1] == 3:
            input_tensor = np.transpose(face_img, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        elif self.input_shape and len(self.input_shape) == 4:
            input_tensor = np.expand_dims(face_img, axis=0).astype(np.float32)
        else:
            input_tensor = np.transpose(face_img, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        embedding = outputs[0][0]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
