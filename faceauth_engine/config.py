from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    # 路径配置（与 face_recognition_enhanced.py 保持一致）
    face_db_path: str = "face_database.pkl"
    liveness_model: str = "models/liveness.onnx"
    mobilefacenet_model: str = "models/w600k_mbf.onnx"
    prototxt: str = "models/deploy.prototxt"
    caffemodel: str = "models/res10_300x300_ssd_iter_140000.caffemodel"

    # 输入尺寸
    face_size: int = 112
    liveness_size: int = 128

    # 原始阈值（严格沿用）
    verification_threshold: float = 0.40
    liveness_threshold: float = 0.60
    min_quality_threshold: float = 0.60
    outlier_threshold: float = 0.60
    face_detection_confidence: float = 0.70

    # 数据库版本信息
    db_version: str = "mobilefacenet_v3_512d_enhanced"
    expected_feature_dim: int = 512

    # 认证采样（替代原 verify 无限循环）
    auth_frames: int = 8
    camera_index: int = 0

    # 注册采样（替代原 capture UI 流程）
    frames_per_angle: int = 8
    enroll_states: tuple[str, ...] = (
        "FRONT_GLASSES",
        "LEFT_GLASSES",
        "RIGHT_GLASSES",
        "UP_GLASSES",
        "DOWN_GLASSES",
        "FRONT_NOGLASSES",
    )

    liveness_check: bool = True


DEFAULT_CONFIG = EngineConfig()
