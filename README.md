# SecureFace Face Authentication Engine

本仓库包含可复用的本地人脸认证引擎，核心实现参考 `face_recognition_enhanced.py`，并重构为模块化结构，便于后续接入 Linux PAM / polkit / daemon。

## 模块

- `faceauth_engine/config.py`
- `faceauth_engine/detector.py`
- `faceauth_engine/liveness.py`
- `faceauth_engine/embedder.py`
- `faceauth_engine/database.py`
- `faceauth_engine/engine.py`

## 引擎接口

```python
from faceauth_engine import FaceAuthEngine

engine = FaceAuthEngine()
ok = engine.enroll("alice")
result = engine.authenticate()  # PASS / UNKNOWN / NOT_LIVE / ERROR
```

## 认证流程（固定）

1. detect face
2. select largest face
3. align to 112x112
4. normalize [-1,1]
5. run embedding model
6. run liveness model
7. compare with database (dot product)
8. apply threshold logic

## 说明

- 认证过程为单次 API（非无限 UI 循环）。
- 引擎层不包含 GUI、键盘交互、`cv2.imshow`。
- 模型路径可配置，模型缺失时不会在导入阶段报错；但缺失 embedding 模型时会导致认证返回 `ERROR`。
