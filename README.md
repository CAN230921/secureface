# SecureFace

本仓库提供两层能力：

1. **人脸认证引擎（Python）**：`faceauth_engine/`
2. **Linux 集成组件（Rust/C）**：`faceauthd`、`pam_faceauth.so`、`faceauth-polkit-agent`、`faceauth-settings`

---

## 目录说明

- `faceauth_engine/`：模块化引擎
- `face_recognition_enhanced.py`：完整参考实现（含 GUI 演示流程）
- `faceauthd/`：D-Bus daemon（Rust）
- `pam/`：PAM 模块（C）
- `faceauth-polkit-agent/`：GTK polkit agent（Rust）
- `faceauth-settings/`：GTK 设置界面（Rust）
- `deploy/`：systemd / D-Bus / desktop 部署文件
- `models/`：**模型目录（已创建，可直接放模型）**
- `data/`：数据库/运行数据目录

---

## 模型放置（你现在可以直接复制模型）

由于 GitHub 不适合上传大模型压缩包，仓库内已预留 `models/` 目录。

请把模型放到以下固定路径（与默认配置一致）：

- `models/w600k_mbf.onnx`（MobileFaceNet embedding）
- `models/liveness.onnx`（活体模型）
- `models/deploy.prototxt`（DNN 人脸检测）
- `models/res10_300x300_ssd_iter_140000.caffemodel`（DNN 人脸检测）

默认配置见：`faceauth_engine/config.py`。

---

## Python 引擎使用

### 1) 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy onnxruntime
```

### 2) 直接调用引擎 API

```python
from faceauth_engine import FaceAuthEngine

engine = FaceAuthEngine()
ok = engine.enroll("alice")
result = engine.authenticate()  # PASS / UNKNOWN / NOT_LIVE / ERROR
print(ok, result)
```

### 3) CLI 使用

```bash
python -m faceauth_engine --enroll alice
python -m faceauth_engine
```

---

## 编译 Linux 集成组件

### 1) Rust 组件

```bash
cargo build --workspace
```

会构建：
- `faceauthd`
- `faceauth-polkit-agent`
- `faceauth-settings`

### 2) PAM 模块

```bash
make -C pam
```

生成：`pam/pam_faceauth.so`

> 若报缺少 `dbus-1` 或 `pam` 头文件，请先安装开发包（如 `libdbus-1-dev`、`libpam0g-dev`）。

---

## 运行示例

### 1) 启动 daemon（开发环境）

```bash
cargo run -p faceauthd
```

### 2) 调用引擎（开发验证）

```bash
python -m faceauth_engine
```

### 3) 部署到系统（生产）

将 `deploy/` 中的文件安装到系统路径：

- `deploy/faceauthd.service`
- `deploy/io.secureface.FaceAuth.service`
- `deploy/secureface-polkit-agent.desktop`

并按你的发行版规范安装二进制到对应目录（例如 `/usr/libexec/secureface/`）。

---

## 认证流程（引擎固定）

1. detect face
2. select largest face
3. align to 112x112
4. normalize [-1,1]
5. run embedding model
6. run liveness model
7. compare with database (dot product)
8. apply threshold logic

返回值严格为：
- `PASS`
- `UNKNOWN`
- `NOT_LIVE`
- `ERROR`
