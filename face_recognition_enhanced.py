import cv2
import numpy as np
import onnxruntime as ort
import os
import pickle
import urllib.request
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
from collections import deque

@dataclass
class Config:
    """
    增强版配置 (更多样本 + 智能特征分析):
    - 每角度采集: 8帧 (原4帧)
    - 总样本: 48帧 (原20帧) 
    - 特征融合: 加权平均 + 异常值剔除
    - 支持戴眼镜/不戴眼镜混合采集
    """
    FACE_DB_PATH = "face_database.pkl"
    FACE_SIZE = 112
    LIVENESS_SIZE = 128
    VERIFICATION_THRESHOLD = 0.40  # 降低阈值，提高跨状态识别率
    LIVENESS_CHECK = True
    LIVENESS_THRESHOLD = 0.60
    LIVENESS_MODEL = "models/liveness.onnx"
    CAPTURE_TIMEOUT = 120  # 增加到120秒，采集更多样本
    MOBILEFACENET_MODEL = "models/w600k_mbf.onnx"
    PROTOTXT = "models/deploy.prototxt"
    CAFFEMODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
    DB_VERSION = "mobilefacenet_v3_512d_enhanced"
    EXPECTED_FEATURE_DIM = 512
    MAX_FACES_ALLOWED = 1
    FACE_DETECTION_CONFIDENCE = 0.7  # 降低到0.7，支持戴眼镜检测
    
    # 增强采集参数
    FRAMES_PER_ANGLE = 8  # 每角度8帧 (原4帧)
    MIN_QUALITY_THRESHOLD = 0.6  # 质量分数阈值
    OUTLIER_THRESHOLD = 0.6  # 异常样本相似度阈值（低于此值剔除）

class FaceDetector:
    def __init__(self):
        self.net = None
        self.cascade = None
        self.load_model()
    
    def load_model(self):
        if os.path.exists(Config.PROTOTXT) and os.path.exists(Config.CAFFEMODEL):
            try:
                self.net = cv2.dnn.readNetFromCaffe(Config.PROTOTXT, Config.CAFFEMODEL)
                print(f"Loaded DNN Face Detector (conf threshold: {Config.FACE_DETECTION_CONFIDENCE})")
            except Exception as e:
                print(f"DNN failed: {e}, using Haar")
                self.net = None
        else:
            print("DNN model not found, using Haar")
            self.net = None
        
        if self.net is None:
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        if self.net is not None:
            return self._detect_dnn(frame)
        else:
            return self._detect_haar(frame)
    
    def _detect_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > Config.FACE_DETECTION_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                width = x2 - x1
                height = y2 - y1
                
                # 尺寸过滤（防眼镜小框）
                if width < 100 or height < 100:
                    continue
                
                # 宽高比检查（防眼镜/嘴巴误检）
                aspect_ratio = width / height if height > 0 else 0
                if not (0.5 < aspect_ratio < 2.0):
                    continue
                
                faces.append((x1, y1, x2, y2, confidence))
        return faces
    
    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.05, 3, minSize=(100, 100))
        return [(x, y, x+w, y+h, 0.8) for (x, y, w, h) in faces]
    
    def align_face(self, frame: np.ndarray, box: Tuple, target_size: int = 112) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, box[:4])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face = cv2.resize(face, (target_size, target_size))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5
        return face

class LivenessDetector:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.load_model()
    
    def load_model(self):
        if not os.path.exists(Config.LIVENESS_MODEL):
            print(f"WARNING: Liveness model not found: {Config.LIVENESS_MODEL}")
            print("Live detection disabled (fallback to quality check)")
            return
        
        try:
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(Config.LIVENESS_MODEL, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            print(f"Loaded Liveness Detector: {Config.LIVENESS_MODEL}")
        except Exception as e:
            print(f"Failed to load liveness model: {e}")
            self.session = None
    
    def is_loaded(self) -> bool:
        return self.session is not None
    
    def check(self, face_img_112: np.ndarray) -> Tuple[bool, float]:
        if self.session is None:
            quality_score = self._quality_check(face_img_112)
            return quality_score > 0.7, quality_score
        
        try:
            input_img = cv2.resize(face_img_112, (Config.LIVENESS_SIZE, Config.LIVENESS_SIZE))
            input_img = ((input_img + 1) * 127.5).astype(np.float32)
            input_tensor = np.transpose(input_img, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            output = outputs[0][0]
            
            if len(output) == 2:
                live_score = float(output[1])
            else:
                live_score = float(output[0])
            
            live_score = np.clip(live_score, 0.0, 1.0)
            is_live = live_score > Config.LIVENESS_THRESHOLD
            return is_live, live_score
            
        except Exception as e:
            print(f"Liveness check error: {e}")
            return False, 0.0
    
    def _quality_check(self, face_img: np.ndarray) -> float:
        gray = cv2.cvtColor(((face_img + 1) * 127.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        clarity_score = min(laplacian_var / 500, 1.0)
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 127) / 127
        std_score = min(np.std(gray) / 50, 1.0)
        return (clarity_score * 0.5 + brightness_score * 0.3 + std_score * 0.2)

class MobileFaceNet:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.load_model()
    
    def load_model(self):
        if not os.path.exists(Config.MOBILEFACENET_MODEL):
            print(f"ERROR: Model not found: {Config.MOBILEFACENET_MODEL}")
            return
        
        try:
            providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(Config.MOBILEFACENET_MODEL, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            print(f"Loaded MobileFaceNet: Input {self.input_shape}, Output {self.session.get_outputs()[0].shape}")
            
            if len(self.input_shape) == 4:
                if self.input_shape[1] == 3:
                    print("  Format: NCHW")
                elif self.input_shape[3] == 3:
                    print("  Format: NHWC")
                    
        except Exception as e:
            print(f"Failed to load MobileFaceNet: {e}")
            self.session = None
    
    def is_loaded(self) -> bool:
        return self.session is not None
    
    def extract(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        if self.session is None:
            return None
        
        if face_img is None or face_img.shape != (112, 112, 3):
            return None
        
        try:
            if self.input_shape and len(self.input_shape) == 4:
                if self.input_shape[1] == 3:  # NCHW
                    input_tensor = np.transpose(face_img, (2, 0, 1))
                    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
                else:  # NHWC
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
            
        except Exception as e:
            print(f"Extract error: {e}")
            return None

class FaceDatabase:
    def __init__(self):
        self.path = Config.FACE_DB_PATH
        self.faces = {}
        self.load()
    
    def load(self):
        if not os.path.exists(self.path):
            self.faces = {}
            return
        
        try:
            with open(self.path, 'rb') as f:
                data = pickle.load(f)
            
            db_version = data.get('version', 'legacy_v0')
            
            if db_version != Config.DB_VERSION:
                print("DATABASE VERSION MISMATCH! Please re-register.")
                self.faces = {}
                return
            
            self.faces = data.get('faces', {})
            print(f"Loaded {len(self.faces)} users")
            
        except Exception as e:
            print(f"Error loading DB: {e}")
            self.faces = {}
    
    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump({
                'version': Config.DB_VERSION,
                'feature_dim': Config.EXPECTED_FEATURE_DIM,
                'faces': self.faces
            }, f)
    
    def add(self, name, emb):
        if emb.shape[0] != Config.EXPECTED_FEATURE_DIM:
            print(f"ERROR: Wrong feature dim {emb.shape[0]}, expected {Config.EXPECTED_FEATURE_DIM}")
            return False
        
        self.faces[name] = emb
        self.save()
        print(f"Added: {name}")
        return True
    
    def remove(self, name):
        if name in self.faces:
            del self.faces[name]
            self.save()
            return True
        return False
    
    def verify(self, emb, threshold=None):
        if not self.faces:
            return None, 0.0
        
        threshold = threshold or Config.VERIFICATION_THRESHOLD
        best, score = None, -1
        
        for name, stored in self.faces.items():
            sim = np.dot(emb, stored)
            if sim > score and sim > threshold:
                score, best = sim, name
        
        return best, score
    
    def list(self):
        return list(self.faces.keys())
    
    def clear(self):
        self.faces = {}
        self.save()
        print("Database cleared")

class FaceRecognitionSystem:
    def __init__(self):
        print("=" * 60)
        print("Enhanced Face Recognition (Smart Analysis System)")
        print(f"Detection confidence: {Config.FACE_DETECTION_CONFIDENCE}")
        print(f"Frames per angle: {Config.FRAMES_PER_ANGLE}")
        print(f"Verification threshold: {Config.VERIFICATION_THRESHOLD}")
        print("=" * 60)
        
        self.detector = FaceDetector()
        self.liveness_detector = LivenessDetector() if Config.LIVENESS_CHECK else None
        self.extractor = MobileFaceNet()
        self.db = FaceDatabase()
        
        if not self.extractor.is_loaded():
            print("\nWARNING: Face recognition model not loaded!")
    
    def _check_face_quality(self, face_img: np.ndarray) -> float:
        """返回0-1的质量分数（清晰度/亮度/对比度）"""
        gray = cv2.cvtColor(((face_img + 1) * 127.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # 清晰度（拉普拉斯方差）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        clarity = min(laplacian_var / 300, 1.0)
        
        # 亮度
        mean_brightness = np.mean(gray)
        brightness = 1.0 - abs(mean_brightness - 127) / 127
        
        # 对比度
        contrast = min(np.std(gray) / 40, 1.0)
        
        quality_score = clarity * 0.5 + brightness * 0.3 + contrast * 0.2
        return quality_score
    
    def _get_main_face(self, faces: List[Tuple]) -> List[Tuple]:
        """只取面积最大的人脸"""
        if len(faces) <= 1:
            return faces
        main_face = max(faces, key=lambda f: (f[2]-f[0]) * (f[3]-f[1]))
        return [main_face]
    
    def _analyze_features(self, embeddings: List[np.ndarray]) -> Tuple[np.ndarray, dict]:
        """
        智能特征分析算法:
        1. 找到中心样本（与其他样本平均相似度最高）
        2. 剔除异常值（与中心样本相似度<0.6的）
        3. 加权平均（与中心样本相似度作为权重）
        """
        if len(embeddings) == 0:
            return None, {"error": "No embeddings"}
        
        if len(embeddings) == 1:
            return embeddings[0], {"method": "single", "samples": 1}
        
        embeddings_array = np.array(embeddings)
        n_samples = len(embeddings)
        
        # 计算相似度矩阵
        sim_matrix = np.dot(embeddings_array, embeddings_array.T)
        
        # 找到中心样本
        avg_sims = np.mean(sim_matrix, axis=1)
        center_idx = np.argmax(avg_sims)
        
        # 剔除异常值
        center_sim = sim_matrix[center_idx]
        valid_mask = center_sim > Config.OUTLIER_THRESHOLD
        
        valid_embeddings = embeddings_array[valid_mask]
        valid_count = len(valid_embeddings)
        
        analysis_info = {
            "total_samples": n_samples,
            "valid_samples": valid_count,
            "outliers_removed": n_samples - valid_count,
            "center_sim_max": float(np.max(center_sim)),
            "center_sim_min": float(np.min(center_sim)),
            "center_sim_mean": float(np.mean(center_sim))
        }
        
        if valid_count == 0:
            print("  WARNING: All samples inconsistent, using median fallback...")
            median_emb = np.median(embeddings_array, axis=0)
            norm = np.linalg.norm(median_emb)
            median_emb = median_emb / norm if norm > 0 else median_emb
            analysis_info["method"] = "median_fallback"
            return median_emb, analysis_info
        
        # 加权平均
        weights = center_sim[valid_mask]
        weights = weights / np.sum(weights)
        
        weighted_emb = np.average(valid_embeddings, axis=0, weights=weights)
        
        # L2归一化
        norm = np.linalg.norm(weighted_emb)
        if norm > 0:
            weighted_emb = weighted_emb / norm
        
        analysis_info["method"] = "weighted_average" if valid_count > 1 else "single_valid"
        return weighted_emb, analysis_info
    
    def capture(self, name, cam=0):
        if not self.extractor.is_loaded():
            print("ERROR: Model not loaded")
            return False
        
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return False
        
        # 增强采集阶段：更多角度，更多帧，支持戴眼镜/不戴眼镜
        capture_stages = [
            ("FRONT_GLASSES", "正面 (戴眼镜)", Config.FRAMES_PER_ANGLE, (0, 255, 0)),
            ("LEFT_GLASSES", "左转 (戴眼镜)", Config.FRAMES_PER_ANGLE, (255, 255, 0)),
            ("RIGHT_GLASSES", "右转 (戴眼镜)", Config.FRAMES_PER_ANGLE, (0, 255, 255)),
            ("UP_GLASSES", "抬头 (戴眼镜)", Config.FRAMES_PER_ANGLE, (255, 0, 255)),
            ("DOWN_GLASSES", "低头 (戴眼镜)", Config.FRAMES_PER_ANGLE, (255, 165, 0)),
            ("FRONT_NOGLASSES", "正面 (摘掉眼镜)", Config.FRAMES_PER_ANGLE, (0, 128, 255)),
        ]
        
        total_target = sum(s[2] for s in capture_stages)
        
        print(f"\n{'='*60}")
        print(f"Enhanced Capture: {name}")
        print(f"Algorithm: Outlier removal + Weighted averaging")
        print(f"Total: {total_target} high-quality samples across {len(capture_stages)} states")
        print(f"Supports: Glasses ON/OFF mixed enrollment")
        print(f"{'='*60}")
        
        all_embeddings = []
        all_quality_scores = []
        current_stage_idx = 0
        stage_frame_count = 0
        
        start_time = time.time()
        last_capture = 0
        
        while current_stage_idx < len(capture_stages):
            ret, frame = cap.read()
            if not ret:
                continue
            
            display = cv2.flip(frame, 1)
            faces = self.detector.detect(display)
            faces = self._get_main_face(faces)
            
            stage_name, stage_hint, stage_target, stage_color = capture_stages[current_stage_idx]
            
            status = "No face"
            box_color = (128, 128, 128)
            
            if len(faces) == 0:
                status = "No face detected"
            else:
                main_face = faces[0]
                x1, y1, x2, y2, conf = main_face
                
                if conf < Config.FACE_DETECTION_CONFIDENCE:
                    status = f"Low conf: {conf:.2f}"
                    box_color = (0, 165, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
                else:
                    # 对齐并评估质量
                    aligned = self.detector.align_face(display, main_face, target_size=112)
                    
                    if aligned is not None:
                        quality_score = self._check_face_quality(aligned)
                        
                        # 活体检测
                        is_live = True
                        live_score = 1.0
                        if Config.LIVENESS_CHECK and self.liveness_detector:
                            is_live, live_score = self.liveness_detector.check(aligned)
                        
                        quality_str = f"Q:{quality_score:.2f}"
                        
                        if not is_live:
                            status = f"FAKE! L:{live_score:.2f}"
                            box_color = (0, 0, 255)
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(display, "SPOOF", (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        elif quality_score < Config.MIN_QUALITY_THRESHOLD:
                            status = f"Poor {quality_str}"
                            box_color = (0, 140, 255)
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 140, 255), 2)
                        else:
                            box_color = stage_color
                            cv2.rectangle(display, (x1, y1), (x2, y2), stage_color, 3)
                            
                            current_time = time.time()
                            if current_time - last_capture > 0.5:  # 更快采集
                                emb = self.extractor.extract(aligned)
                                if emb is not None:
                                    all_embeddings.append(emb)
                                    all_quality_scores.append(quality_score)
                                    stage_frame_count += 1
                                    last_capture = current_time
                                    
                                    print(f"  [{stage_name}] {stage_frame_count}/{stage_target} | "
                                          f"Q:{quality_score:.2f} L:{live_score:.2f} | "
                                          f"Total:{len(all_embeddings)}/{total_target}")
                                    
                                    if stage_frame_count >= stage_target:
                                        print(f"  ✓ {stage_name} complete")
                                        current_stage_idx += 1
                                        stage_frame_count = 0
                                        if current_stage_idx < len(capture_stages):
                                            print(f"\n>>> Next: {capture_stages[current_stage_idx][1]}")
                                            if "NOGLASSES" in capture_stages[current_stage_idx][0]:
                                                print("    [提示：请摘掉眼镜]")
                                    
                                    status = f"Cap {stage_frame_count}/{stage_target} {quality_str}"
                            else:
                                status = f"Wait {quality_str}"
                    else:
                        status = "Align fail"
            
            # 显示信息
            cv2.putText(display, f"Stage: {stage_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, stage_color, 2)
            cv2.putText(display, f"Hint: {stage_hint}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Samples: {len(all_embeddings)}/{total_target} | {status}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # 进度条
            bar_width = 400
            filled = int(bar_width * len(all_embeddings) / total_target)
            cv2.rectangle(display, (10, 120), (10 + bar_width, 140), (50, 50, 50), -1)
            cv2.rectangle(display, (10, 120), (10 + filled, 140), stage_color, -1)
            
            # 质量统计
            if len(all_quality_scores) > 0:
                avg_q = np.mean(all_quality_scores)
                cv2.putText(display, f"Avg Quality: {avg_q:.2f}", (10, 165), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Enhanced Capture", display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Cancelled")
                break
            
            if time.time() - start_time > Config.CAPTURE_TIMEOUT:
                print("Timeout")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 智能特征分析
        print(f"\n{'='*60}")
        print("Smart Feature Analysis")
        print(f"{'='*60}")
        print(f"  Raw samples collected: {len(all_embeddings)}")
        
        if len(all_embeddings) < total_target * 0.4:
            print(f"\n✗ Failed: Only {len(all_embeddings)} samples (need {total_target//2}+)")
            return False
        
        # 调用分析算法
        final_emb, analysis_info = self._analyze_features(all_embeddings)
        
        print(f"  Valid samples used: {analysis_info['valid_samples']}/{analysis_info['total_samples']}")
        print(f"  Outliers removed: {analysis_info.get('outliers_removed', 0)}")
        print(f"  Similarity range: {analysis_info['center_sim_min']:.3f} ~ {analysis_info['center_sim_max']:.3f}")
        print(f"  Mean similarity: {analysis_info['center_sim_mean']:.3f}")
        print(f"  Fusion method: {analysis_info['method']}")
        
        if final_emb is None:
            print("✗ Error: Feature analysis failed")
            return False
        
        # 保存
        success = self.db.add(name, final_emb)
        if success:
            print(f"\n✓ Success: {name}")
            print(f"  Feature vector: 512D (enhanced)")
            print(f"  Enrollment states: Glasses ON/OFF mixed")
            print(f"  Cross-state matching: Enabled")
            print(f"{'='*60}")
        return success
    
    def verify(self, cam=0):
        if not self.extractor.is_loaded():
            print("ERROR: Model not loaded")
            return
        
        if not self.db.list():
            print("Database empty")
            return
        
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("\nEnhanced Verification Mode")
        print(f"Users: {self.db.list()}")
        print(f"Threshold: {Config.VERIFICATION_THRESHOLD} (supports glasses on/off)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            display = cv2.flip(frame, 1)
            faces = self.detector.detect(display)
            faces = self._get_main_face(faces)
            
            status = "No face"
            color = (128, 128, 128)
            
            if len(faces) == 0:
                status = "No face"
                color = (128, 128, 128)
            else:
                main_face = faces[0]
                x1, y1, x2, y2, conf = main_face
                
                # 活体检测
                is_live = True
                live_score = 1.0
                if Config.LIVENESS_CHECK and self.liveness_detector:
                    face_112 = self.detector.align_face(display, main_face, target_size=112)
                    if face_112 is not None:
                        is_live, live_score = self.liveness_detector.check(face_112)
                
                if not is_live:
                    status = f"FAKE! {live_score:.2f}"
                    color = (0, 0, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(display, "!!! FAKE FACE !!!", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif conf < Config.FACE_DETECTION_CONFIDENCE:
                    status = f"Low conf: {conf:.2f}"
                    color = (0, 165, 255)
                else:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    aligned = self.detector.align_face(display, main_face, target_size=112)
                    if aligned is not None:
                        emb = self.extractor.extract(aligned)
                        if emb is not None:
                            user, sim = self.db.verify(emb)
                            if user:
                                status = f"{user} ({sim:.2f}) L:{live_score:.2f}"
                                color = (0, 255, 0)
                            else:
                                status = f"Unknown L:{live_score:.2f}"
                                color = (0, 165, 255)
            
            # 显示被过滤的小脸
            all_faces = self.detector.detect(display)
            if len(all_faces) > 1:
                for face in all_faces[1:]:
                    x1, y1, x2, y2 = face[:4]
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(display, f"DB:{len(self.db.list())} | Threshold:{Config.VERIFICATION_THRESHOLD}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Enhanced Verify", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def download_model():
    print("Models download links:")
    print("\n1. Face Recognition (512D):")
    print("   https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx")
    print("   -> models/w600k_mbf.onnx")
    print("\n2. Liveness Detection (128x128):")
    print("   https://github.com/SuriAI/face-antispoof-onnx/releases/download/v1.0.0/best_model.onnx")
    print("   -> models/liveness.onnx")

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['capture', 'verify', 'list', 'delete', 'clear-db', 'download-model'])
    p.add_argument('-n', '--name')
    p.add_argument('-c', '--camera', type=int, default=0)
    p.add_argument('--no-liveness', action='store_true', help="Disable liveness detection")
    args = p.parse_args()
    
    if args.no_liveness:
        Config.LIVENESS_CHECK = False
    
    if args.cmd == 'download-model':
        download_model()
        return
    
    sys = FaceRecognitionSystem()
    
    if args.cmd == 'capture':
        if not args.name:
            p.error("Need --name")
        sys.capture(args.name, args.camera)
    elif args.cmd == 'verify':
        sys.verify(args.camera)
    elif args.cmd == 'list':
        print(f"Users: {sys.db.list()}")
    elif args.cmd == 'delete':
        if not args.name:
            p.error("Need --name")
        print(f"Deleted: {args.name}" if sys.db.remove(args.name) else f"Not found: {args.name}")
    elif args.cmd == 'clear-db':
        if input("Confirm clear? (yes/no): ").lower() == 'yes':
            sys.db.clear()

if __name__ == "__main__":
    main()