"""
ONNX Runtime 推理封装。
加载 best.onnx，输入图片，输出 class_id + 置信度。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _imread_unicode(path: str) -> np.ndarray | None:
    import cv2

    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


class ONNXClassifier:
    """YOLO11n-cls ONNX 推理器。"""

    def __init__(self, model_path: str):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")

        self.session = ort.InferenceSession(
            str(model_file),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        _, _, self.input_h, self.input_w = self.session.get_inputs()[0].shape
        self.class_count = self.session.get_outputs()[0].shape[1]

    def predict(self, img: np.ndarray) -> tuple[int, float, list[tuple[int, float]]]:
        """
        输入 BGR 图片 (H,W,3)，输出 (最佳class_id, 置信度, Top-5列表)。
        """
        if img is None:
            raise ValueError("输入图片为空，无法推理")

        h, w = img.shape[:2]
        scale = max(self.input_h / h, self.input_w / w)
        new_h = max(self.input_h, math.ceil(h * scale))
        new_w = max(self.input_w, math.ceil(w * scale))
        resized = _resize_to_rgb(img, new_w, new_h).astype(np.float32)

        dy = (new_h - self.input_h) // 2
        dx = (new_w - self.input_w) // 2
        cropped = resized[dy:dy + self.input_h, dx:dx + self.input_w]
        if cropped.shape[:2] != (self.input_h, self.input_w):
            raise ValueError(
                f"预处理后尺寸异常: got {cropped.shape[:2]}, expected {(self.input_h, self.input_w)}"
            )

        blob = cropped / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        outputs = self.session.run(None, {self.input_name: blob})[0][0]
        probs = _softmax(outputs)
        best = int(np.argmax(probs))
        top5_idx = np.argsort(probs)[::-1][:5]
        top5 = [(int(i), float(probs[i])) for i in top5_idx]
        return best, float(probs[best]), top5


def _resize_to_rgb(img: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray(img[:, :, ::-1])
    pil = pil.resize((w, h), Image.BILINEAR)
    return np.array(pil)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "best.onnx"
    img_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        clf = ONNXClassifier(model_path)
        print(f"模型加载成功: {clf.input_w}x{clf.input_h}, {clf.class_count} 类")
        if img_path:
            img = _imread_unicode(img_path)
            if img is None:
                raise FileNotFoundError(f"图片读取失败: {img_path}")
            best, conf, top5 = clf.predict(img)
            print(f"  最佳类别 class {best}, 置信度 {conf:.4f}")
            for cid, c in top5:
                print(f"    class {cid}: {c:.4f}")
    except Exception as exc:
        print(f"错误: {exc}")
        raise SystemExit(1)
