from __future__ import annotations

import argparse
import json
import struct
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, logging


TWO_WHEELER_LABELS = {
    1: "自行车",
    3: "电动车/摩托车",
}


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_exact(size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class RTDetrInference:
    def __init__(self, model_path: Path, device: str, confidence: float) -> None:
        self.device = torch.device(device)
        self.confidence = confidence
        self.processor = RTDetrImageProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = RTDetrForObjectDetection.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        self._infer(np.zeros((640, 640, 3), dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> list[dict[str, object]]:
        result = self._infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        height, width = frame.shape[:2]
        detections: list[dict[str, object]] = []
        for score, label, box in zip(
            result["scores"],
            result["labels"],
            result["boxes"],
            strict=True,
        ):
            class_id = int(label)
            if class_id not in TWO_WHEELER_LABELS:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            x1 = float(np.clip(x1, 0, width - 1))
            y1 = float(np.clip(y1, 0, height - 1))
            x2 = float(np.clip(x2, x1 + 1, width))
            y2 = float(np.clip(y2, y1 + 1, height))
            detections.append(
                {
                    "rect": [
                        x1 / width,
                        y1 / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ],
                    "confidence": float(score),
                    "label": TWO_WHEELER_LABELS[class_id],
                }
            )
        return detections

    def _infer(self, rgb_frame: np.ndarray) -> dict[str, torch.Tensor]:
        inputs = self.processor(images=rgb_frame, return_tensors="pt").to(self.device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            outputs = self.model(**inputs)
        target_sizes = torch.tensor(
            [rgb_frame.shape[:2]],
            device=self.device,
        )
        return self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence,
        )[0]


def run_worker(args: argparse.Namespace) -> int:
    logging.set_verbosity_error()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，继续使用 YOLOX 回退模型")
    detector = RTDetrInference(args.model, args.device, args.confidence)
    write_status(
        args.ready_file,
        {
            "status": "ready",
            "device": str(detector.device),
            "model": "PekingU/rtdetr_r50vd_coco_o365",
        },
    )

    while True:
        header = read_exact(4)
        if header is None:
            break
        request_size = struct.unpack("<I", header)[0]
        if request_size == 0:
            break
        if request_size > 50_000_000:
            raise ValueError("输入视频帧数据异常")
        encoded = read_exact(request_size)
        if encoded is None:
            break
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("无法解码输入视频帧")
        response = json.dumps(
            detector.detect(frame),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        sys.stdout.buffer.write(struct.pack("<I", len(response)))
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RT-DETR frame inference worker")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    try:
        return run_worker(args)
    except Exception as error:
        write_status(
            args.ready_file,
            {"status": "error", "message": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
