from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def dataset_summary(normal_dir: Path, abnormal_dir: Path | None = None) -> dict[str, int]:
    return {
        "normal": len(image_files(normal_dir)),
        "abnormal": len(image_files(abnormal_dir)) if abnormal_dir else 0,
    }


def validate_dataset(normal_dir: Path, abnormal_dir: Path | None = None) -> dict[str, int]:
    summary = dataset_summary(normal_dir, abnormal_dir)
    if summary["normal"] < 5:
        raise ValueError("正常图片至少需要 5 张，建议准备 50 张以上")
    if abnormal_dir is not None and summary["abnormal"] == 0:
        raise ValueError("已选择异常图片目录，但目录中没有可识别图片")
    return summary


def _build_model(model_name: str, image_size: int):
    from anomalib.models import Padim, Patchcore

    image_shape = (image_size, image_size)
    if model_name == "patchcore":
        return Patchcore(
            backbone="resnet18",
            layers=("layer2", "layer3"),
            coreset_sampling_ratio=0.1,
            num_neighbors=5,
            pre_processor=Patchcore.configure_pre_processor(image_shape),
        )
    if model_name == "padim":
        return Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
            n_features=100,
            pre_processor=Padim.configure_pre_processor(image_shape),
        )
    raise ValueError(f"不支持的模型：{model_name}")


def _enable_unicode_paths() -> None:
    """Work around Anomalib 2.6.2 rejecting every non-ASCII path as non-printable."""
    from anomalib.data.utils import path as anomalib_path

    anomalib_path.contains_non_printable_characters = lambda value: any(
        not character.isprintable() for character in str(value)
    )


def _json_default(value: Any) -> str | float | int | bool | None:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def train(args: argparse.Namespace) -> Path:
    normal_dir = args.normal_dir.resolve()
    abnormal_dir = args.abnormal_dir.resolve() if args.abnormal_dir else None
    summary = validate_dataset(normal_dir, abnormal_dir)
    run_dir = args.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据集检查完成：正常 {summary['normal']} 张，异常 {summary['abnormal']} 张", flush=True)
    print("正在加载 Anomalib 与 PyTorch...", flush=True)

    import torch
    from anomalib.data import Folder
    from anomalib.data.utils import TestSplitMode, ValSplitMode
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine

    _enable_unicode_paths()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("已选择 CUDA，但当前训练环境未检测到可用显卡")
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    accelerator = "gpu" if use_cuda else "cpu"
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"计算设备：{device_name}", flush=True)

    test_mode = TestSplitMode.FROM_DIR if abnormal_dir else TestSplitMode.SYNTHETIC
    datamodule = Folder(
        name="parking_anomaly",
        normal_dir=normal_dir,
        abnormal_dir=abnormal_dir,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=0,
        test_split_mode=test_mode,
        test_split_ratio=0.2,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        seed=42,
    )
    model = _build_model(args.model, args.image_size)
    engine = Engine(
        default_root_dir=run_dir,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    print(f"开始训练 {args.model.upper()}...", flush=True)
    engine.fit(model=model, datamodule=datamodule)
    print("训练完成，正在评估...", flush=True)
    metrics = engine.test(model=model, datamodule=datamodule)

    exported: dict[str, str] = {}
    if args.export in {"torch", "both"}:
        torch_path = engine.export(
            model=model,
            export_type=ExportType.TORCH,
            export_root=run_dir,
            model_file_name="parking_anomaly",
        )
        if torch_path:
            exported["torch"] = str(torch_path)
            print(f"Torch 模型：{torch_path}", flush=True)

    if args.export in {"openvino", "both"}:
        try:
            openvino_path = engine.export(
                model=model,
                export_type=ExportType.OPENVINO,
                export_root=run_dir,
                model_file_name="parking_anomaly",
                input_size=(args.image_size, args.image_size),
            )
            if openvino_path:
                exported["openvino"] = str(openvino_path)
                print(f"OpenVINO 模型：{openvino_path}", flush=True)
        except Exception as error:  # OpenVINO remains optional after a successful training run.
            if args.export == "openvino":
                raise RuntimeError(f"OpenVINO 导出失败：{error}") from error
            print(f"OpenVINO 导出未完成，Torch 模型仍可使用：{error}", flush=True)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "device": device_name,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "dataset": {**summary, "normal_dir": str(normal_dir), "abnormal_dir": str(abnormal_dir or "")},
        "metrics": metrics,
        "checkpoint": engine.best_model_path,
        "exported": exported,
    }
    metadata_path = run_dir / "training_summary.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"训练结果目录：{run_dir}", flush=True)
    print("TRAINING_COMPLETE", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an Anomalib parking anomaly model")
    parser.add_argument("--normal-dir", type=Path, required=True)
    parser.add_argument("--abnormal-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("patchcore", "padim"), default="patchcore")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--export", choices=("torch", "openvino", "both"), default="both")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        train(args)
    except Exception as error:
        print(f"TRAINING_FAILED: {type(error).__name__}: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
