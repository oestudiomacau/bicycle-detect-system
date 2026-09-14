from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path


MODEL_ID = "PekingU/rtdetr_r50vd_coco_o365"
TRANSFORMERS_REQUIREMENT = "transformers>=4.42,<5"
PYTORCH_CU126_INDEX = "https://download.pytorch.org/whl/cu126"
PYTORCH_VERSION = "2.14.0"
TORCHVISION_VERSION = "0.29.0"


def environment_python(environment: Path) -> Path:
    folder = "Scripts" if sys.platform == "win32" else "bin"
    executable = "python.exe" if sys.platform == "win32" else "python"
    return environment / folder / executable


def run(command: list[str]) -> None:
    print("> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Install the Hugging Face RT-DETR detector")
    parser.add_argument("--env", type=Path, default=root / ".venv-anomalib")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root / "models" / "rtdetr_r50vd_coco_o365",
    )
    parser.add_argument("--backend", choices=("cpu", "cu126"), default="cu126")
    args = parser.parse_args()

    environment = args.env.resolve()
    python = environment_python(environment)
    if not python.exists():
        print(f"创建 AI 推理环境：{environment}", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    if args.backend == "cu126":
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                f"torch=={PYTORCH_VERSION}+cu126",
                f"torchvision=={TORCHVISION_VERSION}+cu126",
                "--extra-index-url",
                PYTORCH_CU126_INDEX,
            ]
        )
    else:
        run([str(python), "-m", "pip", "install", "torch", "torchvision"])

    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            TRANSFORMERS_REQUIREMENT,
            "opencv-python-headless>=4.10,<6",
        ]
    )
    download_script = (
        "from huggingface_hub import snapshot_download; import sys; "
        "snapshot_download(sys.argv[1], local_dir=sys.argv[2], "
        "allow_patterns=['config.json','preprocessor_config.json','model.safetensors'])"
    )
    model_dir = args.model_dir.resolve()
    run([str(python), "-c", download_script, MODEL_ID, str(model_dir)])
    check = (
        "import torch, transformers; "
        "print('Transformers', transformers.__version__); "
        "print('CUDA available', torch.cuda.is_available()); "
        + (
            "assert torch.cuda.is_available(), 'CUDA backend is not available'"
            if args.backend == "cu126"
            else ""
        )
    )
    run([str(python), "-c", check])
    print(f"RTDETR_READY={model_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
