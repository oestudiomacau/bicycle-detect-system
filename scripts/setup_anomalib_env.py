from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path


ANOMALIB_VERSION = "2.6.2"
PYTORCH_CU126_INDEX = "https://download.pytorch.org/whl/cu126"
PYTORCH_VERSION = "2.14.0"
TORCHVISION_VERSION = "0.29.0"
TORCHCODEC_VERSION = "0.16.0"


def environment_python(environment: Path) -> Path:
    folder = "Scripts" if sys.platform == "win32" else "bin"
    executable = "python.exe" if sys.platform == "win32" else "python"
    return environment / folder / executable


def run(command: list[str]) -> None:
    print("> " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the isolated Anomalib environment")
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cu126"), default="cu126")
    args = parser.parse_args()

    environment = args.env.resolve()
    python = environment_python(environment)
    if not python.exists():
        print(f"创建训练环境：{environment}", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)

    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    if args.backend == "cu126":
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                f"torch=={PYTORCH_VERSION}+cu126",
                f"torchvision=={TORCHVISION_VERSION}+cu126",
                f"torchcodec=={TORCHCODEC_VERSION}+cu126",
                "--extra-index-url",
                PYTORCH_CU126_INDEX,
            ]
        )
        package = f"anomalib[openvino]=={ANOMALIB_VERSION}"
    else:
        package = f"anomalib[cpu,openvino]=={ANOMALIB_VERSION}"
    run([str(python), "-m", "pip", "install", package])
    run(
        [
            str(python),
            "-c",
            "import anomalib, torch; "
            "print('Anomalib', anomalib.__version__); "
            "print('PyTorch', torch.__version__); "
            "print('CUDA available', torch.cuda.is_available()); "
            + ("assert torch.cuda.is_available(), 'CUDA backend is not available'" if args.backend == "cu126" else ""),
        ]
    )
    print("ENVIRONMENT_READY", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
