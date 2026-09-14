from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SourceStatus:
    name: str
    connected: bool
    detail: str


class VideoSourceAdapter(ABC):
    """Minimal contract shared by the mock source and future HCNetSDK adapter."""

    @abstractmethod
    def open(self) -> SourceStatus:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Any:
        raise NotImplementedError

    def move_to_preset(self, preset: int) -> None:
        raise NotImplementedError("当前视频源不支持云台预置点切换")


class HikvisionSdkAdapter(VideoSourceAdapter):
    """Placeholder for HCNetSDK login, preview callback and PTZ preset control."""

    def __init__(self, sdk_library: Path | None = None) -> None:
        self.sdk_library = sdk_library

    def open(self) -> SourceStatus:
        return SourceStatus("海康监控相机", False, "HCNetSDK 尚未接入")

    def close(self) -> None:
        return None

    def read(self) -> Any:
        return None

    def move_to_preset(self, preset: int) -> None:
        raise RuntimeError(f"预置点 {preset} 已保留，需接入 HCNetSDK 后启用")


class ParkingAnomalyAdapter:
    """Stable boundary for connecting an exported Anomalib model later."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    @property
    def ready(self) -> bool:
        return bool(self.model_path and self.model_path.exists())

    def predict(self, frame: Any) -> dict[str, Any]:
        if not self.ready:
            return {"score": 0.0, "label": "model_not_loaded", "mask": None}
        raise NotImplementedError("在这里接入 Anomalib/OpenVINO 推理结果")

