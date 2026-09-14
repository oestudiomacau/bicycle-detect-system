from __future__ import annotations

import json
from pathlib import Path

from .domain import (
    DEFAULT_PARKING_POLYGON,
    DEFAULT_PARKING_ZONE,
    DEFAULT_SPEED_LINES,
    LineSegment,
    Point,
    validate_parking_polygon,
    validate_parking_zone,
    validate_speed_lines,
)


class SettingsStore:
    """Persist the small set of user-adjustable monitoring settings."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load_payload(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def load_parking_zone(self) -> tuple[float, float, float, float]:
        try:
            payload = self._load_payload()
            return validate_parking_zone(tuple(payload["parking_zone"]))
        except (KeyError, TypeError, ValueError):
            return DEFAULT_PARKING_ZONE

    def load_parking_polygon(self) -> tuple[Point, ...]:
        payload = self._load_payload()
        try:
            return validate_parking_polygon(payload["parking_polygon"])
        except (KeyError, TypeError, ValueError):
            zone = self.load_parking_zone()
            if zone == DEFAULT_PARKING_ZONE:
                return DEFAULT_PARKING_POLYGON
            left, top, right, bottom = zone
            return ((left, top), (right, top), (right, bottom), (left, bottom))

    def load_speed_lines(self) -> tuple[LineSegment, LineSegment]:
        try:
            payload = self._load_payload()
            raw_lines = tuple(tuple(line) for line in payload["speed_lines"])
            return validate_speed_lines(raw_lines)
        except (KeyError, TypeError, ValueError):
            return DEFAULT_SPEED_LINES

    def save_parking_zone(self, zone: tuple[float, float, float, float]) -> None:
        validated = validate_parking_zone(zone)
        left, top, right, bottom = validated
        polygon = ((left, top), (right, top), (right, bottom), (left, bottom))
        self.save_parking_polygon(polygon)

    def save_parking_polygon(self, points: tuple[Point, ...]) -> None:
        polygon = validate_parking_polygon(points)
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        zone = (min(xs), min(ys), max(xs), max(ys))
        self._save_values(
            {
                "parking_polygon": [list(point) for point in polygon],
                "parking_zone": list(zone),
            }
        )

    def save_speed_lines(self, lines: tuple[LineSegment, LineSegment]) -> None:
        validated = validate_speed_lines(lines)
        self._save_values({"speed_lines": [list(line) for line in validated]})

    def load_startup_video(self) -> tuple[Path, str] | None:
        payload = self._load_payload()
        try:
            startup_video = payload["startup_video"]
            path = Path(startup_video["path"]).expanduser()
            mode = startup_video["mode"]
        except (KeyError, TypeError, ValueError):
            return None
        if mode not in {"road", "parking"} or not path.is_absolute() or not path.is_file():
            return None
        return path.resolve(), mode

    def save_startup_video(
        self,
        path: Path,
        mode: str,
        *,
        require_exists: bool = True,
    ) -> None:
        resolved = path.expanduser().resolve()
        if mode not in {"road", "parking"}:
            raise ValueError("启动示范视频模式必须是 road 或 parking")
        if require_exists and not resolved.is_file():
            raise FileNotFoundError(f"启动示范视频不存在：{resolved}")
        self._save_values(
            {
                "startup_video": {"path": str(resolved), "mode": mode},
                "startup_video_disabled": False,
            }
        )

    def clear_startup_video(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._load_payload()
        payload.pop("startup_video", None)
        payload["startup_video_disabled"] = True
        self._write_payload(payload)

    def startup_video_enabled(self) -> bool:
        return self._load_payload().get("startup_video_disabled") is not True

    def _save_values(self, values: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._load_payload()
        payload.update(values)
        self._write_payload(payload)

    def _write_payload(self, payload: dict) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
