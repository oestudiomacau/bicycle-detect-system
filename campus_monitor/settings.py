from __future__ import annotations

import json
from pathlib import Path

from .domain import DEFAULT_PARKING_ZONE, validate_parking_zone


class SettingsStore:
    """Persist the small set of user-adjustable monitoring settings."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load_parking_zone(self) -> tuple[float, float, float, float]:
        if not self.path.exists():
            return DEFAULT_PARKING_ZONE
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return validate_parking_zone(tuple(payload["parking_zone"]))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return DEFAULT_PARKING_ZONE

    def save_parking_zone(self, zone: tuple[float, float, float, float]) -> None:
        validated = validate_parking_zone(zone)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {}
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload.update(existing)
            except (OSError, json.JSONDecodeError):
                pass
        payload["parking_zone"] = list(validated)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
