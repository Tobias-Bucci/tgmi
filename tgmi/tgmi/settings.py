"""Application settings management."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from .constants import CONFIG_PATH, DEFAULT_API_KEY, DEFAULT_HISTORY_PATH, DEFAULT_MODEL, DEFAULT_REQUEST_TIMEOUT


@dataclass
class Settings:
    """Persisted configuration options for the terminal client."""

    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    output_format: str = "markdown"
    history_path: str = str(DEFAULT_HISTORY_PATH)
    extended_thinking: bool = False
    hacking_mode: bool = False
    google_search: bool = True
    max_output_tokens: int | None = None
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT


class SettingsManager:
    """Load and persist settings on disk."""

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = path
        self._model_migrated = False
        self.settings = self._load()
        if self._model_migrated:
            self.save()

    def _load(self) -> Settings:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                settings = Settings(**data)
                if settings.model == "gemini-1.5-flash-latest":
                    settings.model = DEFAULT_MODEL
                    self._model_migrated = True
                return settings
            except (json.JSONDecodeError, TypeError):
                Console().print(
                    Panel.fit("Could not read <config.json> – using default values.", style="red")
                )
        return Settings()

    def save(self) -> None:
        payload = json.dumps(asdict(self.settings), indent=2, ensure_ascii=False)
        self.path.write_text(payload, encoding="utf-8")

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self.settings, key) and value is not None:
                setattr(self.settings, key, value)
        self.save()
