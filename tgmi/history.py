"""Chat history persistence and export helpers."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class ChatHistoryManager:
    """Store, load, and query conversation history."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: List[Dict[str, str]] = []
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                Console().print(Panel.fit("Historie konnte nicht gelesen werden – wird zurückgesetzt.", style="yellow"))
                self.entries = []
        else:
            self.entries = []

    def add_entry(self, role: str, content: str) -> None:
        self.entries.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def clear(self) -> None:
        self.entries.clear()
        self.save()

    def save(self) -> None:
        payload = json.dumps(self.entries, indent=2, ensure_ascii=False)
        self.path.write_text(payload, encoding="utf-8")

    def export_markdown(
        self,
        path: Path,
        *,
        role_labels: Optional[Dict[str, str]] = None,
        include_timestamps: bool = True,
    ) -> None:
        if not self.entries:
            raise ValueError("empty")

        role_labels = role_labels or {}
        lines: List[str] = ["# Gemini Chat Transcript"]
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        lines.append(f"_Generated: {generated_at}_")
        lines.append("")

        for item in self.entries:
            role = item.get("role", "assistant")
            label = role_labels.get(role, role.capitalize())

            heading = f"### {label}"
            timestamp = item.get("timestamp")
            if include_timestamps and isinstance(timestamp, str):
                formatted_ts = self._format_timestamp(timestamp)
                if formatted_ts:
                    heading += f" ({formatted_ts})"

            lines.append(heading)
            lines.append("")
            content = item.get("content")
            lines.append(content if isinstance(content, str) and content else "_(no content)_")
            lines.append("")

        markdown = "\n".join(lines).rstrip() + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")

    def build_gemini_payload(self) -> List[Dict[str, object]]:
        payload: List[Dict[str, object]] = []
        for item in self.entries:
            payload.append(
                {
                    "role": "user" if item["role"] == "user" else "model",
                    "parts": [{"text": item["content"]}],
                }
            )
        return payload

    @staticmethod
    def _format_timestamp(value: str) -> Optional[str]:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return value

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")

    def search(
        self,
        query: str,
        labels: Dict[str, str],
        *,
        column_titles: Dict[str, str],
        radius: int = 60,
    ) -> Table:
        query_lower = query.lower()
        table = Table(title=column_titles.get("title", "Search results"))
        table.add_column(column_titles.get("index", "#"), style="cyan", justify="right")
        table.add_column(column_titles.get("role", "Role"), style="magenta")
        table.add_column(column_titles.get("time", "Time"), style="green")
        table.add_column(column_titles.get("snippet", "Snippet"), style="white")

        for index, entry in enumerate(self.entries, start=1):
            content = entry.get("content", "")
            if not isinstance(content, str):
                continue
            if query_lower not in content.lower():
                continue
            ts_str = self._format_timestamp(entry.get("timestamp", "")) or "—"
            role = entry.get("role", "assistant")
            label = labels.get(role, role.capitalize())
            snippet = self._build_highlight_snippet(content, query, radius)
            table.add_row(str(index), label, ts_str, snippet)

        return table

    def _build_highlight_snippet(self, content: str, query: str, radius: int = 60) -> Text:
        content_single_line = re.sub(r"\s+", " ", content.strip())
        lower_content = content_single_line.lower()
        lower_query = query.lower()
        index = lower_content.find(lower_query)
        if index == -1:
            snippet = content_single_line[: radius * 2 + len(query)]
        else:
            start = max(index - radius, 0)
            end = min(index + len(query) + radius, len(content_single_line))
            snippet = content_single_line[start:end]
            if start > 0:
                snippet = "…" + snippet
            if end < len(content_single_line):
                snippet = snippet + "…"

        text = Text(snippet)
        pattern = re.compile(re.escape(query), flags=re.IGNORECASE)
        text.highlight_regex(pattern, style="bold magenta")
        return text
