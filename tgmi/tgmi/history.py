"""Chat history persistence and export helpers."""
from __future__ import annotations

import json
import re
import uuid
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
        self.sessions: List[Dict[str, Any]] = []
        self.current_session_id: Optional[str] = None
        self.load()

    @property
    def entries(self) -> List[Dict[str, str]]:
        """Return messages of the current session."""
        session = self._get_current_session()
        if session:
            return session.get("messages", [])
        return []

    def _get_current_session(self) -> Optional[Dict[str, Any]]:
        if not self.current_session_id:
            return None
        for session in self.sessions:
            if session["id"] == self.current_session_id:
                return session
        return None

    def load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    # Migration from old format
                    self._migrate_from_list(data)
                elif isinstance(data, dict) and "sessions" in data:
                    self.sessions = data["sessions"]
                    self.current_session_id = data.get("current_session_id")
                    # Ensure valid current session
                    if not self._get_current_session() and self.sessions:
                        self.current_session_id = self.sessions[-1]["id"]
                else:
                    # Unknown format or empty dict
                    self.sessions = []
                    self.current_session_id = None
            except json.JSONDecodeError:
                Console().print(Panel.fit("Historie konnte nicht gelesen werden – wird zurückgesetzt.", style="yellow"))
                self.sessions = []
                self.current_session_id = None
        else:
            self.sessions = []
            self.current_session_id = None
        
        if not self.sessions:
            self.start_new_session()

    def _migrate_from_list(self, old_entries: List[Dict[str, str]]) -> None:
        session_id = str(uuid.uuid4())
        self.sessions = [{
            "id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": old_entries
        }]
        self.current_session_id = session_id
        self.save()

    def start_new_session(self) -> None:
        session_id = str(uuid.uuid4())
        new_session = {
            "id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": []
        }
        self.sessions.append(new_session)
        self.current_session_id = session_id
        self.save()

    def add_entry(self, role: str, content: str) -> None:
        session = self._get_current_session()
        if not session:
            self.start_new_session()
            session = self._get_current_session()
        
        # Ensure messages list exists
        if "messages" not in session:
            session["messages"] = []

        session["messages"].append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def clear(self) -> None:
        session = self._get_current_session()
        if session:
            session["messages"] = []
            self.save()

    def save(self) -> None:
        data = {
            "sessions": self.sessions,
            "current_session_id": self.current_session_id
        }
        payload = json.dumps(data, indent=2, ensure_ascii=False)
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
