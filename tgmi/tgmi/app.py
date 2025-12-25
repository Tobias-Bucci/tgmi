"""Console application flow for the Gemini terminal client."""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.spinner import Spinner
from rich.table import Table

from .client import GeminiClient
from .clipboard import ClipboardHelper
from .constants import (
    MESSAGES,
    SUGGESTED_MODELS,
    THINKING_GENERATION_CONFIG_BASE,
    THINKING_MODELS,
    THINKING_SYSTEM_PROMPT,
    HACKING_MODE_PROMPT,
)
from .history import ChatHistoryManager
from .settings import SettingsManager


class TerminalApp:
    """Handle user interaction inside the terminal."""

    FILE_COMMAND_PATTERN = re.compile(r"^\s*([#!])\s+(.+)$")
    CODE_FENCE_MAP = {
        ".py": "python",
        ".pyw": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".json": "json",
        ".md": "markdown",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".ps1": "powershell",
        ".psm1": "powershell",
        ".bat": "batch",
        ".cmd": "batch",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".rs": "rust",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".sql": "sql",
    }

    def __init__(self) -> None:
        self.console = Console()
        self.settings_manager = SettingsManager()
        self.history_manager = ChatHistoryManager(Path(self.settings_manager.settings.history_path))
        settings = self.settings_manager.settings
        self.client = GeminiClient(
            api_key=settings.api_key,
            model=settings.model,
            timeout=settings.request_timeout,
            max_output_tokens_retry=self._determine_retry_limit(),
        )
        self._ensure_api_key()
        self.workspace_root = Path.cwd()
        self._last_response: Optional[str] = None

    def _ensure_api_key(self) -> None:
        if not self.settings_manager.settings.api_key:
            self.console.print(Panel.fit(MESSAGES["missing_key"], style="red"))

    def run(self) -> None:
        self._print_welcome()
        self.show_help()
        while True:
            user_input = self.console.input(MESSAGES["prompt"]).strip()
            if not user_input:
                continue
            if user_input.startswith(":"):
                if self.handle_shortcut(user_input[1:].lower()):
                    break
                continue
            self.process_user_message(user_input)

    def _print_welcome(self) -> None:
        welcome_panel = Panel(
            MESSAGES["welcome"],
            title="Gemini",
            border_style="cyan",
            box=box.ROUNDED,
        )
        self.console.print(welcome_panel)

    def show_help(self) -> None:
        help_text = f"{MESSAGES['help_text']}\n\n{MESSAGES['file_command_hint']}"
        help_panel = Panel(help_text, title=MESSAGES["help_title"], border_style="green", box=box.ROUNDED)
        self.console.print(help_panel)

    def handle_shortcut(self, command: str) -> bool:
        if command == "q":
            self.console.print(Panel.fit(MESSAGES["goodbye"], style="magenta"))
            return True
        if command == "h":
            self.show_help()
        elif command == "s":
            self.history_manager.save()
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.console.print(Panel.fit(MESSAGES["status_saved_at"].format(time=timestamp), style="green"))
        elif command == "l":
            self.history_manager.load()
            self.console.print(Panel.fit(MESSAGES["history_loaded"], style="green"))
        elif command == "c":
            if Confirm.ask(MESSAGES["confirm_clear"], default=False):
                self.history_manager.clear()
                self.console.print(Panel.fit(MESSAGES["history_cleared"], style="yellow"))
        elif command == "n":
            self.history_manager.start_new_session()
            self.console.print(Panel.fit(MESSAGES["new_chat_started"], style="green"))
        elif command == "x":
            self.export_history_markdown()
        elif command == "f":
            self.search_history()
        elif command == "i":
            self.show_history_insights()
        elif command == "o":
            self.open_options()
        elif command == "cp":
            self.copy_last_response()
        else:
            self.console.print(Panel.fit(MESSAGES["invalid_choice"], style="yellow"))
        return False

    def open_options(self) -> None:
        while True:
            table = Table(title=MESSAGES["options_title"], box=box.SIMPLE, show_edge=False)
            table.add_column("#", style="cyan")
            table.add_column("Option", style="white")
            for row in MESSAGES["options_menu"].splitlines():
                number, text = row.split(") ", maxsplit=1)
                table.add_row(number, text)
            table.add_row(
                "-",
                MESSAGES["current_format"].format(fmt=self.settings_manager.settings.output_format),
            )
            table.add_row(
                "-",
                MESSAGES["current_model"].format(model=self.settings_manager.settings.model),
            )
            max_tokens = self.settings_manager.settings.max_output_tokens
            if isinstance(max_tokens, int) and max_tokens > 0:
                max_tokens_label = str(max_tokens)
            else:
                max_tokens_label = MESSAGES["max_tokens_unlimited"]
            table.add_row(
                "-",
                MESSAGES["current_max_tokens"].format(value=max_tokens_label),
            )
            table.add_row(
                "-",
                MESSAGES["current_timeout"].format(seconds=self.settings_manager.settings.request_timeout),
            )
            supports_thinking = self.settings_manager.settings.model in THINKING_MODELS
            if self.settings_manager.settings.extended_thinking and not supports_thinking:
                thinking_state = f"{MESSAGES['thinking_state_off']} ({MESSAGES['thinking_state_blocked']})"
            elif self.is_thinking_enabled():
                thinking_state = MESSAGES["thinking_state_on"]
            else:
                thinking_state = MESSAGES["thinking_state_off"]
            table.add_row(
                "-",
                MESSAGES["current_thinking"].format(state=thinking_state),
            )
            
            if self.settings_manager.settings.hacking_mode:
                hacking_state = MESSAGES["hacking_mode_on"]
            else:
                hacking_state = MESSAGES["hacking_mode_off"]
            table.add_row(
                "-",
                MESSAGES["current_hacking_mode"].format(state=hacking_state),
            )

            if self.settings_manager.settings.google_search:
                search_state = MESSAGES["google_search_on"]
            else:
                search_state = MESSAGES["google_search_off"]
            table.add_row(
                "-",
                MESSAGES["current_google_search"].format(state=search_state),
            )

            self.console.print(table)

            choice = self.console.input(MESSAGES["enter_choice"]).strip()
            if choice == "1":
                self.change_model()
            elif choice == "2":
                self.toggle_thinking_mode()
            elif choice == "3":
                self.toggle_hacking_mode()
            elif choice == "4":
                self.toggle_google_search()
            elif choice == "5":
                self.change_output_format()
            elif choice == "6":
                self.change_max_tokens()
            elif choice == "7":
                self.change_timeout()
            elif choice == "8":
                self.update_api_key()
            elif choice == "9":
                break
            else:
                self.console.print(Panel.fit(MESSAGES["invalid_choice"], style="yellow"))

    def update_api_key(self) -> None:
        new_key = self.console.input(MESSAGES["new_api_key"]).strip()
        if new_key:
            self.settings_manager.update(api_key=new_key)
            self.client.update_key(new_key)
            self.console.print(Panel.fit(MESSAGES["api_updated"], style="green"))
        else:
            self.console.print(Panel.fit(MESSAGES["invalid_choice"], style="yellow"))

    def change_output_format(self) -> None:
        new_format = self.console.input(MESSAGES["format_prompt"]).strip().lower()
        if new_format in {"plain", "markdown"}:
            self.settings_manager.update(output_format=new_format)
            self.console.print(Panel.fit(MESSAGES["format_updated"], style="green"))
        else:
            self.console.print(Panel.fit(MESSAGES["invalid_choice"], style="yellow"))

    def change_model(self) -> None:
        models_hint = ", ".join(SUGGESTED_MODELS)
        self.console.print(Panel.fit(MESSAGES["available_models"].format(models=models_hint), style="cyan"))
        new_model = self.console.input(MESSAGES["model_prompt"]).strip()
        if new_model:
            self.settings_manager.update(model=new_model)
            self.client.update_model(new_model)
            self.console.print(Panel.fit(MESSAGES["model_updated"], style="green"))
            if self.settings_manager.settings.extended_thinking:
                if new_model in THINKING_MODELS:
                    self.console.print(Panel.fit(MESSAGES["thinking_enabled"], style="cyan"))
                else:
                    self.console.print(Panel.fit(MESSAGES["thinking_unavailable"], style="yellow"))
        else:
            self.console.print(Panel.fit(MESSAGES["invalid_choice"], style="yellow"))

    def is_thinking_enabled(self) -> bool:
        return (
            self.settings_manager.settings.extended_thinking
            and self.settings_manager.settings.model in THINKING_MODELS
        )

    def toggle_thinking_mode(self) -> None:
        model = self.settings_manager.settings.model
        if model not in THINKING_MODELS:
            self.console.print(Panel.fit(MESSAGES["thinking_unavailable"], style="yellow"))
            return

        new_state = not self.settings_manager.settings.extended_thinking
        self.settings_manager.update(extended_thinking=new_state)
        if new_state:
            self.console.print(Panel.fit(MESSAGES["thinking_enabled"], style="green"))
        else:
            self.console.print(Panel.fit(MESSAGES["thinking_disabled"], style="green"))

    def toggle_hacking_mode(self) -> None:
        new_state = not self.settings_manager.settings.hacking_mode
        self.settings_manager.update(hacking_mode=new_state)

        # Always start a new session when toggling
        self.history_manager.start_new_session()

        if new_state:
            self.console.print(Panel.fit(MESSAGES["hacking_mode_enabled"], style="green"))
            self.process_user_message(HACKING_MODE_PROMPT)
        else:
            self.console.print(Panel.fit(MESSAGES["hacking_mode_disabled"], style="green"))

    def toggle_google_search(self) -> None:
        new_state = not self.settings_manager.settings.google_search
        self.settings_manager.update(google_search=new_state)
        if new_state:
            self.console.print(Panel.fit(MESSAGES["google_search_enabled"], style="green"))
        else:
            self.console.print(Panel.fit(MESSAGES["google_search_disabled"], style="green"))

    def _determine_retry_limit(self) -> Optional[int]:
        max_tokens = self.settings_manager.settings.max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            return max_tokens
        return None

    def _build_generation_config(self, thinking_enabled: bool) -> Optional[Dict[str, Any]]:
        config: Dict[str, Any] = {}
        if thinking_enabled:
            config.update(THINKING_GENERATION_CONFIG_BASE)
        max_tokens = self.settings_manager.settings.max_output_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            config["maxOutputTokens"] = max_tokens
        return config or None

    def change_max_tokens(self) -> None:
        user_input = self.console.input(MESSAGES["max_tokens_prompt"]).strip()
        if not user_input:
            self.settings_manager.update(max_output_tokens=None)
            self.client.update_max_output_tokens_retry(self._determine_retry_limit())
            self.console.print(Panel.fit(MESSAGES["max_tokens_cleared"], style="green"))
            return
        try:
            value = int(user_input)
            if value <= 0:
                raise ValueError
        except ValueError:
            self.console.print(Panel.fit(MESSAGES["invalid_choice"], style="yellow"))
            return
        self.settings_manager.update(max_output_tokens=value)
        self.client.update_max_output_tokens_retry(self._determine_retry_limit())
        self.console.print(Panel.fit(MESSAGES["max_tokens_updated"], style="green"))

    def change_timeout(self) -> None:
        user_input = self.console.input(MESSAGES["timeout_prompt"]).strip()
        try:
            value = int(user_input)
            if value <= 0:
                raise ValueError
        except (ValueError, TypeError):
            self.console.print(Panel.fit(MESSAGES["invalid_choice"], style="yellow"))
            return
        self.settings_manager.update(request_timeout=value)
        self.client.update_timeout(value)
        self.console.print(Panel.fit(MESSAGES["timeout_updated"], style="green"))

    def export_history_markdown(self) -> None:
        if not self.history_manager.entries:
            self.console.print(Panel.fit(MESSAGES["history_empty_export"], style="yellow"))
            return

        default_path = Path(self.settings_manager.settings.history_path).with_suffix(".md")
        user_input = self.console.input(MESSAGES["export_prompt"].format(path=default_path)).strip()
        export_path = Path(user_input).expanduser() if user_input else default_path

        try:
            self.history_manager.export_markdown(
                export_path,
                role_labels=self._get_role_labels_for_export(),
            )
        except ValueError:
            self.console.print(Panel.fit(MESSAGES["history_empty_export"], style="yellow"))
        except Exception as exc:
            self.console.print(
                Panel.fit(
                    MESSAGES["history_export_failed"].format(error=str(exc)),
                    title=MESSAGES["error"],
                    style="red",
                )
            )
        else:
            display_path = export_path.resolve()
            self.console.print(
                Panel.fit(MESSAGES["history_exported"].format(path=display_path), style="green")
            )

    def _get_role_labels_for_export(self) -> Dict[str, str]:
        return {
            "user": MESSAGES.get("role_user", "User"),
            "assistant": MESSAGES.get("role_assistant", "Assistant"),
            "model": MESSAGES.get("role_assistant", "Assistant"),
            "system": MESSAGES.get("role_system", "System"),
        }

    def search_history(self) -> None:
        if not self.history_manager.entries:
            self.console.print(Panel.fit(MESSAGES["history_empty"], style="yellow"))
            return

        query = self.console.input(MESSAGES["search_prompt"]).strip()
        if not query:
            return

        labels = self._get_role_labels_for_export()
        table = self.history_manager.search(
            query,
            labels,
            column_titles={
                "title": MESSAGES["search_results_title"],
                "index": MESSAGES["search_result_column_index"],
                "role": MESSAGES["search_result_column_role"],
                "time": MESSAGES["search_result_column_time"],
                "snippet": MESSAGES["search_result_column_snippet"],
            },
        )

        if len(table.rows) == 0:
            self.console.print(Panel.fit(MESSAGES["search_no_results"], style="yellow"))
            return

        self.console.print(table)

    def show_history_insights(self) -> None:
        entries = self.history_manager.entries
        if not entries:
            self.console.print(Panel.fit(MESSAGES["insights_no_data"], style="yellow"))
            return

        total_messages = len(entries)
        user_messages = [e for e in entries if e.get("role") == "user"]
        assistant_messages = [
            e for e in entries if e.get("role") in {"assistant", "model"}
        ]

        first_ts = self._parse_timestamp(entries[0].get("timestamp"))
        last_ts = self._parse_timestamp(entries[-1].get("timestamp"))

        if first_ts and last_ts:
            span = f"{ChatHistoryManager._format_timestamp(entries[0].get('timestamp', ''))} – " \
                   f"{ChatHistoryManager._format_timestamp(entries[-1].get('timestamp', ''))}"
        else:
            span = "—"

        avg_user_length = self._compute_average_length(user_messages)
        avg_response_time = self._compute_average_response_time(entries)

        table = Table(
            title=MESSAGES["insights_title"], box=box.ROUNDED, show_edge=False, show_header=False
        )
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row(MESSAGES["insights_total"], str(total_messages))
        table.add_row(MESSAGES["insights_user"], str(len(user_messages)))
        table.add_row(MESSAGES["insights_assistant"], str(len(assistant_messages)))
        table.add_row(MESSAGES["insights_span"], span)
        table.add_row(
            MESSAGES["insights_avg_user_length"],
            self._format_length(avg_user_length),
        )
        table.add_row(
            MESSAGES["insights_avg_response_time"],
            self._format_seconds(avg_response_time),
        )

        self.console.print(table)

    def _compute_average_length(self, messages: List[Dict[str, Any]]) -> Optional[float]:
        lengths = [len(m.get("content", "")) for m in messages if isinstance(m.get("content"), str)]
        if not lengths:
            return None
        return sum(lengths) / len(lengths)

    def _compute_average_response_time(self, entries: List[Dict[str, Any]]) -> Optional[float]:
        deltas: List[float] = []
        pending_user_ts: Optional[datetime] = None
        for entry in entries:
            role = entry.get("role")
            ts = self._parse_timestamp(entry.get("timestamp"))
            if ts is None:
                continue
            if role == "user":
                pending_user_ts = ts
            elif role in {"assistant", "model"} and pending_user_ts is not None:
                delta = (ts - pending_user_ts).total_seconds()
                if delta >= 0:
                    deltas.append(delta)
                pending_user_ts = None
        if not deltas:
            return None
        return sum(deltas) / len(deltas)

    def _format_length(self, value: Optional[float]) -> str:
        if value is None:
            return "—"
        return f"{value:.0f} {MESSAGES['unit_characters']}"

    def _format_seconds(self, value: Optional[float]) -> str:
        if value is None:
            return "—"
        return f"{value:.1f} {MESSAGES['unit_seconds']}"

    def _parse_timestamp(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt

    def process_user_message(self, message: str) -> None:
        if not self.settings_manager.settings.api_key:
            self.console.print(Panel.fit(MESSAGES["missing_key"], style="red"))
            return

        prepared_message, attached_paths, attachment_errors = self._prepare_message_with_files(message)
        for error in attachment_errors:
            self.console.print(Panel.fit(error, style="red", border_style="red", title=MESSAGES["error"]))
        for path in attached_paths:
            display = self._format_display_path(path)
            self.console.print(Panel.fit(MESSAGES["file_attached"].format(path=display), style="cyan"))

        if not prepared_message:
            self.console.print(Panel.fit(MESSAGES["message_empty"], style="yellow"))
            return

        thinking_enabled = self.is_thinking_enabled()
        hacking_enabled = self.settings_manager.settings.hacking_mode

        system_instruction_parts = []
        if hacking_enabled:
            system_instruction_parts.append(HACKING_MODE_PROMPT)
        if thinking_enabled:
            system_instruction_parts.append(THINKING_SYSTEM_PROMPT)
        
        system_instruction = "\n\n".join(system_instruction_parts) if system_instruction_parts else None

        generation_config = self._build_generation_config(thinking_enabled)
        status_message = MESSAGES["thinking_in_progress"] if thinking_enabled else MESSAGES["sending"]
        spinner = Spinner("dots", text=status_message)
        status_panel = Panel(spinner, border_style="blue", box=box.ROUNDED)
        try:
            with Live(status_panel, refresh_per_second=12, console=self.console, transient=True):
                response = self.client.send_message(
                    self.history_manager.build_gemini_payload(),
                    prepared_message,
                    system_instruction=system_instruction,
                    generation_config=generation_config,
                    google_search=self.settings_manager.settings.google_search,
                )
        except RuntimeError as exc:
            self.console.print(Panel.fit(MESSAGES["http_error"].format(detail=str(exc)), title=MESSAGES["error"], style="red"))
            return

        self.history_manager.add_entry("user", prepared_message)
        self.history_manager.add_entry("assistant", response)
        self.history_manager.save()
        self.display_response(response)

    def _prepare_message_with_files(self, raw_message: str) -> Tuple[str, List[Path], List[str]]:
        attachments: List[Tuple[Path, str]] = []
        errors: List[str] = []
        lines: List[str] = []

        for line in raw_message.splitlines():
            match = self.FILE_COMMAND_PATTERN.match(line)
            if not match:
                lines.append(line)
                continue

            path_string = match.group(2).strip()
            if not path_string:
                lines.append(line)
                continue

            cleaned_path = self._strip_quotes(path_string)
            if not self._looks_like_path(cleaned_path):
                lines.append(line)
                continue
            attachment = self._read_attachment(cleaned_path)
            if isinstance(attachment, tuple):
                path_obj, content = attachment
                attachments.append((path_obj, content))
            else:
                errors.append(attachment)

        base_message = "\n".join(lines).strip()
        attachment_blocks = [self._format_attachment_block(path, content) for path, content in attachments]
        combined_message = base_message
        if attachment_blocks:
            attachments_text = "\n\n".join(attachment_blocks)
            combined_message = f"{base_message}\n\n{attachments_text}" if base_message else attachments_text

        attached_paths = [path for path, _ in attachments]
        return combined_message, attached_paths, errors

    def _strip_quotes(self, value: str) -> str:
        if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        return value

    def _read_attachment(self, path_string: str) -> Tuple[Path, str] | str:
        candidate = Path(path_string).expanduser()
        if not candidate.is_absolute():
            candidate = (self.workspace_root / candidate).resolve(strict=False)
        else:
            candidate = candidate.resolve(strict=False)

        if not candidate.exists():
            return MESSAGES["file_not_found"].format(path=str(candidate))
        if not candidate.is_file():
            return MESSAGES["file_not_file"].format(path=str(candidate))

        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                return MESSAGES["file_read_error"].format(path=str(candidate), error=str(exc))
        except OSError as exc:
            return MESSAGES["file_read_error"].format(path=str(candidate), error=str(exc))

        return candidate, text

    def _format_attachment_block(self, path: Path, content: str) -> str:
        fence = self._detect_code_fence(path)
        header = f"File: {path}"
        body = content.rstrip("\n")
        if not body:
            body = "(empty file)"
        return f"{header}\n```{fence}\n{body}\n```"

    def _detect_code_fence(self, path: Path) -> str:
        suffix = path.suffix.lower()
        return self.CODE_FENCE_MAP.get(suffix, "text")

    def _format_display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return str(path)

    def _looks_like_path(self, candidate: str) -> bool:
        if not candidate:
            return False
        if any(sep in candidate for sep in ("/", "\\")):
            return True
        if "." in candidate:
            return True
        return False

    def display_response(self, response: str) -> None:
        self._last_response = response
        if self.settings_manager.settings.output_format == "markdown":
            rendered = Markdown(response)
        else:
            rendered = response
        
        title = f"{MESSAGES['model_label']} ({self.settings_manager.settings.model})"
        if self.settings_manager.settings.hacking_mode:
            title += " [HACKING MODE]"
            
        panel = Panel(rendered, title=title, border_style="magenta", box=box.ROUNDED)
        self.console.print(panel)

    def copy_last_response(self) -> None:
        if not self._last_response:
            self.console.print(Panel.fit(MESSAGES["copy_no_response"], style="yellow"))
            return
        success = ClipboardHelper.copy_text(self._last_response)
        if success:
            self.console.print(Panel.fit(MESSAGES["copy_success"], style="green"))
        else:
            self.console.print(Panel.fit(MESSAGES["copy_failure"], style="yellow"))


def main() -> None:
    app = TerminalApp()
    try:
        app.run()
    except KeyboardInterrupt:
        Console().print("\nProgramm beendet.")
        if app.settings_manager.settings.hacking_mode:
            app.settings_manager.update(hacking_mode=False)
        time.sleep(0.2)
        sys.exit(0)
