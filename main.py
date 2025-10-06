"""Terminalbasierter Gemini-Chat-Client.

Dieses Skript stellt einen plattformunabhängigen Chat-Client für die Google Gemini API
bereit. Es nutzt Rich für eine farbige Terminalausgabe und Requests für HTTP-Calls.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

# Konstante Konfigurationspfade
CONFIG_PATH = Path("config.json")
DEFAULT_HISTORY_PATH = Path("history.json")
DEFAULT_MODEL = "gemini-2.5-flash"
# Vom Nutzer bereitgestellter API-Key
DEFAULT_API_KEY = "AIzaSyAuu8RN779mRyz5p_k7bUMRvJmC6xuPQDA"
DEFAULT_REQUEST_TIMEOUT = 120
SUGGESTED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-live-2.5-flash",
    "gemini-live-2.5-flash-preview-native-audio",
]
THINKING_MODELS = {
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
}
THINKING_SYSTEM_PROMPT = (
    "You are an expert assistant with advanced reasoning abilities. Take the time to think deeply, "
    "analyse the task step by step, consider alternative perspectives, and only then craft a concise "
    "final answer for the user. Use structured reasoning internally and return only the final response."
)
THINKING_GENERATION_CONFIG_BASE = {
    "temperature": 0.6,
    "topP": 0.95,
}

# Sprachpakete für das Interface
LANG = {
    "de": {
        "welcome": "Willkommen beim Gemini Terminal Client",
        "prompt": "[bold cyan]Du:[/bold cyan] ",
        "model_label": "Gemini",
        "options_title": "Optionen",
    "options_menu": (
            "1) API-Schlüssel ändern\n"
            "2) Sprache ändern\n"
            "3) Ausgabeformat ändern\n"
            "4) Modell ändern\n"
            "5) Denkmode umschalten\n"
            "6) Token-Limit einstellen\n"
            "7) Timeout einstellen\n"
            "8) Zurück"
        ),
    "enter_choice": "Wähle eine Option (1-8): ",
        "new_api_key": "Neuen API-Schlüssel eingeben: ",
        "api_updated": "API-Schlüssel wurde aktualisiert.",
        "language_updated": "Sprache wurde geändert.",
        "format_updated": "Ausgabeformat wurde geändert.",
        "current_language": "Aktuelle Sprache: {lang}",
        "current_format": "Ausgabeformat: {fmt}",
    "current_max_tokens": "Maximale Antwort-Tokens: {value}",
    "current_timeout": "HTTP-Timeout: {seconds} s",
    "current_model": "Aktuelles Modell: {model}",
    "available_models": "Verfügbare Modelle (Beispiele): {models}",
    "model_prompt": "Modellname eingeben: ",
    "model_updated": "Modell wurde geändert.",
    "max_tokens_prompt": "Maximale Antwort-Tokens eingeben (leer für unbegrenzt): ",
    "max_tokens_updated": "Token-Limit wurde aktualisiert.",
    "max_tokens_cleared": "Token-Limit wurde entfernt.",
    "max_tokens_unlimited": "unbegrenzt",
    "timeout_prompt": "Timeout in Sekunden eingeben: ",
    "timeout_updated": "Timeout wurde aktualisiert.",
    "current_thinking": "Denkmode: {state}",
    "thinking_state_on": "aktiv",
    "thinking_state_off": "deaktiviert",
    "thinking_state_blocked": "nicht verfügbar für aktuelles Modell",
    "thinking_enabled": "Denkmode aktiviert – das Modell nutzt längere Überlegungen.",
    "thinking_disabled": "Denkmode deaktiviert.",
    "thinking_unavailable": "Für dieses Modell ist der Denkmode nicht verfügbar.",
        "history_saved": "Chat-Historie gespeichert.",
        "history_loaded": "Chat-Historie geladen.",
        "history_cleared": "Chat-Historie geleert.",
        "confirm_clear": "Möchtest du die Chat-Historie wirklich löschen?",
        "missing_key": "Kein API-Schlüssel konfiguriert. Öffne das Optionen-Menü (:o).",
        "error": "Fehler",
        "http_error": "API-Anfrage fehlgeschlagen: {detail}",
        "help_title": "Schnellbefehle",
        "help_text": (
            "Verfügbare Shortcuts:\n"
            ":q - Beenden\n:s - Historie speichern\n:l - Historie laden\n"
            ":o - Optionen\n:c - Historie leeren\n:h - Hilfe"
        ),
        "output_plain": "Klartext",
        "output_markdown": "Markdown",
        "language_prompt": "Sprache wählen (de/en): ",
        "format_prompt": "Ausgabeformat wählen (plain/markdown): ",
        "invalid_choice": "Ungültige Eingabe. Bitte erneut versuchen.",
        "sending": "Nachricht wird an Gemini gesendet…",
        "no_response": "Die API lieferte keine Antwort.",
        "goodbye": "Auf Wiedersehen!",
        "status_saved_at": "Historie gespeichert um {time}.",
    },
    "en": {
        "welcome": "Welcome to the Gemini Terminal Client",
        "prompt": "[bold cyan]You:[/bold cyan] ",
        "model_label": "Gemini",
        "options_title": "Settings",
    "options_menu": (
            "1) Change API key\n"
            "2) Switch language\n"
            "3) Change output format\n"
            "4) Change model\n"
            "5) Toggle thinking mode\n"
            "6) Set token limit\n"
            "7) Set timeout\n"
            "8) Back"
        ),
    "enter_choice": "Choose an option (1-8): ",
        "new_api_key": "Enter new API key: ",
        "api_updated": "API key updated.",
        "language_updated": "Language switched.",
        "format_updated": "Output format updated.",
        "current_language": "Current language: {lang}",
        "current_format": "Output format: {fmt}",
    "current_max_tokens": "Max output tokens: {value}",
    "current_timeout": "HTTP timeout: {seconds} s",
    "current_model": "Current model: {model}",
    "available_models": "Sample models: {models}",
    "model_prompt": "Enter model identifier: ",
    "model_updated": "Model updated.",
    "max_tokens_prompt": "Enter maximum output tokens (leave empty for unlimited): ",
    "max_tokens_updated": "Token limit updated.",
    "max_tokens_cleared": "Token limit removed.",
    "max_tokens_unlimited": "unlimited",
    "timeout_prompt": "Enter timeout in seconds: ",
    "timeout_updated": "Timeout updated.",
    "current_thinking": "Thinking mode: {state}",
    "thinking_state_on": "enabled",
    "thinking_state_off": "disabled",
    "thinking_state_blocked": "not available for current model",
    "thinking_enabled": "Thinking mode enabled – the model will take extra reasoning time.",
    "thinking_disabled": "Thinking mode disabled.",
    "thinking_unavailable": "This model does not support the extended thinking mode.",
        "history_saved": "Chat history saved.",
        "history_loaded": "Chat history reloaded.",
        "history_cleared": "Chat history cleared.",
        "confirm_clear": "Do you really want to delete the chat history?",
        "missing_key": "No API key configured. Open the settings menu (:o).",
        "error": "Error",
        "http_error": "API request failed: {detail}",
        "help_title": "Shortcuts",
        "help_text": (
            "Available shortcuts:\n"
            ":q - Quit\n:s - Save history\n:l - Load history\n"
            ":o - Settings\n:c - Clear history\n:h - Help"
        ),
        "output_plain": "Plain",
        "output_markdown": "Markdown",
        "language_prompt": "Select language (de/en): ",
        "format_prompt": "Select output format (plain/markdown): ",
        "invalid_choice": "Invalid choice. Try again.",
        "sending": "Sending message to Gemini…",
        "no_response": "The API returned no content.",
        "goodbye": "Goodbye!",
        "status_saved_at": "History saved at {time}.",
    },
}


@dataclass
class Settings:
    """Hält alle konfigurierbaren Einstellungen der Anwendung."""

    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    language: str = "en"
    output_format: str = "markdown"
    history_path: str = str(DEFAULT_HISTORY_PATH)
    extended_thinking: bool = False
    max_output_tokens: Optional[int] = None
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT


class SettingsManager:
    """Verwaltet das Laden und Speichern der Anwendungseinstellungen."""

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        self.path = path
        self._model_migrated = False
        # Einstellungen beim Start aus der JSON-Datei laden
        self.settings = self._load()
        if self._model_migrated:
            # Migration persistieren, damit zukünftige Starts das neue Modell nutzen
            self.save()

    def _load(self) -> Settings:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                settings = Settings(**data)
                # Migration auf neues Standardmodell ohne "-latest"-Suffix
                if settings.model == "gemini-1.5-flash-latest":
                    settings.model = DEFAULT_MODEL
                    self._model_migrated = True
                return settings
            except (json.JSONDecodeError, TypeError):
                # Fallback bei beschädigter Datei
                Console().print(
                    Panel
                    .fit("<config.json> konnte nicht gelesen werden – Standardwerte werden genutzt.", style="red")
                )
        # Standardwerte, sofern keine Konfiguration vorliegt
        return Settings()

    def save(self) -> None:
        # Persistiert die Einstellungen dauerhaft auf die Festplatte
        self.path.write_text(json.dumps(asdict(self.settings), indent=2, ensure_ascii=False), encoding="utf-8")

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self.settings, key) and value is not None:
                setattr(self.settings, key, value)
        self.save()


class ChatHistoryManager:
    """Speichert und lädt die Chat-Historie in einer JSON-Datei."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: List[Dict[str, str]] = []
        # Beim Start vorhandene Historie lesen, um den Gesprächskontext zu behalten
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
        # Jede Chatnachricht wird mit Zeitstempel abgelegt
        self.entries.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def clear(self) -> None:
        self.entries.clear()
        self.save()

    def save(self) -> None:
        # Historie als JSON sichern
        self.path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False), encoding="utf-8")

    def build_gemini_payload(self) -> List[Dict[str, object]]:
        # Konvertiert die Historie in das von Gemini erwartete Format
        payload: List[Dict[str, object]] = []
        for item in self.entries:
            payload.append(
                {
                    "role": "user" if item["role"] == "user" else "model",
                    "parts": [{"text": item["content"]}],
                }
            )
        return payload


class GeminiClient:
    """Kapselt HTTP-Aufrufe zur Google Gemini API."""

    API_BASES = (
        "https://generativelanguage.googleapis.com/v1",
        "https://generativelanguage.googleapis.com/v1beta",
    )

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        timeout: int = DEFAULT_REQUEST_TIMEOUT,
    max_output_tokens_retry: Optional[int] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.api_base = self.API_BASES[0]
        self.timeout = timeout
        self.max_output_tokens_retry = max_output_tokens_retry

    def send_message(
        self,
        history: List[Dict[str, object]],
        message: str,
        *,
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        allow_max_tokens_retry: bool = True,
    ) -> str:
        """Sendet eine Nachricht an Gemini und gibt den Antworttext zurück."""

        headers = {"Content-Type": "application/json"}
        contents = [*history]
        contents.append({"role": "user", "parts": [{"text": message}]})

        payload: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["system_instruction"] = {
                "role": "system",
                "parts": [{"text": system_instruction}],
            }
        if generation_config:
            payload["generationConfig"] = copy.deepcopy(generation_config)

        bases_to_try = [self.api_base] + [base for base in self.API_BASES if base != self.api_base]
        if system_instruction:
            beta_base = self.API_BASES[-1]
            if beta_base in bases_to_try:
                bases_to_try.remove(beta_base)
            bases_to_try.insert(0, beta_base)
        not_found_messages: List[str] = []

        for base in bases_to_try:
            url = f"{base}/models/{self.model}:generateContent"
            try:
                # HTTP-Anfrage mit Timeout absenden
                response = requests.post(
                    url,
                    headers=headers,
                    params={"key": self.api_key},
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except requests.HTTPError as http_err:
                detail = self._extract_error(http_err.response) or str(http_err)
                status_code = http_err.response.status_code if http_err.response is not None else None
                if status_code == 404:
                    not_found_messages.append(detail)
                    continue
                if (
                    status_code == 400
                    and system_instruction is not None
                    and (
                        "systemInstruction" in (detail or "")
                        or "system_instruction" in (detail or "")
                    )
                    and base != self.API_BASES[-1]
                ):
                    # Fallback auf beta-Endpunkt, das Systeminstruktionen unterstützt
                    continue
                raise RuntimeError(detail) from http_err
            except requests.RequestException as exc:  # Netzwerk- oder sonstige Fehler
                raise RuntimeError(str(exc)) from exc

            data = response.json()
            prompt_feedback = data.get("promptFeedback")
            if isinstance(prompt_feedback, dict):
                block_reason = prompt_feedback.get("blockReason")
                if block_reason:
                    safety_info = self._format_safety_details(prompt_feedback.get("safetyRatings"))
                    message = f"Prompt blocked by safety filters ({block_reason})."
                    if safety_info:
                        message += f" Details: {safety_info}"
                    raise RuntimeError(message)
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("No candidates returned")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason")
            if finish_reason == "SAFETY":
                safety_info = self._format_safety_details(candidate.get("safetyRatings"))
                message = "Response blocked by safety filters."
                if safety_info:
                    message += f" Details: {safety_info}"
                raise RuntimeError(message)
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            texts: List[str] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("text"):
                    texts.append(part["text"])
                elif "functionCall" in part:
                    func = part["functionCall"] or {}
                    name = func.get("name", "function")
                    raw_args = func.get("args") or func.get("arguments")
                    if isinstance(raw_args, (dict, list)):
                        args_text = json.dumps(raw_args, ensure_ascii=False, indent=2)
                    else:
                        args_text = str(raw_args) if raw_args is not None else "{}"
                    texts.append(f"[Function call requested: {name}]\n{args_text}")
                elif "inlineData" in part:
                    data_info = part["inlineData"] or {}
                    mime = data_info.get("mimeType", "application/octet-stream")
                    size = len(data_info.get("data", ""))
                    texts.append(f"[Inline data payload received: {mime}, {size} base64 chars]")
            reply = "\n\n".join(filter(None, texts))
            if not reply:
                if (
                    finish_reason == "MAX_TOKENS"
                    and allow_max_tokens_retry
                ):
                    upgraded_config: Dict[str, Any]
                    retry_limit = self.max_output_tokens_retry
                    if isinstance(retry_limit, int) and retry_limit > 0:
                        if generation_config is None:
                            upgraded_config = {"maxOutputTokens": retry_limit}
                        else:
                            upgraded_config = copy.deepcopy(generation_config)
                            current_max = upgraded_config.get("maxOutputTokens")
                            if not isinstance(current_max, int) or current_max < retry_limit:
                                upgraded_config["maxOutputTokens"] = retry_limit
                            else:
                                upgraded_config = {}
                    else:
                        upgraded_config = {}
                    if upgraded_config:
                        return self.send_message(
                            history,
                            message,
                            system_instruction=system_instruction,
                            generation_config=upgraded_config,
                            allow_max_tokens_retry=False,
                        )
                if finish_reason == "MAX_TOKENS":
                    suggestions = [
                        "• Shorten the latest request or remove repeated context",
                        "• Clear old chat history via :c to free up tokens",
                        "• Increase the token limit via the settings menu (:o) or config.json",
                    ]
                    hint = "\n".join(suggestions)
                    return (
                        "⚠️ Gemini stopped because it hit the maximum output token limit before returning any text.\n"
                        f"Try one of these adjustments:\n{hint}"
                    )
                safety_info = self._format_safety_details(candidate.get("safetyRatings"))
                if safety_info:
                    raise RuntimeError(f"Model produced no text. Safety ratings: {safety_info}")
                if finish_reason and finish_reason != "STOP":
                    raise RuntimeError(
                        f"Model finished with reason '{finish_reason}' but returned no textual content."
                    )
                raise RuntimeError("Empty response returned")

            # Erfolgreiches Base-Endpoint für künftige Aufrufe merken
            self.api_base = base
            return reply

        suggestions = ", ".join(SUGGESTED_MODELS)
        detail_suffix = f" Details: {not_found_messages[-1]}" if not_found_messages else ""
        raise RuntimeError(
            f"Model '{self.model}' not found on the Gemini API endpoints. Try enabling the Generative Language API "
            f"or switch to another model via the settings menu (:o). Suggestions: {suggestions}.{detail_suffix}"
        )

    @staticmethod
    def _extract_error(response: Optional[requests.Response]) -> Optional[str]:
        if response is None:
            return None
        try:
            payload = response.json()
        except ValueError:
            return response.text.strip() or None

        if isinstance(payload, dict):
            if "error" in payload and isinstance(payload["error"], dict):
                error_obj = payload["error"]
                message = error_obj.get("message")
                status = error_obj.get("status")
                if status and message:
                    return f"{status}: {message}"
                return message or None
            if "message" in payload:
                return str(payload["message"])
        return response.text.strip() or None

    def update_key(self, api_key: str) -> None:
        self.api_key = api_key

    def update_model(self, model: str) -> None:
        self.model = model

    def update_timeout(self, timeout: int) -> None:
        self.timeout = timeout

    def update_max_output_tokens_retry(self, value: Optional[int]) -> None:
        self.max_output_tokens_retry = value

    @staticmethod
    def _format_safety_details(ratings: Optional[List[Dict[str, Any]]]) -> str:
        if not ratings:
            return ""
        details: List[str] = []
        for rating in ratings:
            if not isinstance(rating, dict):
                continue
            category = rating.get("category") or "unknown"
            severity = rating.get("severity") or rating.get("probability") or rating.get("probabilityScore")
            if isinstance(severity, dict):
                severity = severity.get("value")
            details.append(f"{category}: {severity}")
        return " | ".join(details)


class TerminalApp:
    """Steuert die Nutzerinteraktion in der Konsole."""

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
        self._refresh_language()

    def _ensure_api_key(self) -> None:
        """Weist auf fehlende API-Schlüssel hin, damit der Nutzer ihn nachträgt."""
        if not self.settings_manager.settings.api_key:
            self.console.print(Panel.fit(self.lang["missing_key"], style="red"))

    def _refresh_language(self) -> None:
        language = self.settings_manager.settings.language.lower()
        if language not in LANG:
            language = "de"
            self.settings_manager.update(language=language)
        self.lang = LANG[language]

    def run(self) -> None:
        self._print_welcome()
        self.show_help()
        while True:
            user_input = self.console.input(self.lang["prompt"]).strip()
            if not user_input:
                continue
            if user_input.startswith(":"):
                if self.handle_shortcut(user_input[1:].lower()):
                    break
                continue
            # Jede reguläre Eingabe wird als Nachricht behandelt
            self.process_user_message(user_input)

    def _print_welcome(self) -> None:
        welcome_panel = Panel(
            self.lang["welcome"],
            title="Gemini",
            border_style="cyan",
            box=box.ROUNDED,
        )
        self.console.print(welcome_panel)

    def show_help(self) -> None:
        help_panel = Panel(self.lang["help_text"], title=self.lang["help_title"], border_style="green", box=box.ROUNDED)
        self.console.print(help_panel)

    def handle_shortcut(self, command: str) -> bool:
        if command == "q":
            self.console.print(Panel.fit(self.lang["goodbye"], style="magenta"))
            return True
        if command == "h":
            self.show_help()
        elif command == "s":
            self.history_manager.save()
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.console.print(Panel.fit(self.lang["status_saved_at"].format(time=timestamp), style="green"))
        elif command == "l":
            self.history_manager.load()
            self.console.print(Panel.fit(self.lang["history_loaded"], style="green"))
        elif command == "c":
            if Confirm.ask(self.lang["confirm_clear"], default=False):
                self.history_manager.clear()
                self.console.print(Panel.fit(self.lang["history_cleared"], style="yellow"))
        elif command == "o":
            self.open_options()
        else:
            self.console.print(Panel.fit(self.lang["invalid_choice"], style="yellow"))
        return False

    def open_options(self) -> None:
        while True:
            table = Table(title=self.lang["options_title"], box=box.SIMPLE, show_edge=False)
            table.add_column("#", style="cyan")
            table.add_column("Option", style="white")
            for row in self.lang["options_menu"].splitlines():
                number, text = row.split(") ", maxsplit=1)
                table.add_row(number, text)
            table.add_row(
                "-",
                self.lang["current_language"].format(lang=self.settings_manager.settings.language.upper()),
            )
            table.add_row(
                "-",
                self.lang["current_format"].format(fmt=self.settings_manager.settings.output_format),
            )
            table.add_row(
                "-",
                self.lang["current_model"].format(model=self.settings_manager.settings.model),
            )
            max_tokens = self.settings_manager.settings.max_output_tokens
            if isinstance(max_tokens, int) and max_tokens > 0:
                max_tokens_label = str(max_tokens)
            else:
                max_tokens_label = self.lang["max_tokens_unlimited"]
            table.add_row(
                "-",
                self.lang["current_max_tokens"].format(value=max_tokens_label),
            )
            table.add_row(
                "-",
                self.lang["current_timeout"].format(seconds=self.settings_manager.settings.request_timeout),
            )
            supports_thinking = self.settings_manager.settings.model in THINKING_MODELS
            if self.settings_manager.settings.extended_thinking and not supports_thinking:
                thinking_state = (
                    f"{self.lang['thinking_state_off']} ({self.lang['thinking_state_blocked']})"
                )
            elif self.is_thinking_enabled():
                thinking_state = self.lang["thinking_state_on"]
            else:
                thinking_state = self.lang["thinking_state_off"]
            table.add_row(
                "-",
                self.lang["current_thinking"].format(state=thinking_state),
            )
            self.console.print(table)

            choice = self.console.input(self.lang["enter_choice"]).strip()
            if choice == "1":
                self.update_api_key()
            elif choice == "2":
                self.change_language()
            elif choice == "3":
                self.change_output_format()
            elif choice == "4":
                self.change_model()
            elif choice == "5":
                self.toggle_thinking_mode()
            elif choice == "6":
                self.change_max_tokens()
            elif choice == "7":
                self.change_timeout()
            elif choice == "8":
                break
            else:
                self.console.print(Panel.fit(self.lang["invalid_choice"], style="yellow"))

    def update_api_key(self) -> None:
        new_key = self.console.input(self.lang["new_api_key"]).strip()
        if new_key:
            self.settings_manager.update(api_key=new_key)
            self.client.update_key(new_key)
            self.console.print(Panel.fit(self.lang["api_updated"], style="green"))
        else:
            self.console.print(Panel.fit(self.lang["invalid_choice"], style="yellow"))

    def change_language(self) -> None:
        new_lang = self.console.input(self.lang["language_prompt"]).strip().lower()
        if new_lang in LANG:
            self.settings_manager.update(language=new_lang)
            self._refresh_language()
            self.console.print(Panel.fit(self.lang["language_updated"], style="green"))
        else:
            self.console.print(Panel.fit(self.lang["invalid_choice"], style="yellow"))

    def change_output_format(self) -> None:
        new_format = self.console.input(self.lang["format_prompt"]).strip().lower()
        if new_format in {"plain", "markdown"}:
            self.settings_manager.update(output_format=new_format)
            self.console.print(Panel.fit(self.lang["format_updated"], style="green"))
        else:
            self.console.print(Panel.fit(self.lang["invalid_choice"], style="yellow"))

    def change_model(self) -> None:
        models_hint = ", ".join(SUGGESTED_MODELS)
        self.console.print(Panel.fit(self.lang["available_models"].format(models=models_hint), style="cyan"))
        new_model = self.console.input(self.lang["model_prompt"]).strip()
        if new_model:
            self.settings_manager.update(model=new_model)
            self.client.update_model(new_model)
            self.console.print(Panel.fit(self.lang["model_updated"], style="green"))
            if self.settings_manager.settings.extended_thinking:
                if new_model in THINKING_MODELS:
                    self.console.print(Panel.fit(self.lang["thinking_enabled"], style="cyan"))
                else:
                    self.console.print(Panel.fit(self.lang["thinking_unavailable"], style="yellow"))
        else:
            self.console.print(Panel.fit(self.lang["invalid_choice"], style="yellow"))

    def is_thinking_enabled(self) -> bool:
        return (
            self.settings_manager.settings.extended_thinking
            and self.settings_manager.settings.model in THINKING_MODELS
        )

    def toggle_thinking_mode(self) -> None:
        model = self.settings_manager.settings.model
        if model not in THINKING_MODELS:
            self.console.print(Panel.fit(self.lang["thinking_unavailable"], style="yellow"))
            return

        new_state = not self.settings_manager.settings.extended_thinking
        self.settings_manager.update(extended_thinking=new_state)
        if new_state:
            self.console.print(Panel.fit(self.lang["thinking_enabled"], style="green"))
        else:
            self.console.print(Panel.fit(self.lang["thinking_disabled"], style="green"))

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
        user_input = self.console.input(self.lang["max_tokens_prompt"]).strip()
        if not user_input:
            self.settings_manager.update(max_output_tokens=None)
            self.client.update_max_output_tokens_retry(self._determine_retry_limit())
            self.console.print(Panel.fit(self.lang["max_tokens_cleared"], style="green"))
            return
        try:
            value = int(user_input)
            if value <= 0:
                raise ValueError
        except ValueError:
            self.console.print(Panel.fit(self.lang["invalid_choice"], style="yellow"))
            return
        self.settings_manager.update(max_output_tokens=value)
        self.client.update_max_output_tokens_retry(self._determine_retry_limit())
        self.console.print(Panel.fit(self.lang["max_tokens_updated"], style="green"))

    def change_timeout(self) -> None:
        user_input = self.console.input(self.lang["timeout_prompt"]).strip()
        try:
            value = int(user_input)
            if value <= 0:
                raise ValueError
        except (ValueError, TypeError):
            self.console.print(Panel.fit(self.lang["invalid_choice"], style="yellow"))
            return
        self.settings_manager.update(request_timeout=value)
        self.client.update_timeout(value)
        self.console.print(Panel.fit(self.lang["timeout_updated"], style="green"))

    def process_user_message(self, message: str) -> None:
        if not self.settings_manager.settings.api_key:
            self.console.print(Panel.fit(self.lang["missing_key"], style="red"))
            return

        self.console.print(Panel.fit(self.lang["sending"], style="blue"))
        thinking_enabled = self.is_thinking_enabled()
        generation_config = self._build_generation_config(thinking_enabled)
        try:
            # Der komplette Verlauf wird an Gemini übergeben, um Kontext zu erhalten
            response = self.client.send_message(
                self.history_manager.build_gemini_payload(),
                message,
                system_instruction=THINKING_SYSTEM_PROMPT if thinking_enabled else None,
                generation_config=generation_config,
            )
        except RuntimeError as exc:
            self.console.print(Panel.fit(self.lang["http_error"].format(detail=str(exc)), title=self.lang["error"], style="red"))
            return

        # Bei Erfolg werden Frage und Antwort persistiert
        self.history_manager.add_entry("user", message)
        self.history_manager.add_entry("assistant", response)
        self.history_manager.save()
        self.display_response(response)

    def display_response(self, response: str) -> None:
        if self.settings_manager.settings.output_format == "markdown":
            rendered = Markdown(response)
        else:
            rendered = response
        panel = Panel(rendered, title=f"{self.lang['model_label']} ({self.settings_manager.settings.model})", border_style="magenta", box=box.ROUNDED)
        self.console.print(panel)


def main() -> None:
    app = TerminalApp()
    try:
        app.run()
    except KeyboardInterrupt:
        Console().print("\nProgramm beendet.")
        time.sleep(0.2)
        sys.exit(0)


if __name__ == "__main__":
    main()
