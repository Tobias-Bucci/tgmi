"""Shared constants for the Gemini terminal client."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

CONFIG_PATH = Path("config.json")
DEFAULT_HISTORY_PATH = Path("history.json")
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_API_KEY = "Test"
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

LANG: Dict[str, Dict[str, str]] = {
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
        "thinking_in_progress": "Denkmode aktiv – Gemini denkt…",
        "copy_success": "Antwort wurde in die Zwischenablage kopiert.",
        "copy_failure": "Antwort konnte nicht kopiert werden.",
        "copy_no_response": "Es liegt keine Antwort zum Kopieren vor.",
        "export_prompt": "Dateiname für Markdown-Export angeben (Enter für {path}): ",
        "history_exported": "Markdown-Export gespeichert unter {path}.",
        "history_export_failed": "Markdown-Export fehlgeschlagen: {error}",
        "history_empty_export": "Keine Einträge vorhanden – nichts zu exportieren.",
        "history_empty": "Keine Historie vorhanden.",
        "role_user": "Du",
        "role_assistant": "Gemini",
        "role_system": "System",
        "search_prompt": "Suchbegriff eingeben (Enter zum Abbrechen): ",
        "search_no_results": "Keine Treffer gefunden.",
        "search_results_title": "Suchergebnisse",
        "search_result_column_index": "#",
        "search_result_column_role": "Rolle",
        "search_result_column_time": "Zeit",
        "search_result_column_snippet": "Ausschnitt",
        "insights_title": "Gesprächsstatistik",
        "insights_no_data": "Noch keine Nachrichten vorhanden.",
        "insights_total": "Nachrichten insgesamt",
        "insights_user": "Du",
        "insights_assistant": "Gemini",
        "insights_span": "Zeitraum",
        "insights_avg_user_length": "Ø Länge deiner Nachrichten",
        "insights_avg_response_time": "Ø Antwortzeit von Gemini",
        "unit_seconds": "Sek.",
        "unit_characters": "Zeichen",
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
            ":o - Optionen\n:c - Historie leeren\n:x - Historie als Markdown exportieren\n:f - Verlauf durchsuchen\n:i - Statistik anzeigen\n:h - Hilfe\n:cp - Letzte Antwort kopieren"
        ),
        "file_command_hint": "Zeilen mit # <Pfad> oder ! <Pfad> fügen Dateien zur nächsten Nachricht hinzu.",
        "file_attached": "Datei angehängt: {path}",
        "file_not_found": "Datei nicht gefunden: {path}",
        "file_not_file": "Pfad ist keine Datei: {path}",
        "file_read_error": "Datei konnte nicht gelesen werden: {path} ({error})",
        "message_empty": "Es ist keine Nachricht vorhanden, die gesendet werden könnte.",
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
        "thinking_in_progress": "Thinking mode active – Gemini is reasoning…",
        "copy_success": "Response copied to clipboard.",
        "copy_failure": "Could not copy the response to the clipboard.",
        "copy_no_response": "There is no response available to copy.",
        "export_prompt": "Enter filename for Markdown export (press Enter for {path}): ",
        "history_exported": "History exported to {path}.",
        "history_export_failed": "Markdown export failed: {error}",
        "history_empty_export": "No entries to export yet.",
        "history_empty": "History is empty so far.",
        "role_user": "You",
        "role_assistant": "Gemini",
        "role_system": "System",
        "search_prompt": "Enter search term (press Enter to cancel): ",
        "search_no_results": "No matches found.",
        "search_results_title": "Search results",
        "search_result_column_index": "#",
        "search_result_column_role": "Role",
        "search_result_column_time": "Time",
        "search_result_column_snippet": "Snippet",
        "insights_title": "Conversation insights",
        "insights_no_data": "No messages yet.",
        "insights_total": "Total messages",
        "insights_user": "You",
        "insights_assistant": "Gemini",
        "insights_span": "Time span",
        "insights_avg_user_length": "Avg. length of your messages",
        "insights_avg_response_time": "Avg. Gemini response time",
        "unit_seconds": "s",
        "unit_characters": "chars",
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
            ":o - Settings\n:c - Clear history\n:x - Export history as Markdown\n:f - Search history\n:i - Conversation insights\n:h - Help\n:cp - Copy last response"
        ),
        "file_command_hint": "Lines starting with # <path> or ! <path> attach files to the next request.",
        "file_attached": "Attached file: {path}",
        "file_not_found": "File not found: {path}",
        "file_not_file": "Path is not a file: {path}",
        "file_read_error": "Could not read file: {path} ({error})",
        "message_empty": "There is no message to send after processing file commands.",
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
