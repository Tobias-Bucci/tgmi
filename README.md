# Terminal Gemini Chat Client

Dieses Projekt bietet einen terminalbasierten Chat-Client für Googles Gemini API. Die Anwendung läuft auf Windows und Linux und setzt auf eine leichtgewichtige Bedienung mit Rich-TUI, konfigurierbaren Einstellungen und Verwaltung der Chat-Historie.

## Voraussetzungen

- Python 3.9 oder neuer
- Abhängigkeiten gemäß `requirements.txt`
- Ein gültiger Google Gemini API-Key

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Unter Linux lautet der Aktivierungsbefehl:

```bash
source .venv/bin/activate
```

## Nutzung

```powershell
python main.py
```

### Tastenkombinationen

| Shortcut | Funktion                |
| -------- | ----------------------- |
| `:q`     | Programm beenden        |
| `:h`     | Hilfe anzeigen          |
| `:s`     | Chat-Historie speichern |
| `:l`     | Chat-Historie laden     |
| `:o`     | Einstellungen öffnen    |
| `:c`     | Chat-Historie leeren    |

### Einstellungen

Im Optionsmenü können folgende Parameter angepasst werden:

- API-Schlüssel
- Menü-Sprache (Deutsch/Englisch)
- Ausgabeformat (Klartext oder Markdown-Rendering)
- Verwendetes Modell (z. B. `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash` …)
- Denkmode (für Modelle mit erweiterten Reasoning-Fähigkeiten)

> **Hinweis:** Falls der Standardwert `gemini-2.5-flash` in deiner Region nicht verfügbar ist, kannst du über `:o` → „Modell ändern“ direkt auf ein anderes Modell wechseln.

Unterstützte Modellkandidaten:

- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-2.0-flash`
- `gemini-2.0-flash-lite`
- `gemini-live-2.5-flash`
- `gemini-live-2.5-flash-preview-native-audio`
- (Legacy) `gemini-1.5-flash`, `gemini-1.5-pro`

Der Denkmode lässt sich für die Modelle `gemini-2.5-*` sowie `gemini-2.0-*` aktivieren; das Programm ergänzt dann intern eine ausführliche Systemanweisung und hebt das Token-Limit für längere Antworten an.

### Historie

Die Chat-Historie wird als JSON-Datei gespeichert (Standard: `history.json`). Änderungen werden automatisch nach jeder gesendeten Nachricht sowie über das Shortcut `:s` gespeichert.

## Paketierung

Für eine eigenständige Binary bietet sich beispielsweise [PyInstaller](https://pyinstaller.org) an:

```powershell
pyinstaller --onefile main.py
```

## Fehlerbehandlung

- API-Fehler werden mit klaren Hinweisen im Terminal angezeigt.
- Ungültige Eingaben in Menüs werden abgefangen und führen zu einer Wiederholung der Abfrage.

## Lizenz

Dieses Projekt wird ohne Garantie bereitgestellt. Verwenden Sie es auf eigenes Risiko.
