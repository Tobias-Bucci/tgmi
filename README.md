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

| Shortcut | Funktion                           |
| -------- | ---------------------------------- |
| `:q`     | Programm beenden                   |
| `:h`     | Hilfe anzeigen                     |
| `:s`     | Chat-Historie speichern            |
| `:l`     | Chat-Historie laden                |
| `:o`     | Einstellungen öffnen               |
| `:c`     | Chat-Historie leeren               |
| `:x`     | Historie als Markdown exportieren  |
| `:f`     | Verlauf nach Stichwort durchsuchen |
| `:i`     | Gesprächsstatistik anzeigen        |
| `:cp`    | Letzte Antwort kopieren            |

### Einstellungen

Im Optionsmenü können folgende Parameter angepasst werden:

- API-Schlüssel
- Menü-Sprache (Deutsch/Englisch)
- Ausgabeformat (Klartext oder Markdown-Rendering)
- Verwendetes Modell (z. B. `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash` …)
- Denkmode (für Modelle mit erweiterten Reasoning-Fähigkeiten)
- Maximale Antwort-Tokens (leer lassen für unbegrenzt oder bei Bedarf erhöhen)
- HTTP-Timeout für API-Aufrufe (praktisch bei sehr langen Antworten)
- Shortcut `:cp`, der die letzte KI-Antwort erneut in die Zwischenablage kopiert

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

Der Denkmode lässt sich für die Modelle `gemini-2.5-*` sowie `gemini-2.0-*` aktivieren; das Programm ergänzt dann intern eine ausführliche Systemanweisung und passt die Sampling-Parameter für bessere Reasoning-Antworten an. Während das Modell denkt, zeigt der Client einen temporären Hinweis an, der nach Abschluss automatisch verschwindet. Nach jeder Antwort kannst du bei Bedarf mit `:cp` den angezeigten Text (Markdown oder Plaintext) in die Zwischenablage kopieren.

> Tipp: Für beste Kompatibilität bei der Zwischenablage kannst du optional `pyperclip` installieren (`pip install pyperclip`). Andernfalls versucht die Anwendung, plattformspezifische Tools wie `clip`, `pbcopy`, `wl-copy` oder `xclip` zu verwenden.

### Historie

Die Chat-Historie wird als JSON-Datei gespeichert (Standard: `history.json`). Änderungen werden automatisch nach jeder gesendeten Nachricht sowie über das Shortcut `:s` gespeichert. Über `:x` lässt sich der Gesprächsverlauf außerdem als Markdown-Datei exportieren (Standard: `history.md`). Mit `:f` kannst du in der Historie nach Stichwörtern suchen und mit `:i` erhältst du eine kompakte Statistik zum Verlauf (Anzahl Nachrichten, durchschnittliche Antwortzeit u. v. m.).

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
