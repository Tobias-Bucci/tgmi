"""Utility helpers for interacting with the system clipboard."""
from __future__ import annotations

import shutil
import subprocess
import sys
from typing import List


class ClipboardHelper:
    """Cross-platform clipboard abstraction."""

    @staticmethod
    def copy_text(text: str) -> bool:
        if not text:
            return False

        try:
            import pyperclip  # type: ignore
        except ImportError:
            pyperclip = None  # type: ignore

        if pyperclip is not None:
            try:
                pyperclip.copy(text)
                return True
            except Exception:
                pass

        platform = sys.platform
        commands: List[List[str]] = []

        if sys.platform.startswith("win"):
            commands.append(["clip"])
        elif platform == "darwin":
            commands.append(["pbcopy"])
        else:
            commands.extend(
                [
                    ["wl-copy"],
                    ["xclip", "-selection", "clipboard"],
                    ["xsel", "--clipboard", "--input"],
                ]
            )

        for command in commands:
            if shutil.which(command[0]) is None:
                continue
            try:
                subprocess.run(command, input=text, text=True, check=True)
                return True
            except Exception:
                continue

        return False
