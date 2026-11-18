"""Tiny helpers for coloured terminal output (respects ``NO_COLOR`` and non-terminal output)."""

from __future__ import annotations

import os
import sys

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
}


def _enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def style(text: str, *names: str) -> str:
    """Wrap ``text`` in ANSI codes, e.g. ``style("done", "green", "bold")``."""
    if not _enabled() or not names:
        return text
    return f"\033[{';'.join(_CODES[n] for n in names)}m{text}\033[0m"


def heading(text: str) -> None:
    print(style("✦ ", "cyan", "bold") + style(text, "bold"))


def row(label: str, value: str) -> None:
    print(f"  {style(f'{label:<12}', 'dim')}{value}")


def success(text: str) -> None:
    print(style("✔ ", "green") + text)


def warn(text: str) -> None:
    print(style("! ", "yellow") + text, file=sys.stderr)


def error(text: str) -> None:
    print(style("error: ", "red", "bold") + text, file=sys.stderr)


def swatch(hex_color: str) -> str:
    """Two characters of 24-bit background colour (plain text when colour is off)."""
    if not _enabled():
        return "  "
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"\033[48;2;{r};{g};{b}m  \033[0m"
