"""Colours: CSS/HTML names, hex codes and RGB triples, all normalised to ``#rrggbb``.

Accepted spellings in a config::

    color: orange              # any of the 148 CSS/HTML colour names
    color: Dark Orange         # case, spaces, '_' and '-' are ignored
    color: "#ff8800"           # hex; quote it, because '#' starts a YAML comment
    color: "#f80"              # short hex
    color: ff8800              # bare 6-digit hex also works
    color: [255, 136, 0]       # RGB, 0-255
    color: [1.0, 0.53, 0.0]    # RGB, 0-1
    color: tab:blue            # matplotlib's tab: and xkcd: palettes
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Sequence

from matplotlib.colors import CSS4_COLORS, to_hex

DEFAULT_PALETTE: tuple[str, ...] = (
    "#4cc9f0",
    "#f72585",
    "#ffd166",
    "#06d6a0",
    "#ff9f1c",
    "#b388ff",
    "#ef476f",
    "#90e0ef",
)
"""Colours handed out to bodies that do not specify one; chosen to read well on a dark background."""

_HEX = re.compile(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})|([0-9a-fA-F]{6})")


def _key(name: str) -> str:
    return re.sub(r"[\s_\-]", "", name.lower())


NAMED_COLORS: dict[str, str] = {name: value.lower() for name, value in CSS4_COLORS.items()}
"""The CSS/HTML named colours, name → ``#rrggbb``."""

_BY_KEY: dict[str, str] = {_key(name): value for name, value in NAMED_COLORS.items()}


def resolve_color(value: object) -> str:
    """Convert a colour written in a config to ``#rrggbb``.

    Raises:
        ValueError: if the colour is not recognised. The message suggests the
            closest CSS name, and reminds that hex codes must be quoted in YAML.
    """
    if value is None:
        raise ValueError(
            'colour is empty — hex codes must be quoted in YAML, e.g. color: "#ff8800"'
        )
    if isinstance(value, Sequence) and not isinstance(value, str):
        return _rgb_to_hex(value)

    text = str(value).strip()
    match = _HEX.fullmatch(text)
    if match:
        digits = match.group(1) or match.group(2)
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        return f"#{digits.lower()}"

    if _key(text) in _BY_KEY:
        return _BY_KEY[_key(text)]

    if ":" in text:  # matplotlib palettes such as tab:blue or xkcd:sky blue
        try:
            return to_hex(text)
        except ValueError:
            pass

    hints = difflib.get_close_matches(_key(text), list(_BY_KEY), n=3)
    names = {_key(n): n for n in NAMED_COLORS}
    tail = f" — did you mean {', '.join(repr(names[h]) for h in hints)}?" if hints else ""
    raise ValueError(
        f"unknown colour {value!r}{tail} (use a CSS name or a hex code like '#ff8800')"
    )


def _rgb_to_hex(components: Sequence[object]) -> str:
    if len(components) != 3:
        raise ValueError(f"RGB colour needs 3 components, got {list(components)!r}")
    try:
        rgb = [float(c) for c in components]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"RGB components must be numbers, got {list(components)!r}") from None
    if any(c > 1 for c in rgb):
        rgb = [c / 255 for c in rgb]
    if not all(0 <= c <= 1 for c in rgb):
        raise ValueError(f"RGB components must be within 0-255 (or 0-1), got {list(components)!r}")
    return to_hex(rgb)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """``"#rrggbb"`` → ``(r, g, b)`` with components in 0-255."""
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
