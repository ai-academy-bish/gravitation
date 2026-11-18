"""Choosing a HUD font that exists on this machine (both Manims render text through Pango)."""

from __future__ import annotations

from functools import cache

PREFERRED_FONTS = (
    "Inter",
    "Helvetica Neue",
    "Helvetica",
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
    "Noto Sans",
    "Ubuntu",
    "Cantarell",
)
"""Clean sans-serif fonts, tried in order. Without one, Pango falls back to a serif font."""


@cache
def pick_font() -> str:
    """The first installed font from :data:`PREFERRED_FONTS`, or ``""`` for Pango's default."""
    try:
        import manimpango

        available = set(manimpango.list_fonts())
    except Exception:
        return ""
    return next((font for font in PREFERRED_FONTS if font in available), "")
