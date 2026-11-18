"""Quick static previews with matplotlib: true orbits and the energy error.

Unlike the videos, the preview applies no visual tricks (no satellite
exaggeration and no size scaling), so it shows exactly what the integrator
computed. The 3-D axes have equal aspect ratio; the original notebook did
not set this, so its circular orbits looked like ellipses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gravitation.config import SystemConfig
from gravitation.physics import center_of_mass
from gravitation.simulation import SimulationResult
from gravitation.units import pick_time_unit

_TEXT = "#e6e6e6"
_GRID = "#3a3f55"


def _style_axes(ax, background: str) -> None:
    ax.set_facecolor(background)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in getattr(ax, "spines", {}).values():
        spine.set_color(_GRID)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)


def _visible(color: str, glow: str | None) -> str:
    """A line colour that shows on the dark background: very dark bodies use their glow or grey."""
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    if 0.2126 * r + 0.7152 * g + 0.0722 * b > 40:
        return color
    return glow or "#9aa0b4"


def _time_axis(result: SimulationResult, config: SystemConfig) -> tuple[np.ndarray, str]:
    if config.units.natural:
        return result.t, "time (N-body units)"
    unit, size = pick_time_unit(result.t[-1])
    return result.t / size, f"time [{unit}]"


def plot_preview(
    result: SimulationResult, config: SystemConfig, path: Path | None = None, *, show: bool = False
) -> Path | None:
    """Draw orbits (left) and the relative energy error (right).

    Args:
        result: a finished simulation.
        config: its config (for colours, units and the view centre).
        path: where to save a PNG; ``None`` with ``show=True`` only opens a window.
        show: open an interactive window.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bg = config.render.background
    units = config.units
    length_name = "N-body units" if units.natural else units.length

    pos = result.x.copy()
    center = config.render.center
    if center == "barycenter":
        pos -= center_of_mass(pos, result.mu)[:, np.newaxis, :]
    elif center != "origin":
        pos -= pos[:, config.index(center), np.newaxis, :]
    pos /= units.length_si

    fig = plt.figure(figsize=(14, 6.5), facecolor=bg)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_facecolor(bg)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0.02))
        pane.label.set_color(_TEXT)
        pane._axinfo["grid"]["color"] = _GRID
    ax.tick_params(colors=_TEXT, labelsize=7)

    for i, body in enumerate(config.bodies):
        track = pos[:, i]
        color = _visible(body.color, body.glow)
        ax.plot(*track.T, color=color, lw=1.2, alpha=0.9, label=body.name)
        ax.scatter(*track[0], color=color, s=18, facecolors="none", linewidths=1)
        ax.scatter(*track[-1], color=color, s=30)

    lo, hi = pos.reshape(-1, 3).min(axis=0), pos.reshape(-1, 3).max(axis=0)
    span = np.maximum(hi - lo, 1e-9 * max(np.abs(hi - lo).max(), 1e-300))
    ax.set_box_aspect(span / span.max())
    ax.set_xlabel(f"x [{length_name}]")
    ax.set_ylabel(f"y [{length_name}]")
    ax.set_zlabel(f"z [{length_name}]")
    ax.set_title(f"{config.name} — trajectories (○ start, ● end)", color=_TEXT, fontsize=11)
    legend = ax.legend(loc="upper left", fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(_TEXT)

    ax2 = fig.add_subplot(1, 2, 2)
    _style_axes(ax2, bg)
    t, t_label = _time_axis(result, config)
    error = np.maximum(result.energy_error, 1e-17)
    ax2.semilogy(t, error, color="#4cc9f0", lw=1.2)
    ax2.set_xlabel(t_label)
    ax2.set_ylabel("relative energy error |ΔE / E₀|")
    ax2.grid(True, color=_GRID, lw=0.5, alpha=0.6)
    ax2.set_title(
        f"{result.integrator}, {result.steps:,} steps, {result.wall_time:.2f} s",
        color=_TEXT,
        fontsize=11,
    )

    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, facecolor=bg)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_integrator_comparison(
    runs: dict[str, SimulationResult], config: SystemConfig, path: Path
) -> Path:
    """Energy error versus time for several integrators on one log-scale chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gravitation.integrators import INTEGRATORS

    palette = ["#ef476f", "#ffd166", "#06d6a0", "#4cc9f0", "#b388ff", "#ff9f1c"]
    bg = config.render.background
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=bg)
    _style_axes(ax, bg)
    for color, (name, result) in zip(palette, runs.items(), strict=False):
        t, t_label = _time_axis(result, config)
        meta = INTEGRATORS[name]
        label = f"{meta.title} (order {meta.order}{', symplectic' if meta.symplectic else ''})"
        ax.semilogy(t, np.maximum(result.energy_error, 1e-17), color=color, lw=1.3, label=label)
    ax.set_xlabel(t_label)
    ax.set_ylabel("relative energy error |ΔE / E₀|")
    ax.grid(True, color=_GRID, lw=0.5, alpha=0.6)
    some = next(iter(runs.values()))
    ax.set_title(
        f"{config.name}: same step size for every method ({some.steps:,} steps)",
        color=_TEXT,
        fontsize=11,
    )
    legend = ax.legend(
        fontsize=9, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13)
    )
    for text in legend.get_texts():
        text.set_color(_TEXT)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=bg)
    plt.close(fig)
    return path
