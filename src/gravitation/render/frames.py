"""Backend-agnostic preparation of everything a renderer has to draw.

The slow part of the original notebook was the scene, not the physics: it
called ``self.play`` once per stored sample (~11 000 times), each call
deep-copying 3-D spheres and writing a separate partial movie file.

Here the work is split up:

1. :func:`build_frames` (plain NumPy, milliseconds) resamples the trajectory
   onto the exact video frame times, applies the visual transforms (centring,
   scaling, satellite exaggeration) and precomputes the HUD labels.
2. A Manim scene loads the resulting :class:`FrameData` and drives everything
   from one ``ValueTracker`` animated by a single ``play`` call. That gives
   one movie file and exactly ``fps × duration`` frames.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gravitation.config import SystemConfig
from gravitation.physics import center_of_mass
from gravitation.simulation import SimulationResult
from gravitation.units import LENGTH_UNITS, pick_time_unit

FIT_RADIUS = 3.4
"""Auto-scale target: the system's extent in scene units. Manim's frame is 8 units tall."""

HUD_UPDATES_PER_SECOND = 8
"""How often the on-screen clock changes; re-typesetting text every frame would be wasteful."""

TRAIL_OVERSAMPLE = 4
"""Trail points per video frame. Fast bodies (the Moon moves ~6 days per frame in a
two-year, 20-second video) would draw jagged trails if sampled only once per frame."""


@dataclass(eq=False)
class FrameData:
    """What a scene needs to draw, frame by frame, in scene units.

    Attributes:
        track: body centres at ``oversample`` points per frame, shape
            ``((F - 1)·oversample + 1, N, 3)``; frame ``k`` is ``track[k·oversample]``.
        oversample: track points per frame (for smooth trails).
        names, colors, radii: per-body appearance.
        glow: halo colour per body, ``""`` for none.
        fps: video frame rate.
        trail_frames: trail length in frames (``0`` = none, ``F`` = whole history).
        trail_width, trail_opacity: trail stroke style.
        labels: HUD clock text per frame.
        title: system name shown in the HUD.
        background: background colour ``#rrggbb``.
        camera_phi, camera_theta, camera_zoom, camera_rotation: camera setup
            (degrees; rotation in degrees per second).
        hud, axes: optional overlays.
        axis_extent: half-length of the axes in scene units.
        scale_label: e.g. ``"1 au = 2.13 units"``, for the curious.
    """

    track: np.ndarray
    oversample: int
    names: list[str]
    colors: list[str]
    radii: np.ndarray
    glow: list[str]
    fps: int
    trail_frames: int
    trail_width: float
    trail_opacity: float
    labels: list[str]
    title: str
    background: str
    camera_phi: float
    camera_theta: float
    camera_zoom: float
    camera_rotation: float
    hud: bool
    axes: bool
    axis_extent: float
    scale_label: str

    @property
    def positions(self) -> np.ndarray:
        """Body centres per frame, shape ``(F, N, 3)``."""
        return self.track[:: self.oversample]

    @property
    def n_frames(self) -> int:
        return (self.track.shape[0] - 1) // self.oversample + 1

    @property
    def duration(self) -> float:
        """Video length in seconds."""
        return (self.n_frames - 1) / self.fps

    def positions_at(self, k: float) -> np.ndarray:
        """Body centres at a fractional frame index, linearly interpolated, shape ``(N, 3)``."""
        last = self.track.shape[0] - 1
        j = min(max(k * self.oversample, 0.0), float(last))
        i = min(int(j), last - 1)
        f = j - i
        if f <= 0.0:
            return self.track[i]
        return (1.0 - f) * self.track[i] + f * self.track[i + 1]

    def trail_points(self, body: int, k: float, current: np.ndarray) -> np.ndarray:
        """Polyline of the trail behind ``body`` at frame ``k``, ending at ``current`` (≥ 2 points)."""
        end = min(int(k * self.oversample), self.track.shape[0] - 1)
        start = max(0, end - self.trail_frames * self.oversample)
        return np.vstack([self.track[start : end + 1, body], current[np.newaxis]])

    def label_at(self, k: float) -> str:
        return self.labels[min(round(k), self.n_frames - 1)]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = asdict(self)
        arrays = {key: fields.pop(key) for key in ("track", "radii")}
        np.savez_compressed(path, **arrays, meta=json.dumps(fields))
        return path

    @classmethod
    def load(cls, path: str | Path) -> FrameData:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls(
                track=data["track"].astype(float),
                radii=data["radii"],
                **meta,
            )


def build_frames(
    result: SimulationResult,
    config: SystemConfig,
    fps: int,
    *,
    duration: float | None = None,
) -> FrameData:
    """Resample a simulation onto video frames and apply the render settings.

    Args:
        result: a finished simulation.
        config: the config it came from (for appearance and render settings).
        fps: video frame rate.
        duration: override ``render.duration`` (seconds of video).
    """
    rs = config.render
    duration = duration or rs.duration
    n_frames = max(2, round(duration * fps) + 1)

    # 1. Resample snapshots onto the track times (linear interpolation in time).
    oversample = TRAIL_OVERSAMPLE
    t_track = np.linspace(result.t[0], result.t[-1], (n_frames - 1) * oversample + 1)
    t_frames = t_track[::oversample]
    idx = np.clip(np.searchsorted(result.t, t_track, side="right") - 1, 0, len(result.t) - 2)
    span = result.t[idx + 1] - result.t[idx]
    w = ((t_track - result.t[idx]) / span)[:, np.newaxis, np.newaxis]
    true_pos = (1.0 - w) * result.x[idx] + w * result.x[idx + 1]

    # 2. Visual exaggeration of satellites around their parents.
    pos = true_pos.copy()
    for i, body in enumerate(config.bodies):
        if body.exaggerate is not None:
            j = config.index(body.exaggerate.around)
            pos[:, i] = true_pos[:, j] + body.exaggerate.factor * (true_pos[:, i] - true_pos[:, j])

    # 3. Centre of the view.
    if rs.center == "barycenter":
        centre = center_of_mass(true_pos, result.mu)
    elif rs.center == "origin":
        centre = np.zeros((len(t_track), 3))
    else:
        centre = true_pos[:, config.index(rs.center)]
    pos -= centre[:, np.newaxis, :]

    # 4. Physical units → scene units.
    units = config.units
    if rs.scale is None:
        distances = np.linalg.norm(pos, axis=-1)
        extent = float(np.percentile(distances, 99)) or 1.0
        scale = FIT_RADIUS / extent
    else:
        scale = rs.scale / (LENGTH_UNITS[rs.scale_unit] if rs.scale_unit else units.length_si)
    pos *= scale
    unit_name = "unit" if units.natural else (rs.scale_unit or units.length)
    unit_size = (
        units.length_si if units.natural or not rs.scale_unit else LENGTH_UNITS[rs.scale_unit]
    )
    scale_label = f"1 {unit_name} = {scale * unit_size:.3g} scene units"

    # 5. HUD clock, refreshed a few times per second.
    every = max(1, round(fps / HUD_UPDATES_PER_SECOND))
    if units.natural:
        labels = [f"t = {t_frames[(k // every) * every]:.2f}" for k in range(n_frames)]
    else:
        unit, size = pick_time_unit(result.t[-1])
        word = {"s": "s", "min": "min", "h": "h", "day": "days", "yr": "years"}[unit]
        digits = 0 if result.t[-1] / size >= 100 else 1 if result.t[-1] / size >= 10 else 2
        labels = [
            f"t = {t_frames[(k // every) * every] / size:.{digits}f} {word}"
            for k in range(n_frames)
        ]

    trail_frames = n_frames if rs.trail is None else round(rs.trail * fps)

    return FrameData(
        track=pos.astype(np.float32),
        oversample=oversample,
        names=config.names,
        colors=[b.color for b in config.bodies],
        radii=np.array([b.radius for b in config.bodies]),
        glow=[b.glow or "" for b in config.bodies],
        fps=fps,
        trail_frames=trail_frames,
        trail_width=rs.trail_width,
        trail_opacity=rs.trail_opacity,
        labels=labels,
        title=config.name,
        background=rs.background,
        camera_phi=rs.camera.phi,
        camera_theta=rs.camera.theta,
        camera_zoom=rs.camera.zoom,
        camera_rotation=rs.camera.rotation,
        hud=rs.hud,
        axes=rs.axes,
        axis_extent=FIT_RADIUS * 1.15,
        scale_label=scale_label,
    )
