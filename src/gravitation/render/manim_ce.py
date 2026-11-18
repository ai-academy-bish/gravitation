"""Manim Community Edition scene for a prepared :class:`~gravitation.render.frames.FrameData`.

Run through ``gravitation render --backend ce`` (Cairo, CPU) or
``--backend ce-gl`` (CE's OpenGL renderer). The CLI passes the frame file
in the ``GRAVITATION_FRAMES`` environment variable.

Why it is fast compared with the original notebook:

* one ``ValueTracker`` goes from frame 0 to the last frame in a single
  ``play`` call, so Manim writes one movie and renders exactly
  ``fps × duration`` frames;
* a single updater moves every body and redraws every trail from
  precomputed positions, with no ``.animate`` copies and no per-frame
  animation objects;
* with Cairo, bodies are shaded discs that always face the camera instead
  of true spheres. Profiling showed that per-face shading and projection of
  sphere surfaces took 90% of the frame time: a 288-face sphere is 288
  separate vector objects for Cairo. At 10–40 pixels across, a shaded disc
  looks the same. The OpenGL renderer draws real spheres, since the GPU
  makes them cheap (as one mesh: see ``_gpu_sphere``).
"""

from __future__ import annotations

import os
from itertools import pairwise

import numpy as np
from manim import (
    DEGREES,
    DL,
    DOWN,
    LEFT,
    PI,
    TAU,
    UL,
    WHITE,
    Circle,
    Dot,
    Group,
    Mobject,
    Text,
    ThreeDAxes,
    ThreeDScene,
    ValueTracker,
    VGroup,
    VMobject,
    config,
    linear,
)
from manim.constants import RendererType
from manim.mobject.opengl.opengl_surface import OpenGLSurface
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject

from gravitation.render.fonts import pick_font
from gravitation.render.frames import FrameData

TRAIL_SEGMENTS = 8
"""A trail is split into this many pieces of rising opacity, so that it fades out behind the body."""

SPHERE_RESOLUTION = (33, 17)  # sample points in u (longitude) and v (latitude)
GLOW_LAYERS = 5
TRAIL_Z = -1000  # trails are always drawn below bodies


def _mix(color: str, other: str, t: float) -> str:
    """Blend two ``#rrggbb`` colours: ``t = 0`` gives ``other``, ``t = 1`` gives ``color``."""
    a = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)])
    b = np.array([int(other[i : i + 2], 16) for i in (1, 3, 5)])
    c = np.rint(b + t * (a - b)).astype(int)
    return "#" + "".join(f"{v:02x}" for v in c)


def _shaded_disc(radius: float, color: str) -> VGroup:
    """A sphere look-alike for Cairo: three discs, lit from the upper left."""
    r = radius
    base = Circle(radius=r).set_fill(_mix(color, "#000000", 0.55), 1).set_stroke(width=0)
    body = Circle(radius=0.86 * r).set_fill(color, 1).set_stroke(width=0)
    body.shift(np.array([-0.1 * r, 0.1 * r, 0]))
    highlight = (
        Circle(radius=0.32 * r).set_fill(_mix("#ffffff", color, 0.55), 0.85).set_stroke(width=0)
    )
    highlight.shift(np.array([-0.38 * r, 0.38 * r, 0]))
    return VGroup(base, body, highlight)


def _gpu_sphere(
    radius: float, color: str, *, opacity: float = 1.0, depth_test: bool = True
) -> OpenGLSurface:
    """A sphere drawn as one triangle mesh by the GPU.

    CE's ``Sphere`` is a group of separate polygon faces even under the OpenGL
    renderer, and every face is re-triangulated in Python after each move:
    about 3 000 triangulations per frame for six bodies, which made this
    backend ~20x slower. ``OpenGLSurface`` is a single mesh.
    """

    def uv(u: float, v: float) -> np.ndarray:
        return radius * np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), -np.cos(v)])

    return OpenGLSurface(
        uv_func=uv,
        u_range=(0, TAU),
        v_range=(0, PI),
        resolution=SPHERE_RESOLUTION,
        color=color,
        opacity=opacity,
        depth_test=depth_test,
    )


class GravityScene(ThreeDScene):
    """Bodies, fading trails, optional axes and a HUD clock."""

    def construct(self) -> None:
        data = FrameData.load(os.environ["GRAVITATION_FRAMES"])
        opengl = config.renderer == RendererType.OPENGL
        font = pick_font()

        if opengl:
            self.renderer.background_color = data.background
        else:
            self.camera.background_color = data.background
        # Our angles follow ManimGL. CE's OpenGL camera agrees; its Cairo camera
        # measures θ from a different axis (θ = -90° there is ManimGL's θ = 0).
        theta = data.camera_theta if opengl else data.camera_theta - 90
        self.set_camera_orientation(
            phi=data.camera_phi * DEGREES, theta=theta * DEGREES, zoom=data.camera_zoom
        )

        if data.axes:
            extent = data.axis_extent
            axes = ThreeDAxes(
                x_range=(-extent, extent, 1),
                y_range=(-extent, extent, 1),
                z_range=(-extent, extent, 1),
                x_length=2 * extent,
                y_length=2 * extent,
                z_length=2 * extent,
                axis_config={"stroke_width": 1, "stroke_opacity": 0.35, "include_tip": False},
            )
            if not opengl:
                axes.set_z_index(TRAIL_Z - 1)
            self.add(axes)

        start = data.positions[0]
        if opengl:
            bodies = [
                _gpu_sphere(float(r), c).move_to(p)
                for r, c, p in zip(data.radii, data.colors, start, strict=True)
            ]
        else:
            bodies = [
                _shaded_disc(float(r), c).move_to(p)
                for r, c, p in zip(data.radii, data.colors, start, strict=True)
            ]
            self.add_fixed_orientation_mobjects(*bodies)  # billboards: always face the camera

        halos: list[Mobject | None] = []
        for r, p, glow in zip(data.radii, start, data.glow, strict=True):
            if not glow:
                halos.append(None)
                continue
            if opengl:
                # Nested translucent spheres look the same from every side, so the
                # halo needs no billboarding; flat discs showed dark rings here.
                halo = Group(
                    *[
                        _gpu_sphere(
                            float(r) * (1.4 + 0.7 * k), glow, opacity=0.07, depth_test=False
                        )
                        for k in range(GLOW_LAYERS)
                    ]
                ).move_to(p)
                self.add(halo)
            else:
                halo = VGroup(
                    *[
                        Dot(
                            radius=float(r) * (1.4 + 0.7 * k),
                            color=glow,
                            fill_opacity=0.10,
                            stroke_width=0,
                        )
                        for k in range(GLOW_LAYERS)
                    ]
                ).move_to(p)
                self.add_fixed_orientation_mobjects(halo)
            halos.append(halo)

        trails: list[list[VMobject]] = []
        line_class = OpenGLVMobject if opengl else VMobject  # plain VMobject is Cairo-only
        if data.trail_frames > 0:
            for c in data.colors:
                pieces = []
                for k in range(TRAIL_SEGMENTS):
                    t = (k + 1) / TRAIL_SEGMENTS
                    piece = line_class().set_points_as_corners([start[0], start[0]])
                    piece.set_stroke(
                        _mix(c, data.background, t),
                        width=data.trail_width,
                        opacity=data.trail_opacity,
                    )
                    if not opengl:  # OpenGL uses the depth buffer instead of z-indices
                        piece.set_z_index(TRAIL_Z)
                    pieces.append(piece)
                trails.append(pieces)
                self.add(*pieces)

        if opengl:
            self.add(*bodies)

        clock = title = None
        if data.hud:
            title = Text(data.title, font=font, font_size=28, color=WHITE).to_corner(UL)
            clock = self._clock(data.labels[0], font, title)
            hint = Text(data.scale_label, font=font, font_size=16, color=WHITE).set_opacity(0.45)
            self.add_fixed_in_frame_mobjects(title, clock, hint.to_corner(DL))

        frame = ValueTracker(0.0)
        shown_label = [data.labels[0]]

        def update(_: VGroup) -> None:
            k = frame.get_value()
            positions = data.positions_at(k)
            for i, body in enumerate(bodies):
                body.move_to(positions[i])
                if halos[i] is not None:
                    halos[i].move_to(positions[i])
            if not opengl:
                # Painter's algorithm: draw the farthest body first.
                depth = positions @ self.camera.get_rotation_matrix().T[:, 2]
                for rank, i in enumerate(np.argsort(depth)):
                    bodies[i].set_z_index(2 * rank + 1)
                    if halos[i] is not None:
                        halos[i].set_z_index(2 * rank)
            for i, pieces in enumerate(trails):
                points = data.trail_points(i, k, positions[i])
                for piece, chunk in zip(pieces, _split(points, TRAIL_SEGMENTS), strict=True):
                    piece.set_points_as_corners(chunk)
            if clock is not None and data.label_at(k) != shown_label[0]:
                shown_label[0] = data.label_at(k)
                clock.become(self._clock(shown_label[0], font, title))
                if opengl:
                    clock.fix_in_frame()
                else:
                    self.camera.add_fixed_in_frame_mobjects(clock)

        driver = VGroup()
        driver.add_updater(update)
        self.add(driver)

        if data.camera_rotation:
            self.begin_ambient_camera_rotation(rate=data.camera_rotation * DEGREES)
        self.play(
            frame.animate.set_value(data.n_frames - 1),
            run_time=data.duration,
            rate_func=linear,
        )

    @staticmethod
    def _clock(label: str, font: str, title: Text) -> Text:
        clock = Text(label, font=font, font_size=22, color=WHITE).set_opacity(0.8)
        return clock.next_to(title, DOWN, aligned_edge=LEFT)


def _split(points: np.ndarray, parts: int) -> list[np.ndarray]:
    """Split a polyline into ``parts`` consecutive pieces that share their end points."""
    n = len(points)
    edges = np.linspace(0, n - 1, parts + 1).round().astype(int)
    chunks = []
    for a, b in pairwise(edges):
        chunk = points[a : b + 1]
        if len(chunk) < 2:
            chunk = np.vstack([points[b], points[b]])
        chunks.append(chunk)
    return chunks
