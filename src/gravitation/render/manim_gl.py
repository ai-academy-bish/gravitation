"""3Blue1Brown's ManimGL scene for a prepared :class:`~gravitation.render.frames.FrameData`.

Run through ``gravitation render --backend gl``. The CLI passes the frame
file in the ``GRAVITATION_FRAMES`` environment variable.

ManimGL renders on the GPU (since August 2026 through wgpu: Metal on macOS, Vulkan
on Linux). With ``-w`` it draws off-screen and needs no window or X server.
The scene is structured like the CE one (one ``ValueTracker``, one updater,
one ``play``) so that the benchmark compares renderers, not scene code. It
also uses two GPU niceties: ``GlowDot`` halos and per-point trail opacity
for a smooth fade.
"""

from __future__ import annotations

import os

import numpy as np
from manimlib import (
    DEGREES,
    DL,
    DOWN,
    LEFT,
    ORIGIN,
    UL,
    WHITE,
    GlowDot,
    Sphere,
    Text,
    ThreeDAxes,
    ThreeDScene,
    ValueTracker,
    VGroup,
    VMobject,
    linear,
)

from gravitation.render.fonts import pick_font
from gravitation.render.frames import FrameData

SPHERE_RESOLUTION = (32, 16)


class GravityScene(ThreeDScene):
    """Bodies, fading trails, optional axes and a HUD clock."""

    def construct(self) -> None:
        data = FrameData.load(os.environ["GRAVITATION_FRAMES"])
        font = pick_font()

        frame = self.frame
        frame.reorient(data.camera_theta, data.camera_phi, 0, ORIGIN, 8.0 / data.camera_zoom)

        if data.axes:
            extent = data.axis_extent
            axes = ThreeDAxes(
                x_range=(-extent, extent, 1),
                y_range=(-extent, extent, 1),
                z_range=(-extent, extent, 1),
                width=2 * extent,
                height=2 * extent,
                depth=2 * extent,
            )
            axes.set_stroke(width=1, opacity=0.35)
            self.add(axes)

        start = data.positions[0]
        bodies = [
            Sphere(radius=float(r), resolution=SPHERE_RESOLUTION).set_color(c).move_to(p)
            for r, c, p in zip(data.radii, data.colors, start, strict=True)
        ]
        halos = [
            GlowDot(p, radius=float(r) * 5, color=glow, glow_factor=2.0) if glow else None
            for r, p, glow in zip(data.radii, start, data.glow, strict=True)
        ]

        trails: list[VMobject] = []
        if data.trail_frames > 0:
            for c in data.colors:
                trail = VMobject().set_points_as_corners([start[0], start[0]])
                trail.set_stroke(c, width=data.trail_width, opacity=data.trail_opacity)
                trails.append(trail)
            # Without depth testing: overlapping translucent segments of one line would
            # otherwise hide each other and the trail would look dotted. Trails are added
            # before the bodies, so bodies still cover them.
            self.add(*trails, set_depth_test=False)

        self.add(*bodies, *[h for h in halos if h is not None])

        clock = title = None
        if data.hud:
            title = (
                Text(data.title, font=font, font_size=34)
                .set_color(WHITE)
                .to_corner(UL)
                .fix_in_frame()
            )
            clock = Text(data.labels[0], font=font, font_size=27).set_color(WHITE).set_opacity(0.8)
            clock.next_to(title, DOWN, aligned_edge=LEFT).fix_in_frame()
            hint = (
                Text(data.scale_label, font=font, font_size=20).set_color(WHITE).set_opacity(0.45)
            )
            hint.to_corner(DL).fix_in_frame()
            self.add(title, clock, hint)

        tracker = ValueTracker(0.0)
        shown_label = [data.labels[0]]
        fade = data.trail_opacity * np.linspace(0.0, 1.0, 64) ** 1.5

        def update(_: VGroup) -> None:
            k = tracker.get_value()
            positions = data.positions_at(k)
            for i, body in enumerate(bodies):
                body.move_to(positions[i])
                if halos[i] is not None:
                    halos[i].move_to(positions[i])
            for i, trail in enumerate(trails):
                trail.set_points_as_corners(data.trail_points(i, k, positions[i]))
                trail.set_stroke(opacity=fade)
            if clock is not None and data.label_at(k) != shown_label[0]:
                shown_label[0] = data.label_at(k)
                fresh = (
                    Text(shown_label[0], font=font, font_size=27).set_color(WHITE).set_opacity(0.8)
                )
                fresh.next_to(title, DOWN, aligned_edge=LEFT).fix_in_frame()
                clock.become(fresh)
                clock.fix_in_frame()

        driver = VGroup()
        driver.add_updater(update)
        self.add(driver)

        if data.camera_rotation:
            rate = data.camera_rotation * DEGREES
            frame.add_updater(lambda f, dt: f.increment_theta(rate * dt))
        self.play(
            tracker.animate.set_value(data.n_frames - 1),
            run_time=data.duration,
            rate_func=linear,
        )
