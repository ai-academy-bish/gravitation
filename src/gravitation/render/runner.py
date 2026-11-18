"""Launching the Manim back-ends as subprocesses.

Both Manim flavours are driven through their own command-line interfaces,
which are their stable public API. Each render runs in a fresh process, so
benchmark timings include start-up cost, exactly as a user experiences it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

QUALITIES: dict[str, tuple[int, int, int]] = {
    "l": (854, 480, 30),
    "m": (1280, 720, 30),
    "h": (1920, 1080, 60),
    "k": (3840, 2160, 60),
}
"""Quality presets: ``(width, height, fps)``. The same values are passed to every backend."""

BACKENDS: dict[str, str] = {
    "ce": "Manim CE · Cairo (CPU)",
    "ce-gl": "Manim CE · OpenGL",
    "gl": "ManimGL · wgpu (GPU)",
}
"""Backend identifiers and descriptions."""

_HERE = Path(__file__).parent


class RenderError(RuntimeError):
    """A Manim subprocess failed; the message contains the tail of its output."""


@dataclass(frozen=True)
class RenderOutcome:
    """A finished render: where the video is and how long it took."""

    backend: str
    video: Path
    seconds: float


def check_backend(backend: str) -> None:
    """Raise a helpful error if a backend's Python package is not installed."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r} (choose from: {', '.join(BACKENDS)})")
    module = "manimlib" if backend == "gl" else "manim"
    import importlib.util

    if importlib.util.find_spec(module) is None:
        extra = "gl" if backend == "gl" else "ce"
        raise RenderError(
            f"backend {backend!r} needs the '{extra}' extra: run `make install EXTRAS={extra}` "
            f"(or `uv sync --extra {extra}`)"
        )


def needs_virtual_display() -> bool:
    """True on Linux without a display, where CE's OpenGL renderer needs Xvfb.

    ManimGL does not: since its move to wgpu it renders off-screen through Vulkan.
    """
    if not sys.platform.startswith("linux"):
        return False
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return False
    if shutil.which("xvfb-run") is None:
        raise RenderError(
            "no display and no xvfb-run: install Xvfb (make sysdeps) or use --backend ce / gl"
        )
    return True


def build_command(
    backend: str, frames: Path, out_dir: Path, name: str, quality: str, background: str
) -> list[str]:
    """The command line that renders ``frames`` with ``backend``."""
    width, height, fps = QUALITIES[quality]
    if backend in ("ce", "ce-gl"):
        return [
            sys.executable,
            "-m",
            "manim",
            "render",
            str(_HERE / "manim_ce.py"),
            "GravityScene",
            "--renderer",
            "cairo" if backend == "ce" else "opengl",
            "--format",
            "mp4",
            "-r",
            f"{width},{height}",
            "--fps",
            str(fps),
            "--media_dir",
            str(out_dir / "media"),
            "-o",
            name,
            "--disable_caching",
            "-v",
            "WARNING",
        ]
    return [
        sys.executable,
        "-m",
        "manimlib",
        str(_HERE / "manim_gl.py"),
        "GravityScene",
        "-w",
        "-r",
        f"{width}x{height}",
        "--fps",
        str(fps),
        "-c",
        background,
        "--video_dir",
        str(out_dir / "media"),
        "--file_name",
        name,
        "--log-level",
        "WARNING",
    ]


def render_video(
    backend: str,
    frames: Path,
    out_dir: Path,
    name: str,
    *,
    quality: str = "m",
    background: str = "#000000",
    quiet: bool = False,
) -> RenderOutcome:
    """Render prepared frames to ``out_dir/<name>.mp4`` and time it.

    Args:
        backend: ``ce``, ``ce-gl`` or ``gl``.
        frames: a file written by :meth:`FrameData.save`.
        out_dir: folder for the final video; Manim's own files go to ``out_dir/media``.
        name: video file name without extension.
        quality: a key of :data:`QUALITIES`.
        background: background colour (ManimGL takes it on the command line).
        quiet: capture Manim's output instead of streaming it (it is shown on failure).

    Raises:
        RenderError: if Manim exits with an error or produces no video.
    """
    check_backend(backend)
    out_dir = out_dir.resolve()  # Manim runs with cwd=out_dir, so relative paths would nest
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(backend, frames, out_dir, name, quality, background)
    env = {**os.environ, "GRAVITATION_FRAMES": str(frames.resolve())}
    if backend == "gl":
        env.setdefault("PYTHONWARNINGS", "ignore")

    if backend == "ce-gl" and needs_virtual_display():
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1920x1080x24", *cmd]

    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=out_dir,
        text=True,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.STDOUT if quiet else None,
        check=False,
    )
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-30:])
        raise RenderError(f"{BACKENDS[backend]} failed (exit {proc.returncode})\n{tail}")

    produced = sorted((out_dir / "media").rglob(f"{name}.mp4"), key=lambda p: p.stat().st_mtime)
    if not produced:
        raise RenderError(f"{BACKENDS[backend]} finished but no {name}.mp4 was found")
    video = out_dir / f"{name}.mp4"
    shutil.move(str(produced[-1]), video)
    return RenderOutcome(backend, video, seconds)
