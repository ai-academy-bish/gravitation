"""Rendering benchmark: the same frames through every backend, timed end to end.

The report records the machine as well, since the numbers mean little
without it: OS, CPU, the GPU adapter wgpu picked (it shows whether Linux got
real hardware or the ``llvmpipe``/``lavapipe`` software fallback) and package
versions.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from gravitation.render.runner import BACKENDS, QUALITIES


@dataclass
class BenchRow:
    backend: str
    description: str
    times: list[float]
    frames: int
    video_seconds: float
    video_bytes: int
    cold: float | None = None
    error: str = ""

    @property
    def best(self) -> float:
        return min(self.times) if self.times else float("nan")

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else float("nan")

    @property
    def frames_per_second(self) -> float:
        return self.frames / self.best if self.times else float("nan")

    @property
    def realtime_factor(self) -> float:
        """Seconds of video produced per second of wall time (> 1 = faster than real time)."""
        return self.video_seconds / self.best if self.times else float("nan")


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _os_name() -> str:
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    try:
        with open("/etc/os-release", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("PRETTY_NAME="):
                    return f"{line.split('=', 1)[1].strip().strip(chr(34))} ({platform.machine()})"
    except OSError:
        pass
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def _gpu_name() -> str:
    """The adapter wgpu (and therefore ManimGL) would use, or a reason it is unavailable."""
    try:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        info = adapter.info
        device = info.get("device") or info.get("description") or "unknown device"
        return f"{device} · {info.get('backend_type', '?')} · {info.get('adapter_type', '?')}"
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


def _version(dist: str) -> str:
    try:
        version = md.version(dist)
    except md.PackageNotFoundError:
        return "not installed"
    try:
        direct = json.loads(md.distribution(dist).read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
        if commit:
            return f"{version} (git {commit[:10]})"
    except (OSError, ValueError):
        pass
    return version


def environment() -> dict[str, str]:
    """Description of the machine and software the benchmark ran on."""
    return {
        "host": platform.node().split(".")[0] or "unknown",
        "os": _os_name(),
        "cpu": f"{_cpu_name()} · {os.cpu_count()} threads",
        "gpu": _gpu_name(),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "manim": _version("manim"),
        "manimgl": _version("manimgl"),
        "wgpu": _version("wgpu"),
    }


def write_report(
    rows: list[BenchRow],
    env: dict[str, str],
    *,
    config_name: str,
    config_path: str,
    quality: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write the benchmark as Markdown (for people) and JSON (for comparing machines)."""
    width, height, fps = QUALITIES[quality]
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    stem = f"{Path(config_path).stem}_{quality}_{env['host']}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Render benchmark: {config_name}",
        "",
        f"- **Config:** `{config_path}`",
        f"- **Video:** {width}×{height} @ {fps} fps, {rows[0].frames} frames "
        f"({rows[0].video_seconds:.1f} s)"
        if rows
        else "",
        f"- **Date:** {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "| | |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in env.items()],
        "",
        "| Backend | Cold start | Best (warm) | Median | Render speed | × real time | Video size |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.error:
            lines.append(f"| {r.description} | failed | | | | | `{r.error.splitlines()[0][:60]}` |")
            continue
        cold = f"{r.cold:.1f} s" if r.cold is not None else "–"
        lines.append(
            f"| {r.description} | {cold} | {r.best:.1f} s | {r.median:.1f} s | "
            f"{r.frames_per_second:.1f} frames/s | {r.realtime_factor:.2f}× | "
            f"{r.video_bytes / 1e6:.1f} MB |"
        )
    lines += [
        "",
        "Wall time includes interpreter start-up and video encoding, as a user would experience it.",
        "Cold start is the first render of each backend, which also fills its caches",
        "(compiled shaders, typeset text); the other columns are later, warm renders.",
        "",
    ]

    md_path = out_dir / f"{stem}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "config": config_path,
                "quality": quality,
                "environment": env,
                "results": [asdict(r) | {"best": r.best, "median": r.median} for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return md_path, json_path


__all__ = ["BACKENDS", "BenchRow", "environment", "write_report"]
