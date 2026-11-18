"""Command-line interface: ``gravitation <command> ...`` (run ``gravitation -h``)."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

from gravitation import _term as term
from gravitation.config import ConfigError, SystemConfig, load_config
from gravitation.integrators import INTEGRATORS
from gravitation.units import pick_time_unit

OUTPUT = Path("output")
CONFIGS = Path("configs")


# --------------------------------------------------------------------------- helpers


def _output_dir(config: SystemConfig) -> Path:
    return OUTPUT / config.slug


def _fmt_time(config: SystemConfig, seconds: float) -> str:
    if config.units.natural:
        return f"{seconds:g} (N-body units)"
    unit, size = pick_time_unit(seconds)
    return f"{seconds / size:,.4g} {unit}"


def _relative_drift(series: np.ndarray, scale: float) -> float:
    return float(np.max(np.linalg.norm(series - series[0], axis=-1)) / scale) if scale else 0.0


def _load_result(config: SystemConfig, force: bool):
    from gravitation.simulation import load_or_simulate

    path = _output_dir(config) / "simulation.npz"
    result, reused = load_or_simulate(config, path, force=force)
    note = "reused cached simulation" if reused else f"simulated in {result.wall_time:.2f} s"
    term.row("simulation", f"{note} → {path}")
    return result


# --------------------------------------------------------------------------- commands


def cmd_list(_: argparse.Namespace) -> int:
    paths = sorted(CONFIGS.rglob("*.yaml"))
    if not paths:
        term.warn(f"no configs found in {CONFIGS}/")
        return 1
    term.heading(f"Configs in {CONFIGS}/")
    for path in paths:
        try:
            config = load_config(path)
            body_list = ", ".join(config.names[:6]) + (" …" if len(config.names) > 6 else "")
            print(
                f"  {term.style(str(path), 'cyan'):<44} {config.name}  "
                f"{term.style(f'[{len(config.bodies)} bodies: {body_list}]', 'dim')}"
            )
        except ConfigError as exc:
            print(f"  {term.style(str(path), 'red'):<44} invalid: {exc}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from gravitation.simulation import simulate

    config = load_config(args.config)
    sim = config.simulation
    method = INTEGRATORS[args.integrator or sim.integrator]
    term.heading(f"{config.name}  {term.style(f'({args.config})', 'dim')}")
    term.row("bodies", f"{len(config.bodies)} · {', '.join(config.names)}")

    result = simulate(config, integrator=args.integrator)
    out = args.output or _output_dir(config) / "simulation.npz"
    if args.integrator is None:
        result.save(out)

    kind = "symplectic" if method.symplectic else "not symplectic"
    term.row(
        "integrator",
        f"{method.title} (order {method.order}, {kind}) · "
        f"dt = {_fmt_time(config, result.dt)} · {result.steps:,} steps",
    )
    per_step = result.wall_time / max(result.steps, 1) * 1e6
    term.row(
        "simulated",
        f"{_fmt_time(config, result.t[-1])} in {result.wall_time:.2f} s ({per_step:.1f} µs/step)",
    )

    err = result.energy_error
    term.row("energy", f"|ΔE/E₀| max {err.max():.2e}, final {err[-1]:.2e}")
    p_scale = float(np.sum(result.mu / result.G * np.linalg.norm(result.v[0], axis=-1)))
    term.row("momentum", f"|ΔP| / Σ m|v| max {_relative_drift(result.momentum, p_scale):.2e}")
    L = result.angular_momentum
    term.row("ang. mom.", f"|ΔL| / |L₀| max {_relative_drift(L, float(np.linalg.norm(L[0]))):.2e}")
    if args.integrator is None:
        term.row("saved", str(out))
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    from gravitation.render.preview import plot_preview

    config = load_config(args.config)
    term.heading(config.name)
    result = _load_result(config, args.force)
    path = args.output or _output_dir(config) / "preview.png"
    plot_preview(result, config, path, show=args.show)
    term.success(f"preview → {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from gravitation.render.preview import plot_integrator_comparison
    from gravitation.simulation import simulate

    config = load_config(args.config)
    names = args.integrators.split(",") if args.integrators else list(INTEGRATORS)
    term.heading(f"{config.name}: integrators at dt = {_fmt_time(config, config.simulation.dt)}")

    reference = simulate(config, integrator="yoshida4", dt=config.simulation.dt / 8)
    scale = float(np.max(np.linalg.norm(reference.x[0], axis=-1))) or 1.0

    runs = {}
    header = f"  {'method':<20}{'order':>6}{'symp.':>7}{'evals':>7}{'time':>9}{'max |ΔE/E|':>13}{'pos. error':>12}"
    print(term.style(header, "dim"))
    for name in names:
        meta = INTEGRATORS[name]
        result = simulate(config, integrator=name)
        runs[name] = result
        pos_err = float(np.max(np.linalg.norm(result.x[-1] - reference.x[-1], axis=-1)) / scale)
        print(
            f"  {meta.title:<20}{meta.order:>6}{'yes' if meta.symplectic else 'no':>7}"
            f"{meta.force_evals:>7}{result.wall_time:>8.2f}s{result.energy_error.max():>13.2e}"
            f"{pos_err:>12.2e}"
        )
    print(
        term.style(
            "  pos. error: final position error relative to the system size, against "
            "Yoshida 4 with dt/8",
            "dim",
        )
    )
    path = plot_integrator_comparison(runs, config, _output_dir(config) / "integrators.png")
    term.success(f"chart → {path}")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from gravitation.render.frames import build_frames
    from gravitation.render.runner import BACKENDS, QUALITIES, render_video

    config = load_config(args.config)
    width, height, fps = QUALITIES[args.quality]
    term.heading(f"{config.name} → {BACKENDS[args.backend]}, {width}×{height} @ {fps} fps")
    result = _load_result(config, args.force)

    out_dir = _output_dir(config)
    frames = build_frames(result, config, fps, duration=args.duration)
    frames_path = frames.save(out_dir / f"frames_{args.quality}.npz")
    term.row(
        "frames", f"{frames.n_frames} ({frames.duration:.1f} s of video) · {frames.scale_label}"
    )

    name = f"{config.slug}_{args.backend}_{args.quality}"
    outcome = render_video(
        args.backend, frames_path, out_dir, name, quality=args.quality, background=frames.background
    )
    term.row(
        "rendered", f"in {outcome.seconds:.1f} s ({frames.n_frames / outcome.seconds:.1f} frames/s)"
    )
    term.success(f"video → {outcome.video}")
    if args.open:
        import subprocess

        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.run([opener, str(outcome.video)], check=False)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from gravitation.render.bench import BenchRow, environment, write_report
    from gravitation.render.frames import build_frames
    from gravitation.render.runner import BACKENDS, QUALITIES, RenderError, render_video

    config = load_config(args.config)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    width, height, fps = QUALITIES[args.quality]
    term.heading(
        f"Benchmark: {config.name}, {width}×{height} @ {fps} fps, {args.repeat} run(s) per backend"
    )

    env = environment()
    for key, value in env.items():
        term.row(key, value)
    result = _load_result(config, False)

    out_dir = _output_dir(config) / "bench"
    frames = build_frames(result, config, fps, duration=args.duration)
    frames_path = frames.save(out_dir / f"frames_{args.quality}.npz")
    term.row("frames", f"{frames.n_frames} ({frames.duration:.1f} s of video)")

    rows = []
    for backend in backends:
        row = BenchRow(
            backend, BACKENDS.get(backend, backend), [], frames.n_frames, frames.duration, 0
        )
        # The first render of a backend also fills caches (compiled shaders, typeset
        # text), so it is timed separately as the cold start and not counted.
        runs = (["cold"] if args.warmup else []) + [
            f"{i + 1}/{args.repeat}" for i in range(args.repeat)
        ]
        for run in runs:
            label = f"{BACKENDS.get(backend, backend)} [{run}]"
            print(f"  {term.style('…', 'yellow')} {label}", end="", flush=True)
            try:
                outcome = render_video(
                    backend,
                    frames_path,
                    out_dir,
                    f"bench_{backend}_{args.quality}",
                    quality=args.quality,
                    background=frames.background,
                    quiet=True,
                )
            except (RenderError, ValueError) as exc:
                row.error = str(exc)
                print(f"\r  {term.style('✘', 'red')} {label}: {str(exc).splitlines()[0]}")
                break
            if run == "cold":
                row.cold = outcome.seconds
            else:
                row.times.append(outcome.seconds)
            row.video_bytes = outcome.video.stat().st_size
            print(
                f"\r  {term.style('✔', 'green')} {label}: {outcome.seconds:.1f} s "
                f"({frames.n_frames / outcome.seconds:.1f} frames/s)"
            )
        rows.append(row)

    md_path, json_path = write_report(
        rows,
        env,
        config_name=config.name,
        config_path=str(args.config),
        quality=args.quality,
        out_dir=OUTPUT / "bench",
    )
    report = md_path.read_text(encoding="utf-8")
    print("\n" + report[report.index("| Backend") :])
    term.success(f"report → {md_path} (+ {json_path.name})")
    return 0 if all(not r.error for r in rows) else 1


def cmd_ephemeris(args: argparse.Namespace) -> int:
    from gravitation.ephemeris import PRESETS, write_ephemeris_config

    preset = PRESETS[args.preset]
    bodies = args.bodies.split(",") if args.bodies else list(preset["bodies"])  # type: ignore[arg-type]
    epoch = date.fromisoformat(args.date)
    out = (
        args.out or CONFIGS / "jpl" / f"{args.preset if not args.bodies else 'custom'}_{epoch}.yaml"
    )
    name = args.name or f"{preset['name']} on {epoch}"
    term.heading(f"NASA JPL Horizons → {out}")
    write_ephemeris_config(
        Path(out),
        bodies=bodies,
        epoch=epoch,
        name=name,
        duration_days=args.duration or float(preset["duration"]),  # type: ignore[arg-type]
        dt_days=args.dt or float(preset["dt"]),  # type: ignore[arg-type]
    )
    load_config(out)  # validate what we wrote
    term.success(f"config written: {out}")
    print(f"  next: {term.style(f'make render CONFIG={out}', 'cyan')}")
    return 0


def cmd_colors(args: argparse.Namespace) -> int:
    from gravitation.colors import NAMED_COLORS

    names = [n for n in NAMED_COLORS if not args.filter or args.filter.lower() in n]
    term.heading(f'{len(names)} named colours (CSS/HTML); hex codes like "#ff8800" work too')
    width = max(map(len, names), default=0) + 2
    columns = 3
    for i in range(0, len(names), columns):
        cells = [
            f"{term.swatch(NAMED_COLORS[n])} {n:<{width}}{term.style(NAMED_COLORS[n], 'dim')}  "
            for n in names[i : i + columns]
        ]
        print("  " + "".join(cells))
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    from gravitation.ephemeris import PRESETS
    from gravitation.render.runner import BACKENDS, QUALITIES

    parser = argparse.ArgumentParser(
        prog="gravitation",
        description="Newtonian N-body simulator with hand-written integrators and Manim rendering.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def config_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("config", type=Path, help="YAML config, e.g. configs/figure_eight.yaml")

    p = sub.add_parser("list", help="list the configs in configs/")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("simulate", help="integrate a system and print diagnostics")
    config_arg(p)
    p.add_argument(
        "--integrator", choices=list(INTEGRATORS), help="override the config (result not cached)"
    )
    p.add_argument("-o", "--output", type=Path, help="where to save the .npz")
    p.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("preview", help="matplotlib preview: orbits and energy error")
    config_arg(p)
    p.add_argument(
        "-o", "--output", type=Path, help="PNG path (default output/<config>/preview.png)"
    )
    p.add_argument("--show", action="store_true", help="also open an interactive window")
    p.add_argument("--force", action="store_true", help="re-run the simulation even if cached")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("compare", help="run several integrators with the same step and compare")
    config_arg(p)
    p.add_argument("--integrators", help=f"comma-separated subset of: {','.join(INTEGRATORS)}")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("render", help="render a video with Manim")
    config_arg(p)
    p.add_argument("-b", "--backend", choices=list(BACKENDS), default="ce")
    p.add_argument("-q", "--quality", choices=list(QUALITIES), default="m")
    p.add_argument("--duration", type=float, help="video length in seconds (default from config)")
    p.add_argument("--force", action="store_true", help="re-run the simulation even if cached")
    p.add_argument("--open", action="store_true", help="open the video when done")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("bench", help="benchmark rendering backends on the same frames")
    config_arg(p)
    p.add_argument("--backends", default=",".join(BACKENDS), help="comma-separated backends")
    p.add_argument("-q", "--quality", choices=list(QUALITIES), default="m")
    p.add_argument(
        "--repeat", type=int, default=1, help="timed renders per backend (best is reported)"
    )
    p.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="skip the untimed cold-start render of each backend",
    )
    p.add_argument("--duration", type=float, help="video length in seconds (default from config)")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("ephemeris", help="fetch real initial conditions from NASA JPL Horizons")
    p.add_argument("--preset", choices=list(PRESETS), default="inner")
    p.add_argument(
        "--bodies", help="comma-separated names, overrides the preset (e.g. sun,earth,moon)"
    )
    p.add_argument("--date", default=str(date.today()), help="epoch YYYY-MM-DD (default: today)")
    p.add_argument(
        "--out", type=Path, help="config path (default configs/jpl/<preset>_<date>.yaml)"
    )
    p.add_argument("--name", help="system name shown in videos")
    p.add_argument("--duration", type=float, help="simulated time in days")
    p.add_argument("--dt", type=float, help="integration step in days")
    p.set_defaults(func=cmd_ephemeris)

    p = sub.add_parser("colors", help="show the named colours accepted in configs")
    p.add_argument("filter", nargs="?", help="only names containing this text")
    p.set_defaults(func=cmd_colors)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        term.error(f"invalid config: {exc}")
    except FileNotFoundError as exc:
        term.error(str(exc))
    except (RuntimeError, ValueError) as exc:
        term.error(str(exc))
    except KeyboardInterrupt:
        term.error("interrupted")
        return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
