"""Loading and validating YAML descriptions of a system of bodies.

A config has six top-level keys: ``name``, ``description``, ``units``,
``simulation``, ``render`` and ``bodies``. The files in ``configs/`` document
every option. Validation is strict: unknown keys are errors (with a "did you
mean" hint), so a typo such as ``integrater:`` cannot silently fall back to a
default.

Numbers may be written as arithmetic expressions such as ``107e6 / 3600`` or
``7 * 10**9``. This also sidesteps a PyYAML quirk: it reads ``5.972e24``
(exponent without a sign) as a *string*.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import math
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gravitation.colors import DEFAULT_PALETTE, resolve_color
from gravitation.integrators import INTEGRATORS
from gravitation.units import LENGTH_UNITS, UnitSystem, normalise_unit


class ConfigError(ValueError):
    """An invalid configuration. The message starts with the offending key, e.g. ``bodies[2].mass``."""


DEFAULT_RADIUS = 0.08
"""Visual radius of a body in scene units when the config gives none."""

SAMPLES_PER_VIDEO_SECOND = 240
"""With ``samples: auto``, the simulation stores this many snapshots per second of video
(enough for smooth trails at 60 fps with 4 trail points per frame)."""


@dataclass(frozen=True)
class Exaggeration:
    """Draw a satellite ``factor`` times further from its parent than it really is.

    This is purely visual: the Moon is only 0.0026 au from the Earth and would
    otherwise disappear inside the Earth's sphere on a solar-system scale.
    """

    around: str
    factor: float


@dataclass(eq=False)
class Body:
    """One body, converted to internal units (SI, or N-body units).

    Attributes:
        name: unique name, shown in legends.
        gm: gravitational parameter μ = G·m [m³/s²].
        position: initial position [m], shape ``(3,)``.
        velocity: initial velocity [m/s], shape ``(3,)``.
        color: ``#rrggbb``.
        radius: visual radius in scene units; it has no effect on the physics.
        glow: halo colour ``#rrggbb`` (stars, black holes), or ``None`` for no halo.
        exaggerate: optional visual exaggeration of the orbit around a parent.
    """

    name: str
    gm: float
    position: np.ndarray
    velocity: np.ndarray
    color: str
    radius: float = DEFAULT_RADIUS
    glow: str | None = None
    exaggerate: Exaggeration | None = None


@dataclass(frozen=True)
class SimulationSettings:
    """How to integrate. Times and lengths are in internal units (s, m)."""

    duration: float
    dt: float
    integrator: str = "verlet"
    softening: float = 0.0
    barycentric: bool = True
    samples: int = 1201


@dataclass(frozen=True)
class CameraSettings:
    """Camera orientation; angles in degrees.

    ``phi`` is the polar angle from the +z axis (0 = looking straight down,
    90 = edge-on), ``theta`` the azimuth around z, ``rotation`` an ambient
    spin in degrees per second of video.
    """

    phi: float = 65.0
    theta: float = -30.0
    zoom: float = 1.0
    rotation: float = 3.0


@dataclass(frozen=True)
class RenderSettings:
    """How to draw.

    ``scale`` is in scene units per ``scale_unit`` (default: the config's
    length unit); ``None`` fits the system to the frame.
    """

    duration: float = 20.0
    center: str = "barycenter"
    scale: float | None = None
    scale_unit: str = ""
    trail: float | None = 4.0
    trail_width: float = 2.0
    trail_opacity: float = 0.75
    camera: CameraSettings = field(default_factory=CameraSettings)
    background: str = "#0b0d17"
    hud: bool = True
    axes: bool = False


@dataclass(eq=False)
class SystemConfig:
    """A fully validated system: units, settings and bodies in internal units."""

    name: str
    description: str
    units: UnitSystem
    simulation: SimulationSettings
    render: RenderSettings
    bodies: list[Body]
    path: Path | None = None

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.bodies]

    @property
    def mu(self) -> np.ndarray:
        """Gravitational parameters, shape ``(N,)``."""
        return np.array([b.gm for b in self.bodies], dtype=float)

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Initial positions and velocities, each of shape ``(N, 3)`` (fresh copies)."""
        x = np.array([b.position for b in self.bodies], dtype=float)
        v = np.array([b.velocity for b in self.bodies], dtype=float)
        return x, v

    def index(self, name: str) -> int:
        """Index of the body called ``name``."""
        return self.names.index(name)

    def physics_hash(self) -> str:
        """Fingerprint of everything that affects the trajectory (not the rendering).

        Used to reuse a saved simulation when only render settings changed.
        """
        sim = self.simulation
        payload = {
            "G": self.units.G,
            "sim": [
                sim.duration,
                sim.dt,
                sim.integrator,
                sim.softening,
                sim.barycentric,
                sim.samples,
            ],
            "bodies": [
                [b.name, b.gm, b.position.tolist(), b.velocity.tolist()] for b in self.bodies
            ],
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def slug(self) -> str:
        """Short identifier for output folders: the config file name without extension."""
        if self.path is not None:
            return self.path.stem
        return "".join(c if c.isalnum() else "_" for c in self.name.lower()).strip("_") or "system"


# --------------------------------------------------------------------------- parsing helpers

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_CONSTANTS = {"pi": math.pi}


def evaluate_number(value: Any) -> float:
    """Turn a YAML scalar into a float, evaluating simple arithmetic safely.

    Allowed: numbers, ``+ - * / **``, parentheses and the constant ``pi``.
    Nothing else is evaluated, so a config cannot run code.
    """
    if isinstance(value, bool):
        raise ValueError(f"expected a number, got {value!r}")
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"expected a number, got {type(value).__name__}")
    try:
        tree = ast.parse(value.strip(), mode="eval")
        result = _eval_node(tree.body)
    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"cannot evaluate {value!r}: {exc}") from None
    if not math.isfinite(result):
        raise ValueError(f"{value!r} is not finite")
    return result


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise OverflowError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise TypeError(f"unsupported expression element {ast.dump(node)[:40]}")


class _Section:
    """A mapping that remembers which keys were read, to reject unknown ones."""

    def __init__(self, data: Any, where: str) -> None:
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(f"{where or 'config'}: expected a mapping, got {type(data).__name__}")
        self.data = data
        self.where = where
        self.known: set[str] = set()

    def path(self, key: str) -> str:
        return f"{self.where}.{key}" if self.where else key

    def raw(self, key: str, default: Any = None) -> Any:
        self.known.add(key)
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        self.known.add(key)
        return self.data.get(key) is not None

    def number(
        self,
        key: str,
        default: float | None = None,
        *,
        minimum: float | None = None,
        positive: bool = False,
    ) -> float:
        value = self.raw(key)
        if value is None:
            if default is None:
                raise ConfigError(f"{self.path(key)}: required")
            return default
        try:
            number = evaluate_number(value)
        except ValueError as exc:
            raise ConfigError(f"{self.path(key)}: {exc}") from None
        if positive and number <= 0:
            raise ConfigError(f"{self.path(key)}: must be positive, got {number:g}")
        if minimum is not None and number < minimum:
            raise ConfigError(f"{self.path(key)}: must be at least {minimum:g}, got {number:g}")
        return number

    def boolean(self, key: str, default: bool) -> bool:
        value = self.raw(key, default)
        if not isinstance(value, bool):
            raise ConfigError(f"{self.path(key)}: expected true or false, got {value!r}")
        return value

    def text(self, key: str, default: str | None = None) -> str:
        value = self.raw(key, default)
        if value is None:
            raise ConfigError(f"{self.path(key)}: required")
        return str(value).strip()

    def vector(self, key: str) -> np.ndarray:
        value = self.raw(key)
        if value is None:
            raise ConfigError(f"{self.path(key)}: required")
        if not isinstance(value, list | tuple) or len(value) not in (2, 3):
            raise ConfigError(f"{self.path(key)}: expected [x, y, z] (or [x, y]), got {value!r}")
        try:
            components = [evaluate_number(c) for c in value]
        except ValueError as exc:
            raise ConfigError(f"{self.path(key)}: {exc}") from None
        if len(components) == 2:
            components.append(0.0)
        return np.array(components, dtype=float)

    def section(self, key: str) -> _Section:
        return _Section(self.raw(key), self.path(key))

    def finish(self) -> None:
        for key in self.data:
            if key not in self.known:
                hint = difflib.get_close_matches(str(key), sorted(self.known), n=1)
                tail = f" — did you mean '{hint[0]}'?" if hint else ""
                raise ConfigError(f"{self.path(str(key))}: unknown key{tail}")


# --------------------------------------------------------------------------- loading


def load_config(path: str | Path) -> SystemConfig:
    """Read and validate a YAML config file.

    Raises:
        ConfigError: with a message naming the offending key.
        FileNotFoundError: if the file does not exist.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML: {exc}") from None
    return parse_config(data, path=path)


def parse_config(data: Any, path: Path | None = None) -> SystemConfig:
    """Validate an already-parsed config mapping. See :func:`load_config`."""
    top = _Section(data, "")
    name = top.text("name", path.stem.replace("_", " ").title() if path else "Untitled system")
    description = top.text("description", "")
    try:
        units = UnitSystem.from_config(top.raw("units"))
    except ValueError as exc:
        raise ConfigError(str(exc)) from None

    render = _parse_render(top.section("render"), units)
    simulation = _parse_simulation(top.section("simulation"), units, render)
    bodies = _parse_bodies(top.raw("bodies"), units)
    top.finish()

    names = [b.name for b in bodies]
    if render.center not in ("barycenter", "origin") and render.center not in names:
        raise ConfigError(
            f"render.center: {render.center!r} is not 'barycenter', 'origin' or a body name"
        )
    for i, body in enumerate(bodies):
        ex = body.exaggerate
        if ex is None:
            continue
        where = f"bodies[{i}].exaggerate.around"
        if ex.around not in names or ex.around == body.name:
            raise ConfigError(f"{where}: {ex.around!r} must be the name of another body")
        if bodies[names.index(ex.around)].exaggerate is not None:
            raise ConfigError(
                f"{where}: {ex.around!r} is itself exaggerated; chains are not supported"
            )

    if simulation.softening == 0:
        for i, a in enumerate(bodies):
            for b in bodies[i + 1 :]:
                if np.array_equal(a.position, b.position):
                    raise ConfigError(
                        f"bodies {a.name!r} and {b.name!r} start at the same position; "
                        "move one of them or set simulation.softening"
                    )

    return SystemConfig(name, description, units, simulation, render, bodies, path)


def _parse_render(sec: _Section, units: UnitSystem) -> RenderSettings:
    duration = sec.number("duration", 20.0, positive=True)
    center = sec.text("center", "barycenter")

    scale_raw = sec.raw("scale", "auto")
    scale = None
    if not (isinstance(scale_raw, str) and scale_raw.strip().lower() == "auto"):
        scale = sec.number("scale", positive=True)
    scale_unit = sec.text("scale_unit", "")
    if scale_unit:
        if units.natural:
            raise ConfigError(f"{sec.path('scale_unit')}: not available with N-body units")
        try:
            scale_unit = normalise_unit(scale_unit, LENGTH_UNITS, "length")
        except ValueError as exc:
            raise ConfigError(f"{sec.path('scale_unit')}: {exc}") from None

    trail_raw = sec.raw("trail", 4.0)
    trail: float | None
    if trail_raw is False or (isinstance(trail_raw, str) and trail_raw.lower() in ("none", "off")):
        trail = 0.0
    elif isinstance(trail_raw, str) and trail_raw.lower() in ("full", "all"):
        trail = None
    else:
        trail = sec.number("trail", 4.0, minimum=0.0)

    cam = sec.section("camera")
    camera = CameraSettings(
        phi=cam.number("phi", CameraSettings.phi),
        theta=cam.number("theta", CameraSettings.theta),
        zoom=cam.number("zoom", CameraSettings.zoom, positive=True),
        rotation=cam.number("rotation", CameraSettings.rotation),
    )
    cam.finish()

    try:
        background = resolve_color(sec.raw("background", RenderSettings.background))
    except ValueError as exc:
        raise ConfigError(f"{sec.path('background')}: {exc}") from None

    opacity = sec.number("trail_opacity", RenderSettings.trail_opacity, minimum=0.0)
    if opacity > 1:
        raise ConfigError(f"{sec.path('trail_opacity')}: must be between 0 and 1")

    settings = RenderSettings(
        duration=duration,
        center=center,
        scale=scale,
        scale_unit=scale_unit,
        trail=trail,
        trail_width=sec.number("trail_width", RenderSettings.trail_width, minimum=0.0),
        trail_opacity=opacity,
        camera=camera,
        background=background,
        hud=sec.boolean("hud", RenderSettings.hud),
        axes=sec.boolean("axes", RenderSettings.axes),
    )
    sec.finish()
    return settings


def _parse_simulation(
    sec: _Section, units: UnitSystem, render: RenderSettings
) -> SimulationSettings:
    duration = sec.number("duration", positive=True) * units.time_si
    dt = sec.number("dt", positive=True) * units.time_si
    if dt > duration:
        raise ConfigError(f"{sec.path('dt')}: step is longer than the whole simulation")

    integrator = sec.text("integrator", "verlet").lower()
    if integrator not in INTEGRATORS:
        hint = difflib.get_close_matches(integrator, list(INTEGRATORS), n=1)
        tail = f" — did you mean '{hint[0]}'?" if hint else ""
        raise ConfigError(
            f"{sec.path('integrator')}: unknown integrator {integrator!r}{tail} "
            f"(choose from: {', '.join(INTEGRATORS)})"
        )

    samples_raw = sec.raw("samples", "auto")
    if isinstance(samples_raw, str) and samples_raw.strip().lower() == "auto":
        samples = math.ceil(render.duration * SAMPLES_PER_VIDEO_SECOND) + 1
    else:
        samples = int(sec.number("samples", minimum=2))

    settings = SimulationSettings(
        duration=duration,
        dt=dt,
        integrator=integrator,
        softening=sec.number("softening", 0.0, minimum=0.0) * units.length_si,
        barycentric=sec.boolean("barycentric", True),
        samples=samples,
    )
    sec.finish()
    return settings


def _parse_bodies(raw: Any, units: UnitSystem) -> list[Body]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError("bodies: expected a non-empty list of bodies")

    bodies: list[Body] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        sec = _Section(item, f"bodies[{i}]")
        name = sec.text("name", f"Body {i + 1}")
        if name in seen:
            raise ConfigError(f"{sec.path('name')}: duplicate body name {name!r}")
        seen.add(name)

        has_mass, has_gm = sec.has("mass"), sec.has("gm")
        if has_mass == has_gm:
            raise ConfigError(f"bodies[{i}] ({name}): give exactly one of 'mass' or 'gm'")
        if has_mass:
            gm = sec.number("mass", minimum=0.0) * units.mass_gm
        else:
            gm = sec.number("gm", minimum=0.0) * units.gm_si

        color_raw = sec.raw("color", DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])
        try:
            color = resolve_color(color_raw)
        except ValueError as exc:
            raise ConfigError(f"{sec.path('color')}: {exc}") from None

        exaggerate = None
        if sec.has("exaggerate"):
            ex = sec.section("exaggerate")
            exaggerate = Exaggeration(ex.text("around"), ex.number("factor", positive=True))
            ex.finish()

        glow_raw = sec.raw("glow", False)
        glow: str | None = None
        if glow_raw is True:
            glow = color
        elif glow_raw not in (False, None):
            try:
                glow = resolve_color(glow_raw)
            except ValueError as exc:
                raise ConfigError(
                    f"{sec.path('glow')}: expected true/false or a colour; {exc}"
                ) from None

        bodies.append(
            Body(
                name=name,
                gm=gm,
                position=sec.vector("position") * units.length_si,
                velocity=sec.vector("velocity") * units.velocity_si,
                color=color,
                radius=sec.number("radius", DEFAULT_RADIUS, positive=True),
                glow=glow,
                exaggerate=exaggerate,
            )
        )
        sec.finish()
    return bodies
