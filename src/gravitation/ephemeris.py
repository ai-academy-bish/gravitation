"""Real initial conditions from NASA JPL Horizons.

Horizons (https://ssd.jpl.nasa.gov/horizons/) returns state vectors (position
and velocity) of solar-system bodies at a chosen instant. We query it in the
ecliptic J2000 frame, centred on the solar-system barycentre, in au and
au/day. Masses come from the DE440 gravitational parameters below, converted
to au³/day².

For planets that have moons we fetch the *planet-system barycentre* (Horizons
IDs 4–9) and use the system GM. A simulation without those moons then moves
the system's true centre of mass. The Earth and Moon are kept separate (IDs
399 and 301) because the Moon is one of the bodies we want to watch.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from gravitation.units import AU, DAY

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"


@dataclass(frozen=True)
class CatalogBody:
    """A body Horizons can provide.

    Attributes:
        name: display name.
        horizons_id: Horizons ``COMMAND`` value.
        gm_km3_s2: gravitational parameter from JPL DE440 [km³/s²].
        radius_km: mean physical radius, used to derive a visual radius.
        color: default display colour.
        glow: draw a halo by default.
    """

    name: str
    horizons_id: str
    gm_km3_s2: float
    radius_km: float
    color: str
    glow: bool = False


CATALOG: dict[str, CatalogBody] = {
    b.name.lower(): b
    for b in (
        CatalogBody("Sun", "10", 132712440041.279419, 695700.0, "#fdb813", glow=True),
        CatalogBody("Mercury", "199", 22031.868551, 2439.7, "#a8a29e"),
        CatalogBody("Venus", "299", 324858.592000, 6051.8, "#e9c46a"),
        CatalogBody("Earth", "399", 398600.435507, 6371.0, "#3a86ff"),
        CatalogBody("Moon", "301", 4902.800118, 1737.4, "#d9d9d9"),
        CatalogBody("Mars", "4", 42828.375816, 3389.5, "#e76f51"),
        CatalogBody("Jupiter", "5", 126712764.100000, 69911.0, "#d4a373"),
        CatalogBody("Saturn", "6", 37940584.841800, 58232.0, "#f4d58d"),
        CatalogBody("Uranus", "7", 5794556.400000, 25362.0, "#8ecae6"),
        CatalogBody("Neptune", "8", 6836527.100580, 24622.0, "#4361ee"),
        CatalogBody("Pluto", "9", 975.500000, 1188.3, "#c9ada7"),
    )
}
"""Bodies known to :func:`fetch_state`, keyed by lower-case name."""

PRESETS: dict[str, dict[str, object]] = {
    "inner": {
        "bodies": ["Sun", "Mercury", "Venus", "Earth", "Moon", "Mars"],
        "name": "Inner Solar System",
        "duration": 730.0,
        "dt": 0.02,
    },
    "outer": {
        "bodies": ["Sun", "Jupiter", "Saturn", "Uranus", "Neptune"],
        "name": "Outer Solar System",
        "duration": 60_000.0,
        "dt": 2.0,
    },
    "full": {
        "bodies": [
            "Sun",
            "Mercury",
            "Venus",
            "Earth",
            "Moon",
            "Mars",
            "Jupiter",
            "Saturn",
            "Uranus",
            "Neptune",
            "Pluto",
        ],
        "name": "Solar System",
        "duration": 4380.0,
        "dt": 0.02,
    },
}
"""Ready-made body selections with a sensible duration and step (days)."""


@dataclass(frozen=True)
class StateVector:
    """Position [au] and velocity [au/day] of a body at a given epoch."""

    body: CatalogBody
    position: np.ndarray
    velocity: np.ndarray


def julian_date(day: date) -> float:
    """Julian Date at 00:00 of a calendar day (TDB, as Horizons expects)."""
    epoch = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return 2440587.5 + epoch.timestamp() / DAY


def _ssl_context() -> ssl.SSLContext:
    """TLS context backed by the operating system's certificate store when possible."""
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


def fetch_state(body: CatalogBody, epoch: date, *, timeout: float = 30.0) -> StateVector:
    """Ask Horizons for the barycentric ecliptic state vector of one body.

    Raises:
        RuntimeError: if Horizons cannot be reached or returns no vectors.
    """
    params = {
        "format": "json",
        "COMMAND": f"'{body.horizons_id}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": "'500@0'",  # solar-system barycentre
        "TLIST": f"'{julian_date(epoch):.6f}'",
        "TLIST_TYPE": "'JD'",
        "VEC_TABLE": "'2'",  # position and velocity
        "REF_PLANE": "'ECLIPTIC'",
        "OUT_UNITS": "'AU-D'",
        "CSV_FORMAT": "'YES'",
        "VEC_LABELS": "'NO'",
    }
    url = f"{HORIZONS_URL}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=_ssl_context()) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"cannot reach JPL Horizons for {body.name}: {exc}") from None

    text = payload.get("result", "")
    if "$$SOE" not in text:
        raise RuntimeError(f"Horizons returned no vectors for {body.name}:\n{text[:500]}")
    row = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0].strip().splitlines()[0]
    fields = [f.strip() for f in row.split(",") if f.strip()]
    numbers = [float(f) for f in fields[2:8]]  # JDTDB, calendar date, X, Y, Z, VX, VY, VZ
    return StateVector(body, np.array(numbers[:3]), np.array(numbers[3:]))


def visual_radius(radius_km: float) -> float:
    """Map a physical radius to a visual one (scene units) on a compressed log scale.

    Real radii span a factor of 600 between the Moon and the Sun, and all of
    them are invisible at orbital scale, so sizes only hint at relative
    magnitude.
    """
    return round(max(0.035, 0.045 + 0.045 * np.log10(radius_km / 1000.0)), 3)


def resolve_bodies(names: list[str]) -> list[CatalogBody]:
    """Map user-supplied names (any case) to catalog entries."""
    bodies = []
    for name in names:
        key = name.strip().lower()
        if key not in CATALOG:
            raise ValueError(
                f"unknown body {name!r} (known: {', '.join(b.name for b in CATALOG.values())})"
            )
        bodies.append(CATALOG[key])
    return bodies


def build_config_yaml(
    states: list[StateVector],
    *,
    epoch: date,
    name: str,
    duration_days: float,
    dt_days: float,
) -> str:
    """Render fetched state vectors as a commented YAML config."""
    gm_unit = AU**3 / DAY**2 / 1e9  # km³/s² per au³/day²
    lines = [
        f"# Generated by `gravitation ephemeris` from NASA JPL Horizons on {date.today()}.",
        f"# Epoch {epoch} 00:00 TDB (JD {julian_date(epoch):.1f}); ecliptic J2000 frame,",
        "# solar-system barycentre. Masses are JPL DE440 GM values.",
        f"name: {name}",
        f"description: Real positions and velocities of {len(states)} bodies on {epoch}.",
        "",
        "units:",
        "  length: au",
        "  time: day",
        "  mass: solar_mass",
        "",
        "simulation:",
        f"  duration: {duration_days:g}       # days",
        f"  dt: {dt_days:g}               # days",
        "  integrator: yoshida4",
        "  barycentric: true",
        "",
        "render:",
        "  duration: 20",
        "  center: barycenter",
        "  scale: auto",
        "  trail: 5",
        "  camera: {phi: 60, theta: -30, zoom: 1.0, rotation: 2}",
        "",
        "bodies:",
    ]
    for state in states:
        b = state.body
        pos = ", ".join(f"{c: .15e}" for c in state.position)
        vel = ", ".join(f"{c: .15e}" for c in state.velocity)
        lines += [
            f"  - name: {b.name}",
            f"    gm: {b.gm_km3_s2 / gm_unit:.15e}   # au³/day² (= {b.gm_km3_s2:.6f} km³/s²)",
            f"    position: [{pos}]",
            f"    velocity: [{vel}]",
            f"    radius: {visual_radius(b.radius_km)}",
            f'    color: "{b.color}"',
        ]
        if b.glow:
            lines.append("    glow: true")
        if b.name == "Moon" and any(s.body.name == "Earth" for s in states):
            lines.append("    exaggerate: {around: Earth, factor: 40}   # visual only")
    return "\n".join(lines) + "\n"


def write_ephemeris_config(
    out: Path,
    *,
    bodies: list[str],
    epoch: date,
    name: str,
    duration_days: float,
    dt_days: float,
    log=print,
) -> Path:
    """Fetch every body from Horizons and write a ready-to-run config file."""
    states = []
    for body in resolve_bodies(bodies):
        log(f"  fetching {body.name:<8} (Horizons id {body.horizons_id})")
        states.append(fetch_state(body, epoch))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_config_yaml(
            states, epoch=epoch, name=name, duration_days=duration_days, dt_days=dt_days
        ),
        encoding="utf-8",
    )
    return out
