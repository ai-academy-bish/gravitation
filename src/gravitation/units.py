"""Units and physical constants.

Physical simulations run internally in SI units (metre, second, kilogram).
Masses are carried as *gravitational parameters* μ = G·m [m³/s²]. For the Sun,
μ is known to about 11 significant digits while G on its own is known to only
about 5, so working with μ keeps real ephemerides accurate.

A config may instead use N-body ("natural") units, in which G = 1 and every
quantity is dimensionless. This suits idealised systems such as the
figure-eight three-body choreography.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any

G_SI = 6.67430e-11
"""Newtonian constant of gravitation, CODATA 2018 [m³ kg⁻¹ s⁻²]."""

AU = 1.495978707e11
"""Astronomical unit, exact by the IAU 2012 definition [m]."""

DAY = 86_400.0
"""One day [s]."""

JULIAN_YEAR = 365.25 * DAY
"""One Julian year [s]."""

GM_SUN = 1.32712440041279419e20
"""Heliocentric gravitational constant, JPL DE440 [m³/s²]."""

GM_EARTH = 3.98600435507e14
"""Geocentric gravitational constant, JPL DE440 [m³/s²]."""

GM_JUPITER_SYSTEM = 1.267127641e17
"""Gravitational parameter of Jupiter and its moons, JPL DE440 [m³/s²]."""

LENGTH_UNITS: dict[str, float] = {
    "m": 1.0,
    "km": 1.0e3,
    "earth_radius": 6.3781e6,
    "solar_radius": 6.957e8,
    "au": AU,
    "ly": 9.4607304725808e15,
    "pc": 3.0856775814913673e16,
}
"""Length units and their size in metres."""

TIME_UNITS: dict[str, float] = {
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "day": DAY,
    "yr": JULIAN_YEAR,
}
"""Time units and their length in seconds."""

MASS_UNITS: dict[str, float] = {
    "kg": G_SI,
    "earth_mass": GM_EARTH,
    "jupiter_mass": GM_JUPITER_SYSTEM,
    "solar_mass": GM_SUN,
}
"""Mass units, each given as the gravitational parameter μ = G·m of one unit [m³/s²]."""

_ALIASES: dict[str, str] = {
    "meter": "m",
    "metre": "m",
    "meters": "m",
    "metres": "m",
    "kilometer": "km",
    "kilometre": "km",
    "astronomical_unit": "au",
    "light_year": "ly",
    "parsec": "pc",
    "r_earth": "earth_radius",
    "r_sun": "solar_radius",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "hr": "h",
    "d": "day",
    "days": "day",
    "y": "yr",
    "year": "yr",
    "years": "yr",
    "kilogram": "kg",
    "msun": "solar_mass",
    "m_sun": "solar_mass",
    "solar_masses": "solar_mass",
    "mearth": "earth_mass",
    "m_earth": "earth_mass",
    "mjup": "jupiter_mass",
    "m_jup": "jupiter_mass",
}

NATURAL_NAMES = frozenset({"nbody", "natural", "g=1"})
"""Spellings accepted for N-body units in a config's ``units:`` field."""


def normalise_unit(name: Any, table: dict[str, float], kind: str) -> str:
    """Return the canonical spelling of a unit, or raise :class:`ValueError`.

    Args:
        name: unit as written by the user, e.g. ``"AU"``, ``"days"``, ``"Msun"``.
        table: one of :data:`LENGTH_UNITS`, :data:`TIME_UNITS`, :data:`MASS_UNITS`.
        kind: ``"length"``, ``"time"`` or ``"mass"``, used in the error message.
    """
    key = str(name).strip().lower().replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key in table:
        return key
    hint = difflib.get_close_matches(key, list(table) + list(_ALIASES), n=1)
    suggestion = f" — did you mean {_ALIASES.get(hint[0], hint[0])!r}?" if hint else ""
    raise ValueError(f"unknown {kind} unit {name!r}{suggestion} (choose from: {', '.join(table)})")


@dataclass(frozen=True)
class UnitSystem:
    """The units in which a config writes its numbers.

    Velocities default to ``length/time`` but may be given separately: real
    planetary data reads better as positions in au and velocities in km/s.

    Attributes:
        length: canonical length unit (a key of :data:`LENGTH_UNITS`).
        time: canonical time unit (a key of :data:`TIME_UNITS`).
        mass: canonical mass unit (a key of :data:`MASS_UNITS`).
        velocity: ``"<length>/<time>"``, or an empty string for ``length/time``.
        natural: N-body units: G = 1 and every scale factor is 1.
    """

    length: str = "m"
    time: str = "s"
    mass: str = "kg"
    velocity: str = ""
    natural: bool = False

    @classmethod
    def nbody(cls) -> UnitSystem:
        """N-body units: G = 1, lengths, times and masses are dimensionless."""
        return cls("nbody", "nbody", "nbody", "nbody", natural=True)

    @classmethod
    def from_config(cls, raw: Any) -> UnitSystem:
        """Build a unit system from the ``units:`` entry of a config.

        Accepted forms: missing (SI), the string ``si`` or ``nbody``, or a
        mapping with any of the keys ``length``, ``time``, ``mass``, ``velocity``.
        """
        if raw is None:
            return cls()
        if isinstance(raw, str):
            key = raw.strip().lower()
            if key in NATURAL_NAMES:
                return cls.nbody()
            if key == "si":
                return cls()
            raise ValueError(f"units: expected 'si', 'nbody' or a mapping, got {raw!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"units: expected a mapping, got {type(raw).__name__}")

        allowed = ("length", "time", "mass", "velocity")
        for key in raw:
            if key not in allowed:
                hint = difflib.get_close_matches(str(key), allowed, n=1)
                tail = f" — did you mean {hint[0]!r}?" if hint else ""
                raise ValueError(f"units.{key}: unknown key{tail}")

        length = normalise_unit(raw.get("length", "m"), LENGTH_UNITS, "length")
        time = normalise_unit(raw.get("time", "s"), TIME_UNITS, "time")
        mass = normalise_unit(raw.get("mass", "kg"), MASS_UNITS, "mass")
        velocity = ""
        if raw.get("velocity"):
            text = str(raw["velocity"])
            if text.count("/") != 1:
                raise ValueError(f"units.velocity: expected '<length>/<time>', got {text!r}")
            v_len, v_time = text.split("/")
            velocity = (
                f"{normalise_unit(v_len, LENGTH_UNITS, 'length')}/"
                f"{normalise_unit(v_time, TIME_UNITS, 'time')}"
            )
        return cls(length, time, mass, velocity)

    @property
    def G(self) -> float:
        """Gravitational constant in the internal units (SI, or 1 for N-body units)."""
        return 1.0 if self.natural else G_SI

    @property
    def length_si(self) -> float:
        """Size of one length unit in internal units [m]."""
        return 1.0 if self.natural else LENGTH_UNITS[self.length]

    @property
    def time_si(self) -> float:
        """Length of one time unit in internal units [s]."""
        return 1.0 if self.natural else TIME_UNITS[self.time]

    @property
    def velocity_si(self) -> float:
        """Size of one velocity unit in internal units [m/s]."""
        if self.natural:
            return 1.0
        if self.velocity:
            v_len, v_time = self.velocity.split("/")
            return LENGTH_UNITS[v_len] / TIME_UNITS[v_time]
        return self.length_si / self.time_si

    @property
    def mass_gm(self) -> float:
        """Gravitational parameter μ of one mass unit [m³/s²]."""
        return 1.0 if self.natural else MASS_UNITS[self.mass]

    @property
    def gm_si(self) -> float:
        """Conversion factor for ``gm`` values written in length³/time² [m³/s² per unit]."""
        return 1.0 if self.natural else self.length_si**3 / self.time_si**2

    @property
    def velocity_label(self) -> str:
        """Human-readable velocity unit, e.g. ``"km/s"``."""
        return self.velocity or f"{self.length}/{self.time}"


def pick_time_unit(seconds: float) -> tuple[str, float]:
    """Choose a readable unit for a time span: the largest unit that fits at least twice.

    Returns:
        ``(unit name, unit length in seconds)``.
    """
    best = ("s", 1.0)
    for name, size in TIME_UNITS.items():
        if seconds >= 2 * size:
            best = (name, size)
    return best
