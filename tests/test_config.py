"""Config parsing: units, expressions, colours and helpful errors."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from gravitation.colors import resolve_color
from gravitation.config import ConfigError, evaluate_number, load_config, parse_config
from gravitation.ephemeris import CATALOG, StateVector, build_config_yaml, julian_date
from gravitation.units import AU, DAY, G_SI, GM_SUN, UnitSystem

CONFIGS = Path(__file__).parent.parent / "configs"


def minimal(**overrides):
    data = {
        "units": "nbody",
        "simulation": {"duration": 1, "dt": 0.1},
        "bodies": [{"name": "A", "mass": 1, "position": [0, 0, 0], "velocity": [0, 0, 0]}],
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_are_valid(path):
    config = load_config(path)
    assert config.bodies


def test_units_conversion():
    config = parse_config(
        {
            "units": {"length": "AU", "time": "days", "mass": "Msun", "velocity": "km/s"},
            "simulation": {"duration": 10, "dt": 1},
            "bodies": [{"mass": 2, "position": [1, 0], "velocity": [0, 30]}],
        }
    )
    body = config.bodies[0]
    assert body.position[0] == pytest.approx(AU)
    assert body.velocity[1] == pytest.approx(30e3)
    assert body.gm == pytest.approx(2 * GM_SUN)
    assert config.simulation.duration == pytest.approx(10 * DAY)
    assert UnitSystem().G == G_SI and UnitSystem.nbody().G == 1.0


def test_gm_in_config_units():
    config = parse_config(
        {
            "units": {"length": "au", "time": "day"},
            "simulation": {"duration": 1, "dt": 1},
            "bodies": [{"gm": 2.959122082841195e-4, "position": [0, 0], "velocity": [0, 0]}],
        }
    )
    assert config.bodies[0].gm == pytest.approx(GM_SUN, rel=1e-12)


def test_expressions():
    assert evaluate_number("107 * 10**6 / 3600") == pytest.approx(29722.2222)
    assert evaluate_number("5.972e24") == 5.972e24  # PyYAML would read this as a string
    assert evaluate_number("2 * pi") == pytest.approx(2 * np.pi)
    for bad in ("__import__('os')", "open('x')", "9**9**9", "1/0", "x + 1"):
        with pytest.raises(ValueError):
            evaluate_number(bad)


def test_typo_gets_suggestion():
    with pytest.raises(
        ConfigError, match=r"simulation\.integrater: unknown key — did you mean 'integrator'"
    ):
        parse_config(minimal(simulation={"duration": 1, "dt": 0.1, "integrater": "verlet"}))
    with pytest.raises(ConfigError, match="did you mean 'yoshida4'"):
        parse_config(minimal(simulation={"duration": 1, "dt": 0.1, "integrator": "yoshida"}))


def test_mass_xor_gm():
    with pytest.raises(ConfigError, match="exactly one of 'mass' or 'gm'"):
        parse_config(minimal(bodies=[{"mass": 1, "gm": 1, "position": [0, 0], "velocity": [0, 0]}]))


def test_unquoted_hex_colour_hint():
    with pytest.raises(ConfigError, match="must be quoted"):
        parse_config(
            minimal(bodies=[{"mass": 1, "position": [0, 0], "velocity": [0, 0], "color": None}])
        )


def test_exaggeration_must_reference_another_body():
    body = {
        "name": "Moon",
        "mass": 1,
        "position": [0, 0],
        "velocity": [0, 0],
        "exaggerate": {"around": "Earth", "factor": 10},
    }
    with pytest.raises(ConfigError, match="must be the name of another body"):
        parse_config(minimal(bodies=[body]))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("orange", "#ffa500"),
        ("Dark Orange", "#ff8c00"),
        ("dark_orange", "#ff8c00"),
        ("#F80", "#ff8800"),
        ("#ff8800", "#ff8800"),
        ("ff8800", "#ff8800"),
        ([255, 136, 0], "#ff8800"),
        ([1.0, 0.0, 0.0], "#ff0000"),
        ("tab:blue", "#1f77b4"),
    ],
)
def test_colours(value, expected):
    assert resolve_color(value) == expected


def test_colour_suggestion():
    with pytest.raises(ValueError, match="did you mean 'green'"):
        resolve_color("grren")


def test_ephemeris_yaml_roundtrip(tmp_path):
    """A generated config (no network: fake state vectors) loads and keeps its numbers."""
    states = [
        StateVector(CATALOG["sun"], np.array([0.001, 0.0, 0.0]), np.array([0.0, 1e-6, 0.0])),
        StateVector(CATALOG["earth"], np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0172, 0.0])),
        StateVector(CATALOG["moon"], np.array([1.00257, 0.0, 0.0]), np.array([0.0, 0.0178, 0.0])),
    ]
    text = build_config_yaml(
        states, epoch=date(2026, 1, 1), name="Test", duration_days=10, dt_days=0.1
    )
    path = tmp_path / "jpl.yaml"
    path.write_text(text)
    config = load_config(path)
    assert config.names == ["Sun", "Earth", "Moon"]
    assert config.bodies[1].position[0] == pytest.approx(AU)
    assert config.bodies[0].gm == pytest.approx(GM_SUN, rel=1e-12)
    assert config.bodies[2].exaggerate is not None


def test_julian_date():
    assert julian_date(date(2000, 1, 1)) == 2451544.5
