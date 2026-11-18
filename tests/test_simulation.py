"""End-to-end simulations against known solutions."""

from pathlib import Path

import numpy as np
import pytest

from gravitation.config import ConfigError, load_config, parse_config
from gravitation.simulation import SimulationError, SimulationResult, load_or_simulate, simulate
from gravitation.units import AU, DAY, GM_SUN

CONFIGS = Path(__file__).parent.parent / "configs"
FIGURE_EIGHT_PERIOD = 6.32591398


def test_figure_eight_returns_after_one_period():
    config = load_config(CONFIGS / "figure_eight.yaml")
    result = simulate(config, samples=4, dt=1e-3)
    # samples 0..3 cover three periods: every snapshot is one period apart
    for s in (1, 2, 3):
        np.testing.assert_allclose(result.x[s], result.x[0], atol=1e-6)
    assert result.energy_error.max() < 1e-10


def test_earth_year():
    """A circular 1 au orbit around one solar mass takes 365.25 days (Kepler's third law)."""
    year = 2 * np.pi * np.sqrt(AU**3 / GM_SUN) / DAY
    config = parse_config(
        {
            "units": {"length": "au", "time": "day", "mass": "solar_mass", "velocity": "km/s"},
            "simulation": {"duration": year, "dt": 0.1, "integrator": "yoshida4", "samples": 2},
            "bodies": [
                {"name": "Sun", "mass": 1, "position": [0, 0], "velocity": [0, 0]},
                {
                    "name": "Earth",
                    "mass": 1e-12,
                    "position": [1, 0],
                    "velocity": [0, float(np.sqrt(GM_SUN / AU) / 1e3)],
                },
            ],
        }
    )
    result = simulate(config)
    np.testing.assert_allclose(result.x[-1, 1], result.x[0, 1], atol=1e-9 * AU)
    assert abs(year - 365.25) < 0.01


def test_barycentric_removes_drift():
    config = load_config(CONFIGS / "binary_star.yaml")
    result = simulate(config, samples=3)
    assert np.abs(result.momentum).max() < 1e-12 * np.abs(result.mu / result.G).sum() * 1e4


def test_coincident_bodies_rejected():
    with pytest.raises(ConfigError, match="start at the same position"):
        parse_config(
            {
                "units": "nbody",
                "simulation": {"duration": 1.0, "dt": 0.1},
                "bodies": [
                    {"mass": 1, "position": [0, 0], "velocity": [0, 0]},
                    {"mass": 1, "position": [0, 0], "velocity": [1, 0]},
                ],
            }
        )


def test_collision_is_reported():
    """A head-on fall with a coarse step: the bodies land exactly on each other."""
    config = parse_config(
        {
            "units": "nbody",
            "simulation": {
                "duration": 2.0,
                "dt": 0.5,
                "samples": 5,
                "integrator": "euler",
                "barycentric": False,
            },
            "bodies": [
                {"mass": 1, "position": [0, 0], "velocity": [0, 0]},
                {"mass": 1, "position": [1, 0], "velocity": [-2, 0]},
            ],
        }
    )
    with pytest.raises(SimulationError, match="non-finite"):
        simulate(config)


def test_save_load_roundtrip(tmp_path):
    config = load_config(CONFIGS / "figure_eight.yaml")
    result = simulate(config, samples=5)
    loaded = SimulationResult.load(result.save(tmp_path / "sim.npz"))
    np.testing.assert_array_equal(loaded.x, result.x)
    assert loaded.body_names == result.body_names
    assert loaded.physics_hash == result.physics_hash


def test_cache_depends_on_physics_only(tmp_path):
    path = tmp_path / "sim.npz"
    config = load_config(CONFIGS / "figure_eight.yaml")
    _, reused = load_or_simulate(config, path)
    assert not reused
    config.render = type(config.render)(duration=99.0)  # rendering change: cache still valid
    _, reused = load_or_simulate(config, path)
    assert reused
    config.bodies[0].velocity = config.bodies[0].velocity * 1.01  # physics change
    _, reused = load_or_simulate(config, path)
    assert not reused
