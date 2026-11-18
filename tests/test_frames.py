"""Frame preparation for the renderers (no Manim needed)."""

from pathlib import Path

import numpy as np
import pytest

from gravitation.config import load_config
from gravitation.render.frames import FIT_RADIUS, FrameData, build_frames
from gravitation.simulation import simulate

CONFIGS = Path(__file__).parent.parent / "configs"


@pytest.fixture(scope="module")
def binary():
    config = load_config(CONFIGS / "binary_star.yaml")
    return config, simulate(config, samples=400)


def test_frame_count_and_shapes(binary):
    config, result = binary
    frames = build_frames(result, config, fps=30, duration=4)
    assert frames.n_frames == 121
    assert frames.positions.shape == (121, len(config.bodies), 3)
    assert len(frames.labels) == 121
    assert frames.duration == pytest.approx(4.0)


def test_auto_scale_fits_frame(binary):
    config, result = binary
    frames = build_frames(result, config, fps=30, duration=4)
    distances = np.linalg.norm(frames.track, axis=-1)
    assert np.percentile(distances, 99) == pytest.approx(FIT_RADIUS, rel=1e-3)


def test_trail_points_end_at_body(binary):
    config, result = binary
    frames = build_frames(result, config, fps=30, duration=4)
    current = frames.positions_at(50.5)
    points = frames.trail_points(0, 50.5, current[0])
    assert len(points) >= 2
    np.testing.assert_allclose(points[-1], current[0])


def test_moon_exaggeration():
    config = load_config(CONFIGS / "inner_solar_system.yaml")
    result = simulate(config, samples=50)
    frames = build_frames(result, config, fps=10, duration=2)
    earth, moon = config.index("Earth"), config.index("Moon")
    true = np.linalg.norm(result.x[0, moon] - result.x[0, earth])
    shown = np.linalg.norm(frames.positions[0, moon] - frames.positions[0, earth])
    scale = float(frames.scale_label.split("=")[1].split()[0])  # scene units per au
    assert shown == pytest.approx(40 * true / 1.495978707e11 * scale, rel=1e-2)


def test_save_load_roundtrip(binary, tmp_path):
    config, result = binary
    frames = build_frames(result, config, fps=30, duration=2)
    loaded = FrameData.load(frames.save(tmp_path / "frames.npz"))
    np.testing.assert_allclose(loaded.track, frames.track)
    assert loaded.labels == frames.labels
    assert loaded.glow == frames.glow
