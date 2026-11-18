"""Gravity, energies and reference frames."""

import numpy as np

from gravitation.physics import (
    accelerations,
    center_of_mass,
    kinetic_energy,
    linear_momentum,
    potential_energy,
    to_barycentric,
    total_energy,
)


def test_two_body_acceleration_matches_newton():
    mu = np.array([3.0, 5.0])
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    a = accelerations(x, mu)
    np.testing.assert_allclose(a[0], [5.0 / 4.0, 0, 0])  # pulled towards body 1
    np.testing.assert_allclose(a[1], [-3.0 / 4.0, 0, 0])


def test_newtons_third_law():
    """Σ m_i a_i = 0 for any configuration."""
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.1, 3.0, 7)
    x = rng.normal(size=(7, 3))
    a = accelerations(x, mu)
    np.testing.assert_allclose(mu @ a, 0.0, atol=1e-12)


def test_batched_accelerations_match_single():
    rng = np.random.default_rng(2)
    mu = rng.uniform(0.1, 3.0, 4)
    xs = rng.normal(size=(6, 4, 3))
    batched = accelerations(xs, mu)
    for s in range(6):
        np.testing.assert_allclose(batched[s], accelerations(xs[s], mu))


def test_precision_for_close_pair_far_from_origin():
    """Earth–Moon-like pair at 1 au: separations must not lose digits."""
    au = 1.495978707e11
    x = np.array([[au, 0, 0], [au + 3.844e8, 0, 0]])
    mu = np.array([3.986e14, 4.903e12])
    a = accelerations(x, mu)
    np.testing.assert_allclose(a[1, 0], -3.986e14 / 3.844e8**2, rtol=1e-12)


def test_softening_caps_force():
    mu = np.array([1.0, 1.0])
    x = np.array([[0.0, 0, 0], [1e-9, 0, 0]])
    assert np.all(np.isfinite(accelerations(x, mu, eps2=0.01)))
    assert np.abs(accelerations(x, mu, eps2=0.01)).max() < 1e-6


def test_energy_of_circular_orbit():
    """For a circular orbit of a test particle, E = -GMm / 2r (virial theorem)."""
    G = 2.0
    mu = np.array([G * 10.0, G * 1e-6])
    x = np.array([[0, 0, 0], [4.0, 0, 0]])
    v = np.array([[0, 0, 0], [0, np.sqrt(G * 10.0 / 4.0), 0]])
    expected = -G * 10.0 * 1e-6 / (2 * 4.0)
    np.testing.assert_allclose(total_energy(x, v, mu, G), expected, rtol=1e-6)
    assert kinetic_energy(v, mu, G) > 0 > potential_energy(x, mu, G)


def test_barycentric_frame():
    rng = np.random.default_rng(3)
    mu = rng.uniform(0.1, 3.0, 5)
    x, v = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
    xb, vb = to_barycentric(x, v, mu)
    np.testing.assert_allclose(center_of_mass(xb, mu), 0.0, atol=1e-14)
    np.testing.assert_allclose(linear_momentum(vb, mu, 1.0), 0.0, atol=1e-13)
