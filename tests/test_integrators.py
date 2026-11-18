"""Integrators: convergence order, energy behaviour and conservation laws."""

from functools import partial

import numpy as np
import pytest

from gravitation.integrators import INTEGRATORS, get_integrator
from gravitation.physics import accelerations, angular_momentum, linear_momentum, total_energy


def circular_orbit(method: str, steps_per_orbit: int, orbits: int = 1):
    """A massless test particle on a unit circular orbit around μ = 1; returns final state."""
    mu = np.array([1.0, 0.0])
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    v = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    accel = partial(accelerations, mu=mu)
    step = get_integrator(method).step
    h = 2 * np.pi / steps_per_orbit
    a = accel(x)
    for _ in range(steps_per_orbit * orbits):
        x, v, a = step(x, v, a, h, accel)
    return x, v


@pytest.mark.parametrize("name", list(INTEGRATORS))
def test_convergence_order(name):
    """Halving the step divides the error after one orbit by about 2^order."""
    method = INTEGRATORS[name]
    start = np.array([1.0, 0.0, 0.0])
    errors = [np.linalg.norm(circular_orbit(name, n)[0][1] - start) for n in (400, 800)]
    measured = np.log2(errors[0] / errors[1])
    # Symplectic Euler is conjugate to a second-order method, so on a closed
    # orbit it can look better than its nominal order; never worse.
    assert measured > method.order - 0.3, f"{name}: measured order {measured:.2f}"


@pytest.mark.parametrize("name", ["euler_cromer", "verlet", "yoshida4"])
def test_symplectic_energy_stays_bounded(name):
    """Over many orbits a symplectic method's energy error does not grow."""
    mu = np.array([1.0, 0.0])
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    v0 = np.array([[0.0, 0.0, 0.0], [0.0, 1.2, 0.0]])  # eccentric orbit
    accel = partial(accelerations, mu=mu)
    step = get_integrator(name).step
    h = 2 * np.pi / 200
    x, v, a = x0, v0, accel(x0)
    e0 = total_energy(x0, v0, np.array([1.0, 1e-12]), 1.0)
    errors = []
    for _ in range(20):
        for _ in range(400):
            x, v, a = step(x, v, a, h, accel)
        errors.append(abs(total_energy(x, v, np.array([1.0, 1e-12]), 1.0) - e0))
    assert max(errors[10:]) < 3 * max(errors[:10]) + 1e-15


def test_explicit_euler_energy_drifts():
    """Explicit Euler pumps energy into a Kepler orbit: the radius grows."""
    x, _ = circular_orbit("euler", 200, orbits=5)
    assert np.linalg.norm(x[1]) > 1.5


@pytest.mark.parametrize("name", list(INTEGRATORS))
def test_momentum_conserved(name):
    """Pairwise forces cancel, so every method conserves total momentum exactly.

    Angular momentum is exact only for the symplectic drift/kick methods: a kick
    with central forces exerts no torque and a drift moves along v. Explicit
    Euler adds h²·Σ m v×a per step, and RK4's mixed stages are not exact either.
    """
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.5, 2.0, 5)
    x = rng.normal(size=(5, 3))
    v = rng.normal(scale=0.3, size=(5, 3))
    accel = partial(accelerations, mu=mu, eps2=0.01)
    step = get_integrator(name).step
    p0, l0 = linear_momentum(v, mu, 1.0), angular_momentum(x, v, mu, 1.0)
    a = accel(x)
    for _ in range(200):
        x, v, a = step(x, v, a, 1e-3, accel)
    np.testing.assert_allclose(linear_momentum(v, mu, 1.0), p0, atol=1e-12)
    if INTEGRATORS[name].symplectic:
        np.testing.assert_allclose(angular_momentum(x, v, mu, 1.0), l0, atol=1e-11)


def test_unknown_integrator():
    with pytest.raises(ValueError, match="unknown integrator"):
        get_integrator("leapfrogg")
