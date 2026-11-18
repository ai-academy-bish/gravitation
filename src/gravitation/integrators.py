r"""Hand-written time integrators for the N-body problem.

Gravity turns the N-body problem into a system of ordinary differential
equations

.. math::  \dot{\mathbf x} = \mathbf v, \qquad \dot{\mathbf v} = \mathbf a(\mathbf x),

which an integrator advances in discrete steps of size *h*. All integrators
here share one signature::

    step(x, v, a, h, accel) -> (x_new, v_new, a_new)

``a`` is the acceleration at ``x``, already computed, and ``a_new`` is
``accel(x_new)``. Handing the acceleration through lets methods such as
velocity Verlet evaluate the expensive force sum only once per step.

The methods differ in *order* (halving *h* divides the error by 2ᵒʳᵈᵉʳ) and in
whether they are *symplectic*. A symplectic method keeps the energy error
bounded and oscillating forever instead of letting it drift, which is why
celestial mechanics prefers them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

AccelFn = Callable[[np.ndarray], np.ndarray]
StepFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, float, AccelFn],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]


def euler_step(x, v, a, h, accel):
    r"""Explicit (forward) Euler.

    .. math::  \mathbf x' = \mathbf x + h\mathbf v, \qquad \mathbf v' = \mathbf v + h\mathbf a(\mathbf x)

    First order and not symplectic: on a Kepler orbit the energy grows
    steadily and the planet spirals outwards.
    """
    x_new = x + h * v
    v_new = v + h * a
    return x_new, v_new, accel(x_new)


def euler_cromer_step(x, v, a, h, accel):
    r"""Semi-implicit (symplectic) Euler, also called Euler–Cromer.

    .. math::  \mathbf v' = \mathbf v + h\mathbf a(\mathbf x), \qquad \mathbf x' = \mathbf x + h\mathbf v'

    The only change from explicit Euler is using the *new* velocity to move
    the position. It is still first order, but symplectic: the energy error
    stays bounded.
    """
    v_new = v + h * a
    x_new = x + h * v_new
    return x_new, v_new, accel(x_new)


def verlet_step(x, v, a, h, accel):
    r"""Velocity Verlet, the workhorse of molecular and celestial dynamics.

    .. math::

        \mathbf x' &= \mathbf x + h\mathbf v + \tfrac12 h^2 \mathbf a(\mathbf x) \\
        \mathbf v' &= \mathbf v + \tfrac12 h \left[\mathbf a(\mathbf x) + \mathbf a(\mathbf x')\right]

    Second order, symplectic and time-reversible, with one force evaluation
    per step. The original notebook had the position update right but
    advanced the velocity with ``a(x)`` alone, which made it first-order Euler.
    """
    x_new = x + h * v + (0.5 * h * h) * a
    a_new = accel(x_new)
    v_new = v + (0.5 * h) * (a + a_new)
    return x_new, v_new, a_new


def rk4_step(x, v, a, h, accel):
    r"""Classical fourth-order Runge–Kutta.

    Samples the derivative four times across the step and combines the
    samples with weights 1/6, 1/3, 1/3, 1/6. It is very accurate per step but
    not symplectic, so its energy error still drifts, only slowly. Four force
    evaluations per step (the last one is reused as the first of the next).
    """
    k1x, k1v = v, a
    k2x = v + 0.5 * h * k1v
    k2v = accel(x + 0.5 * h * k1x)
    k3x = v + 0.5 * h * k2v
    k3v = accel(x + 0.5 * h * k2x)
    k4x = v + h * k3v
    k4v = accel(x + h * k3x)
    x_new = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_new = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return x_new, v_new, accel(x_new)


_CBRT2 = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1 = 1.0 / (2.0 - _CBRT2)
_YOSHIDA_W0 = -_CBRT2 / (2.0 - _CBRT2)


def yoshida4_step(x, v, a, h, accel):
    r"""Yoshida's fourth-order symplectic integrator.

    Three velocity-Verlet sub-steps of sizes w₁h, w₀h, w₁h with

    .. math::  w_1 = \frac{1}{2 - 2^{1/3}}, \qquad w_0 = -\frac{2^{1/3}}{2 - 2^{1/3}}

    The middle sub-step runs *backwards* in time (w₀ < 0), and the weights
    are chosen so that the second- and third-order errors cancel. The result
    keeps Verlet's symplectic structure and is fourth order. Three force
    evaluations per step.
    """
    x, v, a = verlet_step(x, v, a, _YOSHIDA_W1 * h, accel)
    x, v, a = verlet_step(x, v, a, _YOSHIDA_W0 * h, accel)
    return verlet_step(x, v, a, _YOSHIDA_W1 * h, accel)


@dataclass(frozen=True)
class Integrator:
    """A named integration method together with its key properties.

    Attributes:
        name: identifier used in configs and on the command line.
        title: human-readable name.
        order: global order of accuracy.
        symplectic: whether the method conserves phase-space volume, so its
            energy error stays bounded.
        force_evals: force evaluations per step, a fair measure of cost.
        step: the step function, see the module docstring.
    """

    name: str
    title: str
    order: int
    symplectic: bool
    force_evals: int
    step: StepFn


INTEGRATORS: dict[str, Integrator] = {
    i.name: i
    for i in (
        Integrator("euler", "Explicit Euler", 1, False, 1, euler_step),
        Integrator("euler_cromer", "Semi-implicit Euler", 1, True, 1, euler_cromer_step),
        Integrator("verlet", "Velocity Verlet", 2, True, 1, verlet_step),
        Integrator("rk4", "Runge–Kutta 4", 4, False, 4, rk4_step),
        Integrator("yoshida4", "Yoshida 4", 4, True, 3, yoshida4_step),
    )
}
"""All available integrators, keyed by name."""


def get_integrator(name: str) -> Integrator:
    """Look up an integrator by name, with a helpful error for typos."""
    try:
        return INTEGRATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown integrator {name!r} (choose from: {', '.join(INTEGRATORS)})"
        ) from None
