r"""Newtonian gravity: accelerations, conserved quantities and reference frames.

Newton's law of universal gravitation gives the acceleration of body *i* in a
system of *N* point masses:

.. math::

    \mathbf a_i = \sum_{j \ne i} \mu_j
        \frac{\mathbf x_j - \mathbf x_i}{\left(|\mathbf x_j - \mathbf x_i|^2 + \varepsilon^2\right)^{3/2}},
    \qquad \mu_j = G m_j

where ε is an optional *softening length* that caps the force at very short
range (ε = 0 is exact Newtonian gravity).

Every function accepts arrays with optional leading "batch" dimensions, so
``x`` may be one state of shape ``(N, 3)`` or a whole trajectory of shape
``(S, N, 3)``. Masses enter only through ``mu`` (shape ``(N,)``); physical
masses, where needed, are recovered as ``m = mu / G``.
"""

from __future__ import annotations

import numpy as np

_CHUNK = 1024  # samples per batch when evaluating energies along a trajectory


def accelerations(x: np.ndarray, mu: np.ndarray, eps2: float = 0.0) -> np.ndarray:
    """Gravitational acceleration of every body.

    The pairwise separations ``x_j - x_i`` are formed explicitly rather than
    via ``|x_i|² + |x_j|² - 2 x_i·x_j``: that shortcut loses about five
    significant digits for the Earth–Moon pair when positions are ~1 au from
    the origin.

    Args:
        x: positions, shape ``(..., N, 3)``.
        mu: gravitational parameters ``G·m``, shape ``(N,)``.
        eps2: softening length squared, ε².

    Returns:
        Accelerations with the same shape as ``x``.
    """
    d = x[..., np.newaxis, :, :] - x[..., :, np.newaxis, :]  # d[..., i, j] = x_j - x_i
    r2 = np.einsum("...k,...k->...", d, d) + eps2
    n = x.shape[-2]
    r2[..., np.arange(n), np.arange(n)] = np.inf  # a body does not attract itself
    weights = mu * r2**-1.5  # μ_j / r_ij³, broadcast over the j axis
    return np.einsum("...ij,...ijk->...ik", weights, d)


def kinetic_energy(v: np.ndarray, mu: np.ndarray, G: float) -> np.ndarray:
    r"""Total kinetic energy :math:`T = \sum_i \tfrac12 m_i |\mathbf v_i|^2`."""
    return 0.5 * np.einsum("i,...i->...", mu / G, np.einsum("...k,...k->...", v, v))


def potential_energy(
    x: np.ndarray,
    mu: np.ndarray,
    G: float,
    eps2: float = 0.0,
) -> np.ndarray:
    r"""Total potential energy :math:`U = -\sum_{i<j} G m_i m_j / \sqrt{r_{ij}^2 + \varepsilon^2}`."""
    i, j = np.triu_indices(x.shape[-2], k=1)
    d = x[..., j, :] - x[..., i, :]
    r = np.sqrt(np.einsum("...k,...k->...", d, d) + eps2)
    return -np.sum(mu[i] * mu[j] / r, axis=-1) / G


def total_energy(
    x: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
    G: float,
    eps2: float = 0.0,
) -> np.ndarray:
    """Total mechanical energy ``T + U``; constant in exact Newtonian dynamics.

    Long trajectories are processed in chunks to keep memory bounded.
    """
    if x.ndim == 2:
        return kinetic_energy(v, mu, G) + potential_energy(x, mu, G, eps2)
    out = np.empty(x.shape[0])
    for start in range(0, x.shape[0], _CHUNK):
        sl = slice(start, start + _CHUNK)
        out[sl] = kinetic_energy(v[sl], mu, G) + potential_energy(x[sl], mu, G, eps2)
    return out


def linear_momentum(v: np.ndarray, mu: np.ndarray, G: float) -> np.ndarray:
    r"""Total linear momentum :math:`\mathbf P = \sum_i m_i \mathbf v_i`."""
    return np.einsum("i,...ik->...k", mu / G, v)


def angular_momentum(
    x: np.ndarray,
    v: np.ndarray,
    mu: np.ndarray,
    G: float,
) -> np.ndarray:
    r"""Total angular momentum about the origin :math:`\mathbf L = \sum_i m_i\, \mathbf x_i \times \mathbf v_i`."""
    return np.einsum("i,...ik->...k", mu / G, np.cross(x, v))


def center_of_mass(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    r"""Centre of mass :math:`\mathbf R = \sum_i m_i \mathbf x_i / \sum_i m_i`, shape ``(..., 3)``."""
    return np.einsum("i,...ik->...k", mu, x) / mu.sum()


def to_barycentric(x: np.ndarray, v: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift a state into the centre-of-mass frame.

    In that frame the total momentum is zero, so the system as a whole does
    not drift across the screen.
    """
    return x - center_of_mass(x, mu), v - center_of_mass(v, mu)
