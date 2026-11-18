"""Running a simulation and storing its result.

The integrator takes many small steps of size ≤ ``dt`` but stores only
``samples`` evenly spaced snapshots. A video needs about a thousand frames
while an accurate orbit may need a million steps, and keeping every step
(as the original notebook did) wastes memory without improving the picture.

The step is shrunk slightly so that a whole number of steps fits between
snapshots, ``h = Δt_sample / ceil(Δt_sample / dt)``, so snapshots land exactly
on their nominal times.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gravitation.config import SystemConfig
from gravitation.integrators import get_integrator
from gravitation.physics import (
    accelerations,
    angular_momentum,
    linear_momentum,
    to_barycentric,
    total_energy,
)


class SimulationError(RuntimeError):
    """The integration produced non-finite numbers (usually a near-collision)."""


@dataclass(eq=False)
class SimulationResult:
    """Snapshots of a finished simulation, in internal units (SI or N-body).

    Attributes:
        name: name of the simulated system.
        body_names: body names, in config order.
        t: snapshot times, shape ``(S,)``.
        x: positions, shape ``(S, N, 3)``.
        v: velocities, shape ``(S, N, 3)``.
        mu: gravitational parameters, shape ``(N,)``.
        G: gravitational constant in internal units.
        softening: softening length ε.
        integrator: name of the integrator used.
        dt: actual step size used.
        steps: total number of integration steps.
        wall_time: seconds spent integrating.
        physics_hash: :meth:`SystemConfig.physics_hash` of the source config.
    """

    name: str
    body_names: list[str]
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    mu: np.ndarray
    G: float
    softening: float
    integrator: str
    dt: float
    steps: int
    wall_time: float
    physics_hash: str

    @property
    def energy(self) -> np.ndarray:
        """Total energy at every snapshot, shape ``(S,)``."""
        return total_energy(self.x, self.v, self.mu, self.G, self.softening**2)

    @property
    def energy_error(self) -> np.ndarray:
        """Relative energy error |E(t) − E(0)| / |E(0)| at every snapshot.

        For an exact solution this is zero; how it grows over time is the
        simplest health check of an integrator.
        """
        energy = self.energy
        return np.abs((energy - energy[0]) / energy[0])

    @property
    def momentum(self) -> np.ndarray:
        """Total linear momentum at every snapshot, shape ``(S, 3)``."""
        return linear_momentum(self.v, self.mu, self.G)

    @property
    def angular_momentum(self) -> np.ndarray:
        """Total angular momentum at every snapshot, shape ``(S, 3)``."""
        return angular_momentum(self.x, self.v, self.mu, self.G)

    def save(self, path: str | Path) -> Path:
        """Write the result to a compressed ``.npz`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "body_names": self.body_names,
            "G": self.G,
            "softening": self.softening,
            "integrator": self.integrator,
            "dt": self.dt,
            "steps": self.steps,
            "wall_time": self.wall_time,
            "physics_hash": self.physics_hash,
        }
        np.savez_compressed(path, t=self.t, x=self.x, v=self.v, mu=self.mu, meta=json.dumps(meta))
        return path

    @classmethod
    def load(cls, path: str | Path) -> SimulationResult:
        """Read a result written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls(t=data["t"], x=data["x"], v=data["v"], mu=data["mu"], **meta)


def simulate(
    config: SystemConfig,
    *,
    integrator: str | None = None,
    dt: float | None = None,
    samples: int | None = None,
    progress: bool | None = None,
) -> SimulationResult:
    """Integrate a system and return its snapshots.

    Args:
        config: the validated system.
        integrator: override ``simulation.integrator``.
        dt: override the maximum step size (internal time units).
        samples: override the number of stored snapshots.
        progress: show a progress bar (default: only when stderr is a terminal).

    Raises:
        SimulationError: if positions become non-finite.
    """
    sim = config.simulation
    method = get_integrator(integrator or sim.integrator)
    n_samples = samples or sim.samples
    max_dt = dt or sim.dt

    x, v = config.initial_state()
    mu = config.mu
    if sim.barycentric:
        x, v = to_barycentric(x, v, mu)

    sample_dt = sim.duration / (n_samples - 1)
    substeps = max(1, math.ceil(sample_dt / max_dt - 1e-9))
    h = sample_dt / substeps

    accel = partial(accelerations, mu=mu, eps2=sim.softening**2)
    step = method.step

    xs = np.empty((n_samples, *x.shape))
    vs = np.empty_like(xs)
    xs[0], vs[0] = x, v
    a = accel(x)

    if progress is None:
        progress = sys.stderr.isatty()
    bar = tqdm(
        total=(n_samples - 1) * substeps,
        disable=not progress,
        unit="step",
        unit_scale=True,
        desc=f"{config.name} [{method.name}]",
        leave=False,
    )
    start = time.perf_counter()
    # Overflow or 0/0 near a collision is caught below as non-finite positions
    # with a clear message, so NumPy's own warnings would only be noise.
    with bar, np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for s in range(1, n_samples):
            for _ in range(substeps):
                x, v, a = step(x, v, a, h, accel)
            if not np.isfinite(x).all():
                raise SimulationError(
                    f"positions became non-finite at t = {s * sample_dt:.6g} (internal units); "
                    "two bodies probably collided — try a smaller 'dt' or a 'softening' length"
                )
            xs[s], vs[s] = x, v
            bar.update(substeps)
    wall_time = time.perf_counter() - start

    return SimulationResult(
        name=config.name,
        body_names=config.names,
        t=np.linspace(0.0, sim.duration, n_samples),
        x=xs,
        v=vs,
        mu=mu,
        G=config.units.G,
        softening=sim.softening,
        integrator=method.name,
        dt=h,
        steps=(n_samples - 1) * substeps,
        wall_time=wall_time,
        physics_hash=config.physics_hash(),
    )


def load_or_simulate(
    config: SystemConfig, path: str | Path, *, force: bool = False
) -> tuple[SimulationResult, bool]:
    """Reuse a saved simulation if it matches the config's physics, otherwise run and save one.

    Returns:
        ``(result, reused)`` where ``reused`` tells whether the cache was hit.
    """
    path = Path(path)
    if path.exists() and not force:
        try:
            cached = SimulationResult.load(path)
        except (OSError, ValueError, KeyError, TypeError):
            cached = None
        if cached is not None and cached.physics_hash == config.physics_hash():
            return cached, True
    result = simulate(config)
    result.save(path)
    return result, False
