# gravitation

A small, readable **Newtonian N-body simulator** made for teaching: hand-written
integrators in plain NumPy, systems described in YAML (or fetched from NASA JPL),
and videos rendered with [Manim Community Edition](https://www.manim.community/)
or 3Blue1Brown's own [ManimGL](https://github.com/3b1b/manim).

- **Physics core you can read top to bottom**: `units → physics → integrators → simulation`, about 750 lines including docstrings with the maths.
- **Five integrators to compare**: explicit Euler, semi-implicit Euler, velocity Verlet, RK4 and Yoshida-4, with energy, momentum and angular-momentum diagnostics.
- **Systems in YAML**: units of your choice (`au`, `km/s`, `solar_mass`, … or N-body units), arithmetic expressions, CSS colour names or hex codes, and strict validation with "did you mean …?" hints.
- **Real initial conditions** from NASA JPL Horizons for any date: `make ephemeris DATE=1969-07-20`.
- **Three rendering backends** behind one frame-preparation layer, plus a benchmark that compares them on your machine.

---

## Quick start

```bash
make setup      # system packages, uv, .venv and all Python dependencies
make render     # inner Solar System → output/inner_solar_system/*.mp4
make help       # every target and variable
```

`make setup` works on **macOS** (Homebrew) and **Linux** (apt, dnf, pacman,
zypper). It installs only what is missing and asks before installing anything
(`make setup YES=1` to skip the prompts). On Windows, use WSL2.

| What | Why |
|---|---|
| C compiler, `pkg-config`, `cairo` | Manim CE builds `pycairo` from source; it has no binary wheels |
| `pango` (Linux) | `manimpango` has no Linux wheels and is built against it |
| `ffmpeg` | ManimGL pipes frames to the `ffmpeg` binary (Manim CE uses the bundled PyAV) |
| Vulkan loader + Mesa drivers (Linux) | ManimGL renders through wgpu; Mesa's *lavapipe* also works without a GPU |
| `libGL`, Xvfb (headless Linux) | only for Manim CE's OpenGL renderer (`BACKEND=ce-gl`) |
| [uv](https://docs.astral.sh/uv/) | creates `.venv` with its own CPython 3.12 and installs the locked dependencies |

LaTeX is **not** needed: the scenes typeset text with Pango.

## Everyday use

```bash
make list                                         # available configs
make simulate CONFIG=configs/figure_eight.yaml    # integrate and print diagnostics
make preview  CONFIG=configs/black_hole_flyby.yaml  # PNG: orbits + energy error
make compare                                      # all integrators, same step, one chart
make render   CONFIG=configs/binary_star.yaml BACKEND=gl QUALITY=h
make bench    QUALITY=m REPEAT=2                  # time every backend on this machine
make ephemeris PRESET=full DATE=2030-01-01        # real Solar System → configs/jpl/
make colors                                       # named colours with swatches
```

| Variable | Default | Meaning |
|---|---|---|
| `CONFIG` | `configs/inner_solar_system.yaml` | system to simulate |
| `BACKEND` | `ce` | `ce` (Manim CE, Cairo) · `ce-gl` (Manim CE, OpenGL) · `gl` (ManimGL, wgpu) |
| `QUALITY` | `m` | `l` 480p30 · `m` 720p30 · `h` 1080p60 · `k` 2160p60 |
| `DURATION` | from config | video length in seconds |
| `PRESET`, `DATE`, `BODIES` | `inner`, today | what `make ephemeris` fetches |
| `FORCE=1` | | re-run the simulation even if a cached one matches |

Everything is also available directly as `uv run gravitation <command> --help`.

---

## The physics

**Newton's law of universal gravitation.** Body $j$ pulls body $i$ with the force

$$
\mathbf F_{ij} = G\,\frac{m_i m_j}{|\mathbf r_{ij}|^2}\,\hat{\mathbf r}_{ij}
             = G\,m_i m_j\,\frac{\mathbf x_j - \mathbf x_i}{|\mathbf x_j - \mathbf x_i|^3},
\qquad G = 6.674\,30 \times 10^{-11}\ \mathrm{m^3\,kg^{-1}\,s^{-2}} .
$$

**Equations of motion.** Dividing by $m_i$ and summing over all other bodies
gives a system of $6N$ first-order ordinary differential equations:

$$
\dot{\mathbf x}_i = \mathbf v_i, \qquad
\dot{\mathbf v}_i = \mathbf a_i = \sum_{j \ne i} \mu_j\,
  \frac{\mathbf x_j - \mathbf x_i}{\left(|\mathbf x_j - \mathbf x_i|^2 + \varepsilon^2\right)^{3/2}},
\qquad \mu_j = G m_j .
$$

The code works with the **gravitational parameter** $\mu = Gm$ rather than $G$ and $m$
separately: for the Sun, $\mu$ is known to about 11 significant digits, $G$ to only 5.
The optional **softening length** $\varepsilon$ keeps the force finite in head-on
encounters ($\varepsilon = 0$ is exact Newtonian gravity).

**Conserved quantities.** These are exact invariants of the true motion, so their drift
measures the numerical error:

$$
E = \sum_i \tfrac12 m_i |\mathbf v_i|^2 - \sum_{i<j} \frac{G m_i m_j}{\sqrt{r_{ij}^2+\varepsilon^2}},
\qquad
\mathbf P = \sum_i m_i \mathbf v_i,
\qquad
\mathbf L = \sum_i m_i\, \mathbf x_i \times \mathbf v_i .
$$

Simulations start in the **centre-of-mass frame** ($\mathbf P = 0$) unless
`barycentric: false`, so the system does not drift across the screen.

## Numerical integration

With step $h$ and $\mathbf a_n = \mathbf a(\mathbf x_n)$:

| Method | Update | Order | Symplectic | Forces / step |
|---|---|:-:|:-:|:-:|
| Explicit Euler | $\mathbf x_{n+1} = \mathbf x_n + h\mathbf v_n,\ \ \mathbf v_{n+1} = \mathbf v_n + h\mathbf a_n$ | 1 | no | 1 |
| Semi-implicit Euler | $\mathbf v_{n+1} = \mathbf v_n + h\mathbf a_n,\ \ \mathbf x_{n+1} = \mathbf x_n + h\mathbf v_{n+1}$ | 1 | yes | 1 |
| Velocity Verlet | $\mathbf x_{n+1} = \mathbf x_n + h\mathbf v_n + \tfrac12 h^2\mathbf a_n,\ \ \mathbf v_{n+1} = \mathbf v_n + \tfrac12 h(\mathbf a_n + \mathbf a_{n+1})$ | 2 | yes | 1 |
| Runge–Kutta 4 | four derivative samples per step, weights $\tfrac16, \tfrac13, \tfrac13, \tfrac16$ | 4 | no | 4 |
| Yoshida 4 | three Verlet sub-steps $w_1h,\ w_0h,\ w_1h$ with $w_1 = \frac{1}{2-2^{1/3}},\ w_0 = -\frac{2^{1/3}}{2-2^{1/3}}$ | 4 | yes | 3 |

*Order* $p$ means that halving $h$ divides the error by $2^p$. *Symplectic*
methods preserve the geometry of Hamiltonian motion, so their energy error
oscillates but stays **bounded** for ever, while non-symplectic ones drift.
`make compare` shows this on the inner Solar System (2 years, the same 38 400 steps for every method):

| Method | max \|ΔE/E\| | final position error |
|---|---:|---:|
| Explicit Euler | 2.1 × 10⁻² | 3.3 × 10⁻¹ |
| Semi-implicit Euler | 2.7 × 10⁻⁵ | 1.3 × 10⁻³ |
| Velocity Verlet | 3.1 × 10⁻⁸ | 7.4 × 10⁻⁶ |
| Runge–Kutta 4 | 2.7 × 10⁻¹⁴ | 8.7 × 10⁻¹² |
| Yoshida 4 | 1.6 × 10⁻¹³ | 1.4 × 10⁻¹⁰ |

**Step size and snapshots.** The integrator takes many small steps but stores
only `samples` evenly spaced snapshots (by default 240 per second of video).
The step is shrunk slightly so that a whole number of steps fits between
snapshots: $h = \Delta t_\text{sample} / \lceil \Delta t_\text{sample} / dt \rceil$.

## Describing a system

```yaml
name: Binary Star
units: {length: au, time: day, mass: solar_mass, velocity: km/s}   # or: units: nbody
simulation: {duration: 2200, dt: 0.05, integrator: yoshida4}
render: {duration: 20, trail: 6, camera: {phi: 62, theta: -35, rotation: 3}}
bodies:
  - name: Star A
    mass: 1.0
    position: [-0.5 * 0.8 / 1.8, 0, 0]      # arithmetic is allowed
    velocity: [0, -56.51 * 0.8 / 1.8, 0]
    color: gold                             # CSS name, "#ffaa00", "#fa0" or [255, 170, 0]
    glow: true
```

[`configs/binary_star.yaml`](configs/binary_star.yaml) lists and explains **every**
option; copy it to start your own system. Included systems:

| Config | What it shows |
|---|---|
| `inner_solar_system.yaml` | Sun, Mercury–Mars and the Moon from JPL Horizons, 1 Jan 2026 |
| `outer_solar_system.yaml` | Sun and the giant planets over one orbit of Neptune (164 years) |
| `figure_eight.yaml` | Chenciner–Montgomery three-body choreography, a strict integrator test |
| `binary_star.yaml` | binary star, circumbinary planet and an inclined comet: the reference config |
| `black_hole_flyby.yaml` | the original notebook's "Gargantua" scenario, in SI units |
| `four_bodies.yaml` | the original notebook's first example, with a sane time step |

**Visual-only options** do not touch the physics: `radius` (real radii would be
invisible at orbital scale), `glow`, and `exaggerate: {around: Earth, factor: 40}`,
which draws the Moon 40× further from the Earth than it is. At 0.0026 au it
would otherwise sit inside the Earth's sphere.

---

## Rendering

```
YAML ──► simulate (NumPy) ──► simulation.npz ──► build_frames ──► frames_<q>.npz ──► Manim scene ──► .mp4
            cached by a hash          resampled onto exact frame times,           one ValueTracker,
            of the physics            centred, scaled, HUD labels                  one play() call
```

The original notebook called `self.play(...)` once per stored sample, about
11 700 times for the Solar System. Each call deep-copied the 3-D spheres and
wrote its own partial movie file. Here a single `ValueTracker` runs from the
first to the last frame in **one** `play`, and one updater moves every body and
redraws every trail from precomputed positions. Manim writes one file with
exactly `fps × duration` frames.

| Backend | Engine | Notes |
|---|---|---|
| `ce` | Manim CE · Cairo (CPU) | Bodies are shaded discs that always face the camera. Profiling showed per-face shading of 3-D spheres took 90% of the frame time (a 288-face sphere is 288 vector objects for Cairo); at 10–40 px a disc looks the same. |
| `ce-gl` | Manim CE · OpenGL | Real spheres; CE's OpenGL renderer is still experimental. |
| `gl` | ManimGL · wgpu (GPU) | 3Blue1Brown's ManimGL; since August 2026 it renders through **wgpu** (Metal on macOS, Vulkan on Linux), off-screen with no window or X server. Glowing halos and per-point trail fading run on the GPU. Pinned to a git commit because the last PyPI release (1.7.2, 2024) no longer installs cleanly. |

### Benchmark

Inner Solar System, 20 s of video (601 / 1201 frames), measured on an Apple M4
MacBook Air (macOS 27, wgpu on Metal). Wall time per render, warm caches:

| Backend | 720p · 30 fps | 1080p · 60 fps | Speed at 1080p |
|---|---:|---:|---:|
| Manim CE · Cairo (CPU) | 7.7 s | 17.3 s | 69 frames/s |
| Manim CE · OpenGL | 10.2 s | 19.8 s | 61 frames/s |
| **ManimGL · wgpu (GPU)** | **5.8 s** | **12.3 s** | **98 frames/s** |

All three render faster than real time. Where the speed came from, measured
along the way:

| Change | Effect |
|---|---|
| One `play()` for the whole video instead of one per sample | The notebook's approach costs ~78 ms per frame even at 480p15, about 15 min for its Solar System run |
| Cairo: shaded billboard discs instead of 288-face spheres | 6.6 → 45 frames/s at 480p |
| CE OpenGL: one GPU mesh (`OpenGLSurface`) instead of CE's `Sphere`, whose faces are re-triangulated in Python after every move | 107 s → 10 s at 720p |
| ManimGL on wgpu | fastest backend; the very first render on a machine takes several seconds longer while its caches fill |

`make bench` writes the same table, plus the machine description (CPU, the
GPU adapter wgpu picked, versions) to `output/bench/`. Run it on your Linux
machines and compare the JSON files.

---

## Project layout

```
configs/                 YAML systems (configs/jpl/ for `make ephemeris` output)
scripts/setup.sh         OS detection and system packages, uv, .venv
src/gravitation/
  units.py               units, constants, GM values
  config.py              YAML loading and validation, safe arithmetic
  physics.py             accelerations, energy, momenta, barycentre
  integrators.py         Euler, semi-implicit Euler, Verlet, RK4, Yoshida 4
  simulation.py          integration loop, snapshots, .npz cache
  ephemeris.py           NASA JPL Horizons client
  render/frames.py       backend-agnostic frame preparation
  render/manim_ce.py     Manim CE scene (Cairo and OpenGL)
  render/manim_gl.py     ManimGL scene
  render/runner.py       launches Manim, quality presets
  render/bench.py        benchmark report
  render/preview.py      matplotlib previews
  cli.py                 `gravitation` command
tests/                   pytest: convergence orders, conservation laws, Kepler, configs, frames
```

## Development

```bash
make test       # 59 tests, about 5 s
make lint       # ruff
make format
```

## What changed from the 2024 notebook

- **Integrator.** The notebook updated the velocity with $\mathbf a_n$ only, which is explicit
  Euler (first order, energy drifts). Velocity Verlet uses
  $\tfrac12(\mathbf a_n + \mathbf a_{n+1})$ at the same cost and is 10⁵× more accurate here.
- **Time step.** `dt = 100 s` was hard-coded for every system. In the first example
  a body orbits in ~105 s, i.e. one step per orbit. The step now lives in the config.
- **Speed.** Python loops over bodies became vectorised NumPy: 105 µs → 7 µs per step.
- **Rendering.** One `play()` instead of thousands; coordinates are scaled to the
  frame, where the notebook passed metres straight to Manim and everything landed
  off-screen; colours no longer raise `KeyError`.
- **Frames.** Centre-of-mass frame by default, real JPL initial conditions,
  equal-aspect 3-D plots, the initial state is recorded, and bodies keep their names.

## Roadmap

- `.docs/`: a textbook-style guide to the physics, the algorithms and how each is implemented.
