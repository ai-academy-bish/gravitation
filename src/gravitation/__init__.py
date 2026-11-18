"""gravitation — a small, readable Newtonian N-body simulator made for teaching.

The package has two layers:

* a physics core (``units``, ``physics``, ``integrators``, ``simulation``) that
  depends only on NumPy and can be read top to bottom like a textbook chapter;
* renderers (``render``) that turn a finished simulation into a picture or a
  video with matplotlib, Manim Community Edition or 3Blue1Brown's ManimGL.

Systems of bodies are described in YAML files (see ``configs/``) and loaded with
:func:`gravitation.config.load_config`.
"""

from gravitation.config import SystemConfig, load_config
from gravitation.simulation import SimulationResult, simulate

__all__ = ["SimulationResult", "SystemConfig", "load_config", "simulate"]
__version__ = "0.1.0"
