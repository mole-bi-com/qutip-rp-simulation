"""
rp_sim — Radical Pair Spin Dynamics Simulation Package
=======================================================
QuTiP 5-based comprehensive simulation suite for spin biology.

Layers:
  1. Hamiltonian construction (Zeeman, Hyperfine, Exchange, Dipolar)
  2. Lindblad master equation solver (singlet/triplet yield)
  3. Decoherence channel analysis
  4. Anisotropy / Cryptochrome orientation model
  5. Publication-quality visualization

Reference: Simulating spin biology using a digital quantum computer
           APL Quantum (2024) arXiv:2406.12986
"""

from .hamiltonian import (
    zeeman_hamiltonian,
    hyperfine_hamiltonian,
    exchange_hamiltonian,
    dipolar_hamiltonian,
    total_hamiltonian,
    singlet_state,
    triplet_states,
    spin_operators,
)
from .solver import (
    rp_solve,
    singlet_yield,
    triplet_yield,
    field_sweep,
    time_trace,
    mfe,
    singlet_projector,
    triplet_projector,
)
from .decoherence import (
    relaxation_ops,
    dephasing_ops,
    recombination_ops,
    decoherence_channels,
)
from .anisotropy import (
    anisotropic_field_sweep,
    cryptochrome_model,
    orientation_averaged_yield,
)
from .visualization import (
    plot_mfe_curve,
    plot_anisotropy_map,
    plot_time_evolution,
    plot_decoherence_comparison,
    make_publication_figure,
)

__version__ = "1.0.0"
