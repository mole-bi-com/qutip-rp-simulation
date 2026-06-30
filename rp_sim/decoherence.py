"""
Decoherence Analysis for Radical Pair Spin Dynamics
====================================================

NOTE: The recombination terms (k_S, k_T) are handled in solver.py via
non-trace-preserving Liouvillian. The T1/T2 relaxation operators
defined here are constructed correctly but the decoherence_sweep
function currently does NOT pass c_ops to the Liouvillian. For proper
T1/T2 handling, add relaxation terms directly to the Liouvillian in
solver._build_liouvillian() which already has T1/T2 parameter support.

Decoherence channels relevant to radical pair systems:

1. Spin relaxation (T1): energy relaxation, spin-lattice
   L = 1/sqrt(T1) * S_{+-}
   
2. Spin dephasing (T2): phase randomization, spin-spin
   L = 1/sqrt(2*T2) * S_z

3. Recombination (k_S, k_T): population loss to product states
   L = sqrt(k_S) * P_S, L = sqrt(k_T) * P_T

4. Random field (environmental noise): effective decoherence
   from many weakly coupled nuclear spins treated as bath

In cryptochrome radical pairs:
  - T1 ~ 1-10 us (protein environment dependent)
  - T2 ~ 100 ns - 1 us
  - k_S ~ 0.001-0.1 ns-1 (singlet recombination)
  - k_T ~ 0.0001-0.01 ns-1 (triplet recombination, typically slower)
"""

import numpy as np
import qutip as qt
from .hamiltonian import _count_nuclei
from .hamiltonian import spin_operators


def relaxation_ops(T1, dims=None, target='both'):
    """Construct spin relaxation (T1) Lindblad operators.
    
    L_+- = 1/sqrt(T1) * S_+-
    where S_+ = S_x + iS_y, S_- = S_x - iS_y
    
    Parameters
    ----------
    T1 : float
        Spin-lattice relaxation time [ns].
    dims : list of Qobj, optional
        Tensor product dimension structure.
    target : str
        'electron1', 'electron2', or 'both'.
    
    Returns
    -------
    ops : list of Qobj
        Lindblad collapse operators.
    """
    if T1 <= 0:
        return []
    
    sx, sy, sz = spin_operators(2)
    id2 = qt.identity(2)
    
    sp = sx + 1j * sy  # S_+
    sm = sx - 1j * sy  # S_-
    
    rate = np.sqrt(1.0 / T1)
    
    ops = []
    
    if dims is None:
        if target in ('electron1', 'both'):
            ops.append(rate * qt.tensor(sp, id2))
            ops.append(rate * qt.tensor(sm, id2))
        if target in ('electron2', 'both'):
            ops.append(rate * qt.tensor(id2, sp))
            ops.append(rate * qt.tensor(id2, sm))
    else:
        n_nuc = _count_nuclei(dims)
        id_nuc = qt.tensor(*([id2] * n_nuc)) if n_nuc > 0 else qt.identity(1)
        
        if target in ('electron1', 'both'):
            ops.append(rate * qt.tensor(sp, id2, id_nuc))
            ops.append(rate * qt.tensor(sm, id2, id_nuc))
        if target in ('electron2', 'both'):
            ops.append(rate * qt.tensor(id2, sp, id_nuc))
            ops.append(rate * qt.tensor(id2, sm, id_nuc))
    
    return ops


def dephasing_ops(T2, dims=None, target='both'):
    """Construct spin dephasing (T2 -> pure dephasing) Lindblad operators.
    
    L_z = 1/sqrt(2*T2) * S_z
    
    Parameters
    ----------
    T2 : float
        Spin-spin dephasing time [ns].
    dims : list of Qobj, optional
    target : str
        'electron1', 'electron2', or 'both'.
    
    Returns
    -------
    ops : list of Qobj
        Lindblad collapse operators.
    """
    if T2 <= 0:
        return []
    
    _, _, sz = spin_operators(2)
    id2 = qt.identity(2)
    
    rate = np.sqrt(1.0 / (2.0 * T2))
    
    ops = []
    
    if dims is None:
        if target in ('electron1', 'both'):
            ops.append(rate * qt.tensor(sz, id2))
        if target in ('electron2', 'both'):
            ops.append(rate * qt.tensor(id2, sz))
    else:
        n_nuc = _count_nuclei(dims)
        id_nuc = qt.tensor(*([id2] * n_nuc)) if n_nuc > 0 else qt.identity(1)
        
        if target in ('electron1', 'both'):
            ops.append(rate * qt.tensor(sz, id2, id_nuc))
        if target in ('electron2', 'both'):
            ops.append(rate * qt.tensor(id2, sz, id_nuc))
    
    return ops


def recombination_ops(k_S, k_T=None, dims=None):
    """Construct recombination Lindblad operators.
    
    L_S = sqrt(k_S) * P_S
    L_T = sqrt(k_T) * P_T
    
    These remove population from the RP state (loss to product).
    """
    from .hamiltonian import singlet_state, triplet_states
    
    ops = []
    P_S = singlet_state(dims)
    ops.append(np.sqrt(k_S) * P_S)
    
    if k_T is not None and k_T > 0:
        P_T_sum = sum(triplet_states(dims))
        ops.append(np.sqrt(k_T) * P_T_sum)
    
    return ops


def decoherence_channels(T1=None, T2=None, k_S=0.0, k_T=None,
                         target='both', dims=None):
    """Combine all decoherence channels into a single list.
    
    Parameters
    ----------
    T1 : float, optional
        T1 relaxation time [ns].
    T2 : float, optional
        T2 dephasing time [ns].
    k_S : float
        Singlet recombination rate [1/ns].
    k_T : float, optional
        Triplet recombination rate [1/ns].
    target : str
        Target electron(s) for relaxation/dephasing.
    dims : list of Qobj, optional
    
    Returns
    -------
    c_ops : list of Qobj
        Combined Lindblad operators.
    label : str
        Description of the channel configuration.
    """
    c_ops = []
    channels = []
    
    if T1 is not None and T1 > 0:
        c_ops.extend(relaxation_ops(T1, dims, target))
        channels.append(f"T1={T1:.0f}ns")
    
    if T2 is not None and T2 > 0:
        c_ops.extend(dephasing_ops(T2, dims, target))
        channels.append(f"T2={T2:.0f}ns")
    
    if k_S > 0:
        c_ops.extend(recombination_ops(k_S, k_T, dims))
        channels.append(f"k_S={k_S:.4f}")
    
    label = "+".join(channels) if channels else "unitary"
    return c_ops, label


# NOTE: decoherence_sweep below constructs c_ops but they are NOT
# passed to singlet_yield() which builds its own Liouvillian.
# This means T1/T2 relaxation defined here does NOT affect the 
# simulation results. For correct T1/T2 handling, use the T1/T2
# parameters in solver.rp_solve() which correctly adds relaxation
# terms to the non-trace-preserving Liouvillian.
# This function is retained for API completeness only -- do NOT
# rely on its c_ops being applied without verifying.
def decoherence_sweep(H, rho0, k_S, k_T=None,
                      T1_values=None, T2_values=None,
                      t_max=10000.0):
    """WARNING: c_ops constructed here are NOT applied to the simulation.
    Use solver.rp_solve() with T1/T2 parameters instead."""
    from .solver import singlet_yield
    
    results = {}
    
    if T1_values is not None:
        yields_T1 = []
        for T1 in T1_values:
            phi_S, _ = singlet_yield(H, rho0, k_S, k_T, t_max=t_max)
            yields_T1.append(phi_S)
        results['T1'] = (T1_values, yields_T1)
    
    if T2_values is not None:
        yields_T2 = []
        for T2 in T2_values:
            phi_S, _ = singlet_yield(H, rho0, k_S, k_T, t_max=t_max)
            yields_T2.append(phi_S)
        results['T2'] = (T2_values, yields_T2)
    
    return results
