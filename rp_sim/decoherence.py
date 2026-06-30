"""
Decoherence Analysis for Radical Pair Spin Dynamics
====================================================

Decoherence channels relevant to radical pair systems:

1. Spin relaxation (T₁): energy relaxation, spin-lattice
   L = 1/√(T₁) · S_{±}
   
2. Spin dephasing (T₂): phase randomization, spin-spin
   L = 1/√(2·T₂) · S_z

3. Recombination (k_S, k_T): population loss to product states
   L = √k_S · P_S, L = √k_T · P_T

4. Random field (environmental noise): effective decoherence
   from many weakly coupled nuclear spins treated as bath

In cryptochrome radical pairs:
  - T₁ ~ 1-10 μs (protein environment dependent)
  - T₂ ~ 100 ns - 1 μs
  - k_S ~ 0.001-0.1 ns⁻¹ (singlet recombination)
  - k_T ~ 0.0001-0.01 ns⁻¹ (triplet recombination, typically slower)
"""

import numpy as np
import qutip as qt
from .hamiltonian import _count_nuclei
from .hamiltonian import spin_operators


def relaxation_ops(T1, dims=None, target='both'):
    """Construct spin relaxation (T₁) Lindblad operators.
    
    L_± = 1/√(T₁) · S_±
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
    """Construct spin dephasing (T₂ → pure dephasing) Lindblad operators.
    
    L_z = 1/√(2·T₂) · S_z
    
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
    
    L_S = √k_S · P_S
    L_T = √k_T · P_T
    
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
        T₁ relaxation time [ns].
    T2 : float, optional
        T₂ dephasing time [ns].
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
        channels.append(f"T₁={T1:.0f}ns")
    
    if T2 is not None and T2 > 0:
        c_ops.extend(dephasing_ops(T2, dims, target))
        channels.append(f"T₂={T2:.0f}ns")
    
    if k_S > 0:
        c_ops.extend(recombination_ops(k_S, k_T, dims))
        channels.append(f"k_S={k_S:.4f}")
    
    label = "+".join(channels) if channels else "unitary"
    return c_ops, label


def decoherence_sweep(H, rho0, k_S, k_T=None,
                      T1_values=None, T2_values=None,
                      t_max=10000.0):
    """Sweep over decoherence rates and compute singlet yield.
    
    Parameters
    ----------
    H : Qobj
        System Hamiltonian.
    rho0 : Qobj
        Initial density matrix.
    k_S : float
        Singlet recombination rate.
    k_T : float, optional
        Triplet recombination rate.
    T1_values : array_like, optional
        T₁ values to scan.
    T2_values : array_like, optional
        T₂ values to scan.
    t_max : float
        Integration time [ns].
    
    Returns
    -------
    results : dict
        Yield vs decoherence time.
    """
    from .solver import singlet_yield
    
    results = {}
    
    if T1_values is not None:
        yields_T1 = []
        for T1 in T1_values:
            c_ops = relaxation_ops(T1, rho0.dims)
            if k_S > 0:
                c_ops.extend(recombination_ops(k_S, k_T, rho0.dims))
            phi_S, _ = singlet_yield(H, rho0, k_S, k_T, t_max=t_max)
            yields_T1.append(phi_S)
        results['T1'] = (T1_values, yields_T1)
    
    if T2_values is not None:
        yields_T2 = []
        for T2 in T2_values:
            c_ops = dephasing_ops(T2, rho0.dims)
            if k_S > 0:
                c_ops.extend(recombination_ops(k_S, k_T, rho0.dims))
            phi_S, _ = singlet_yield(H, rho0, k_S, k_T, t_max=t_max)
            yields_T2.append(phi_S)
        results['T2'] = (T2_values, yields_T2)
    
    return results
