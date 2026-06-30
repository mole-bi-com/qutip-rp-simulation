"""
RP Solver — Non-trace-preserving Liouvillian Integration
========================================================

The radical pair recombination is described by:

    dρ/dt = -i[H, ρ] - k_S/2 {P_S, ρ} - k_T/2 {P_T, ρ}

This is NOT a standard Lindblad master equation (which preserves trace).
Instead, the anti-commutator terms remove population from the system,
modeling irreversible recombination into product states.

The Liouvillian superoperator is constructed as:
    L = -i(H⊗I - I⊗H^T) - k_S/2 (P_S⊗I + I⊗P_S^T) - k_T/2 (P_T⊗I + I⊗P_T^T)

Integration uses scipy's solve_ivp (RK45) for numerical stability.

Yield:
    Φ_S = k_S ∫₀^∞ Tr[P_S ρ(t)] dt
    Φ_T = k_T ∫₀^∞ Tr[P_T ρ(t)] dt

Reference: Timmel et al., Chem. Phys. Lett. (1998)
"""

import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp
from .hamiltonian import total_hamiltonian, singlet_state, _count_nuclei
from .hamiltonian import singlet_state as _singlet_dm


def singlet_projector(dims=None):
    """Construct the singlet projection operator P_S = |S⟩⟨S|."""
    return _singlet_dm(dims)


def triplet_projector(dims=None):
    """Construct the triplet projection operator P_T = P_T+ + P_T0 + P_T-."""
    from .hamiltonian import triplet_states
    states = triplet_states(dims)
    return states[0] + states[1] + states[2]


def _build_liouvillian(H, k_S, k_T=0.0, T1=None, T2=None, dims=None):
    """Build the non-trace-preserving Liouvillian superoperator.
    
    L[ρ] = -i[H, ρ] - k_S/2{P_S, ρ} - k_T/2{P_T, ρ}
    
    For T1/T2 relaxation, additional Lindblad terms are added
    (these ARE trace-preserving).
    
    Parameters
    ----------
    H : Qobj
        System Hamiltonian.
    k_S : float
        Singlet recombination rate [1/ns].
    k_T : float
        Triplet recombination rate [1/ns].
    T1 : float, optional
        T₁ relaxation time [ns].
    T2 : float, optional
        T₂ dephasing time [ns].
    dims : list or tuple, optional
        System dimensions for constructing relaxation operators.
    
    Returns
    -------
    L : Qobj
        Liouvillian superoperator (as a matrix).
    """
    # Unitary part: -i[H, ρ]
    L = -1j * (qt.spre(H) - qt.spost(H))
    
    # Recombination (non-trace-preserving): -k/2 {P, ρ}
    if k_S > 0:
        P_S = singlet_projector(dims)
        L += -k_S/2 * (qt.spre(P_S) + qt.spost(P_S))
    
    if k_T > 0:
        P_T = triplet_projector(dims)
        L += -k_T/2 * (qt.spre(P_T) + qt.spost(P_T))
    
    # Relaxation (trace-preserving Lindblad terms)
    if T1 is not None and T1 > 0:
        from .decoherence import relaxation_ops
        for c_op in relaxation_ops(T1, dims):
            L += qt.lindblad_dissipator(c_op)
    
    if T2 is not None and T2 > 0:
        from .decoherence import dephasing_ops
        for c_op in dephasing_ops(T2, dims):
            L += qt.lindblad_dissipator(c_op)
    
    return L


def rp_solve(H, rho0, k_S, k_T=0.0, t_max=10000.0, n_steps=5000,
             T1=None, T2=None, e_ops=None,
             method='RK45', atol=1e-10, rtol=1e-8):
    """Solve the RP master equation using scipy's ODE solver.
    
    Parameters
    ----------
    H : Qobj
        System Hamiltonian.
    rho0 : Qobj
        Initial density matrix.
    k_S : float
        Singlet recombination rate [1/ns].
    k_T : float
        Triplet recombination rate [1/ns].
    t_max : float
        Maximum integration time [ns].
    n_steps : int
        Number of output time points.
    T1, T2 : float, optional
        Relaxation/dephasing times [ns].
    e_ops : list of Qobj, optional
        Expectation value operators.
    method : str
        ODE solver method (default: 'RK45').
    atol, rtol : float
        ODE solver tolerances.
    
    Returns
    -------
    result : object
        Object with .times, .expect attributes (QuTiP-compatible interface).
    """
    times = np.linspace(0, t_max, n_steps)
    dims = rho0.dims
    
    # Build Liouvillian
    L = _build_liouvillian(H, k_S, k_T, T1, T2, dims)
    L_mat = L.full()
    
    # Vectorize initial state: ρ₀ → column vector
    rho_vec = rho0.full().reshape(-1)
    dim = rho_vec.size
    
    dt = t_max / n_steps
    output_times = np.linspace(0, t_max, n_steps)
    rho_vecs = np.zeros((dim, n_steps), dtype=complex)
    rho_vecs[:, 0] = rho_vec
    
    if dim < 1024:
        # Small system: dense matrix exponential (fast)
        from scipy.linalg import expm
        U = expm(L_mat * dt)
        for i in range(1, n_steps):
            rho_vecs[:, i] = U @ rho_vecs[:, i-1]
        actual_n_steps = n_steps
    else:
        # Large system: just use dense expm (Liouvillian dim still manageable)
        from scipy.linalg import expm
        U = expm(L_mat * dt)
        for i in range(1, n_steps):
            rho_vecs[:, i] = U @ rho_vecs[:, i-1]
        actual_n_steps = n_steps
    
    # Build result object (QuTiP-compatible interface)
    class Result:
        pass
    
    result = Result()
    result.times = output_times
    result.expect = []
    result.states = None
    
    # Compute expectation values at each time point
    orig_shape = tuple(rho0.shape)
    
    if e_ops is not None:
        for e_op in e_ops:
            e_op_mat = e_op.full()
            expect_vals = np.zeros(actual_n_steps, dtype=float)
            for i in range(actual_n_steps):
                rho_mat = rho_vecs[:, i].reshape(orig_shape)
                expect_vals[i] = np.real(np.trace(e_op_mat @ rho_mat))
            result.expect.append(expect_vals)
    
    return result


def singlet_yield(H, rho0, k_S, k_T=0.0, T1=None, T2=None,
                  t_max=2000.0, n_steps=500):
    """Compute the singlet recombination yield.
    
    Φ_S = k_S ∫₀^∞ Tr[P_S ρ(t)] dt
    
    Parameters
    ----------
    H : Qobj
        System Hamiltonian.
    rho0 : Qobj
        Initial state (|S⟩⟨S|).
    k_S : float
        Singlet recombination rate [1/ns].
    k_T : float
        Triplet recombination rate [1/ns].
    T1, T2 : float, optional
        Relaxation/dephasing times [ns].
    t_max, n_steps : float, int
        Integration parameters.
    
    Returns
    -------
    phi_S : float
        Singlet recombination yield (0 ≤ Φ_S ≤ 1 typically).
    result : object
        Solver result with .times, .expect for analysis.
    """
    dims = rho0.dims
    P_S = singlet_projector(dims)
    e_ops = [P_S]
    
    result = rp_solve(H, rho0, k_S, k_T, t_max, n_steps,
                      T1=T1, T2=T2, e_ops=e_ops)
    
    phi_S = k_S * np.trapezoid(result.expect[0], result.times)
    
    return phi_S, result


def triplet_yield(H, rho0, k_S, k_T, t_max=10000.0, n_steps=5000):
    """Compute the triplet recombination yield.
    
    Φ_T = k_T ∫₀^∞ Tr[P_T ρ(t)] dt
    """
    dims = rho0.dims
    P_T = triplet_projector(dims)
    P_S = singlet_projector(dims)
    
    result = rp_solve(H, rho0, k_S, k_T, t_max, n_steps,
                      e_ops=[P_S, P_T])
    
    phi_T = k_T * np.trapezoid(result.expect[1], result.times)
    return phi_T, result


def field_sweep(B_range, k_S, k_T=0.0, g_A=2.0023, g_B=2.0023,
                hfc_list_A=None, hfc_list_B=None, J=0.0, d=0.0,
                T1=None, T2=None, t_max=2000.0, n_steps=500,
                return_results=False, progress=True):
    """Compute singlet yield as a function of magnetic field strength.
    
    Parameters
    ----------
    B_range : array_like
        Magnetic field strengths [mT].
    k_S : float
        Singlet recombination rate [1/ns].
    k_T : float
        Triplet recombination rate [1/ns].
    g_A, g_B : float
        Electron g-factors.
    hfc_list_A, hfc_list_B : list
        Hyperfine coupling tensors.
    J, d : float
        Exchange and dipolar coupling [μeV].
    T1, T2 : float, optional
        Relaxation/dephasing times [ns].
    return_results : bool
        Return full result objects.
    progress : bool
        Show progress.
    
    Returns
    -------
    yields : ndarray
        Singlet yields for each B value.
    """
    if hfc_list_A is None: hfc_list_A = []
    if hfc_list_B is None: hfc_list_B = []
    
    B_range = np.atleast_1d(B_range)
    yields = np.zeros(len(B_range))
    results_list = []
    
    n_nuc = len(hfc_list_A) + len(hfc_list_B)
    
    for i, B in enumerate(B_range):
        H = total_hamiltonian(
            B=[0, 0, B], g_A=g_A, g_B=g_B,
            hfc_list_A=hfc_list_A, hfc_list_B=hfc_list_B,
            J=J, d=d
        )
        
        dims = ([qt.identity(2)] * 2 + [qt.identity(2)] * n_nuc) if n_nuc > 0 else None
        rho0 = singlet_state(dims)
        
        phi_S, result = singlet_yield(H, rho0, k_S, k_T,
                                      T1=T1, T2=T2, t_max=t_max, n_steps=n_steps)
        yields[i] = phi_S
        
        if return_results:
            results_list.append(result)
        
        if progress:
            pass  # progress handled by caller
    
    if return_results:
        return yields, results_list
    return yields


def time_trace(H, rho0, k_S, k_T=0.0, t_max=1000.0, n_steps=2000,
               T1=None, T2=None):
    """Compute the time-dependent singlet and triplet populations."""
    dims = rho0.dims
    P_S = singlet_projector(dims)
    P_T = triplet_projector(dims)
    
    result = rp_solve(H, rho0, k_S, k_T, t_max, n_steps,
                      T1=T1, T2=T2, e_ops=[P_S, P_T])
    
    return result


def mfe(yields_B, yields_0):
    """Compute the magnetic field effect magnitude.
    
    MFE(B) = (Φ_S(B) − Φ_S(0)) / Φ_S(0)
    """
    return (np.array(yields_B) - yields_0) / yields_0
