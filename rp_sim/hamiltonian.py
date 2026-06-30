"""
Hamiltonian Construction for Radical Pair Spin Dynamics
========================================================

Constructs the full spin Hamiltonian for an electron radical pair
with an arbitrary number of nuclear spins:

    H = H_Z + H_HF + H_Ex + H_D

  - H_Z = μ_B B·(g₁·S₁ + g₂·S₂) / ℏ        [Zeeman]
  - H_HF = Σ S₁·A_k·I_k + Σ S₂·A_j·I_j    [Hyperfine]
  - H_Ex = -J (2 S₁·S₂ + ½)                [Exchange]
  - H_D = d (3(S₁·r̂)(S₂·r̂) − S₁·S₂)       [Dipolar]

Units: All energies in μeV (micro-electronvolt) — standard for radical pair.
       1 μeV ≈ 2π × 0.242 GHz ≈ 8.066 cm⁻¹

References:
  - Timmel et al., Chem. Phys. Lett. (1998) — RP mechanism
  - Hore, Proc. Nat. Acad. Sci. (2007) — bird magnetoreception
  - Kattnig et al., J. Chem. Phys. (2016) — anisotropic RP model
"""

import numpy as np
import qutip as qt

# ─── Fundamental constants (in μeV units, except where noted) ──────────────
MU_B = 5.7883818012e-5   # Bohr magneton [eV/T] → convert to μeV
MU_B_MUEV = MU_B * 1e6   # [μeV/T]
HBAR_OVER_MUEV = 6.582119569e-1  # ℏ in [μeV·ns] (so H/ℏ has units of 1/ns)


def spin_operators(hilbert_dim=2):
    """Return spin operators {S_x, S_y, S_z} for a given Hilbert space dimension.
    
    For dim=2: standard Pauli matrices (S = σ/2).
    For dim>2: generalized spin operators (not needed for spin-1/2 RP, 
    but included for extensibility).
    """
    if hilbert_dim == 2:
        # Pauli matrices for spin-1/2
        sx = 0.5 * qt.sigmax()
        sy = 0.5 * qt.sigmay()
        sz = 0.5 * qt.sigmaz()
        return sx, sy, sz
    else:
        return qt.jmat((hilbert_dim - 1) / 2, 'x'), \
               qt.jmat((hilbert_dim - 1) / 2, 'y'), \
               qt.jmat((hilbert_dim - 1) / 2, 'z')


def zeeman_hamiltonian(B=[0, 0, 1], g_A=2.0023, g_B=2.0023,
                       dims=None):
    """Construct the Zeeman Hamiltonian.
    
    H_Z = μ_B B·(g₁·S₁ + g₂·S₂) / ℏ
    
    Parameters
    ----------
    B : array_like (3,)
        Magnetic field vector [mT]. Default: [0, 0, 1] (1 mT along z).
    g_A, g_B : float
        Electron g-factors. Free electron: 2.0023.
        Common values: FADH• 2.004, Z• 2.002.
    dims : list of Qobj
        List of subsystem operators [S₁, S₂, I₁, I₂, ...].
        If None, assumes no nuclei (2-spin system).
    
    Returns
    -------
    H_Z : Qobj
        Zeeman Hamiltonian in μeV.
    """
    B = np.asarray(B, dtype=float)
    B_norm = np.linalg.norm(B)
    if B_norm < 1e-12:
        return qt.Qobj(np.zeros((4, 4)), dims=[[2, 2], [2, 2]]) if dims is None else qt.Qobj(np.zeros((2**len(dims), 2**len(dims))))
    
    B_hat = B / B_norm  # unit vector
    
    if dims is None:
        # Simple 2-spin system (no nuclei)
        s1x, s1y, s1z = spin_operators(2)
        s2x, s2y, s2z = spin_operators(2)
        
        H_Z = MU_B_MUEV * B_norm * (
            g_A * (B_hat[0] * qt.tensor(s1x, qt.identity(2)) +
                   B_hat[1] * qt.tensor(s1y, qt.identity(2)) +
                   B_hat[2] * qt.tensor(s1z, qt.identity(2))) +
            g_B * (B_hat[0] * qt.tensor(qt.identity(2), s2x) +
                   B_hat[1] * qt.tensor(qt.identity(2), s2y) +
                   B_hat[2] * qt.tensor(qt.identity(2), s2z))
        )
    else:
        # General system with nuclei
        s1x, s1y, s1z = spin_operators(2)
        s2x, s2y, s2z = spin_operators(2)
        id_op = [qt.identity(d.shape[0]) for d in dims[2:]]
        
        # Proper tensor product construction
        op_list_z1 = [g_A * s1z] + [qt.identity(2)] + id_op
        op_list_z2 = [qt.identity(2)] + [g_B * s2z] + id_op
        op_list_x1 = [s1x] + [qt.identity(2)] + id_op
        op_list_x2 = [qt.identity(2)] + [s2x] + id_op
        op_list_y1 = [s1y] + [qt.identity(2)] + id_op
        op_list_y2 = [qt.identity(2)] + [s2y] + id_op
        
        H_Z = MU_B_MUEV * B_norm * (
            g_A * B_hat[0] * qt.tensor(*op_list_x1) +
            g_A * B_hat[1] * qt.tensor(*op_list_y1) +
            g_A * B_hat[2] * qt.tensor(*op_list_z1) +
            g_B * B_hat[0] * qt.tensor(*op_list_x2) +
            g_B * B_hat[1] * qt.tensor(*op_list_y2) +
            g_B * B_hat[2] * qt.tensor(*op_list_z2)
        )
    
    return H_Z


def hyperfine_hamiltonian(hfc_list_A, hfc_list_B=None, dims=None):
    """Construct the hyperfine Hamiltonian.
    
    H_HF = Σ_k S₁·A_k·I_k + Σ_j S₂·A_j·I_j
    
    For each nucleus, A_k is a 3×3 hyperfine tensor.
    For isotropic hyperfine, A = a_iso × I₃ (diagonal with equal values).
    
    Parameters
    ----------
    hfc_list_A : list of 3×3 array_like
        Hyperfine coupling tensors for nuclei coupled to electron 1 (FADH•).
        Each element: [[Axx, Axy, Axz], [Ayx, Ayy, Ayz], [Azx, Azy, Azz]] [μeV].
    hfc_list_B : list of 3×3 array_like, optional
        Same for electron 2.
    dims : list of Qobj, optional
        Pre-constructed tensor product dimension list.
    
    Returns
    -------
    H_HF : Qobj
        Hyperfine Hamiltonian in μeV.
    
    Notes
    -----
    FADH• radical has ~5 dominant N nuclei (N5, N10) with a_iso ~ 10-60 μeV.
    Z• (tryptophan) has ~3-4 dominant H/N nuclei with a_iso ~ 1-30 μeV.
    """
    if hfc_list_B is None:
        hfc_list_B = []
    
    n_nuc_A = len(hfc_list_A)
    n_nuc_B = len(hfc_list_B)
    n_nuc_total = n_nuc_A + n_nuc_B
    
    if n_nuc_total == 0:
        # No nuclei → zero Hamiltonian
        if dims is not None:
            id_list = [qt.identity(2)] * len(dims)
            return 0 * qt.tensor(*id_list)
        else:
            return qt.Qobj(np.zeros((4, 4)), dims=[[2, 2], [2, 2]])
    
    # Ensure all HFC tensors are 3×3 arrays
    hfc_A = [np.asarray(h, dtype=float).reshape(3, 3) for h in hfc_list_A]
    hfc_B = [np.asarray(h, dtype=float).reshape(3, 3) for h in hfc_list_B]
    
    # Electron and nuclear spin operators
    sx, sy, sz = spin_operators(2)
    s_vec = [sx, sy, sz]
    
    # Build nuclear spin operators
    nuc_sx, nuc_sy, nuc_sz = [], [], []
    for _ in range(n_nuc_total):
        nuc_sx.append(spin_operators(2)[0])
        nuc_sy.append(spin_operators(2)[1])
        nuc_sz.append(spin_operators(2)[2])
    
    # Construct operator tensor products
    # Layout: [e1, e2, nuc_1, nuc_2, ..., nuc_N]
    # nuc_1..nuc_A → coupled to e1, nuc_{A+1}..nuc_{A+B} → coupled to e2
    
    H_HF = 0 * qt.tensor(*[qt.identity(2)] * (2 + n_nuc_total))
    
    # Build identity ops
    for n_idx in range(n_nuc_A):
        tensor_list = [s_vec[0], qt.identity(2)]  # e1·x, e2·I
        for j in range(n_nuc_total):
            if j == n_idx:
                tensor_list.append(nuc_sx[j])
            else:
                tensor_list.append(qt.identity(2))
        e1x_Ix = qt.tensor(*tensor_list)
        
        tensor_list = [s_vec[1], qt.identity(2)]
        for j in range(n_nuc_total):
            if j == n_idx:
                tensor_list.append(nuc_sy[j])
            else:
                tensor_list.append(qt.identity(2))
        e1y_Iy = qt.tensor(*tensor_list)
        
        tensor_list = [s_vec[2], qt.identity(2)]
        for j in range(n_nuc_total):
            if j == n_idx:
                tensor_list.append(nuc_sz[j])
            else:
                tensor_list.append(qt.identity(2))
        e1z_Iz = qt.tensor(*tensor_list)
        
        # Include anisotropic components
        H_HF += (hfc_A[n_idx][0, 0] * e1x_Ix +
                 hfc_A[n_idx][1, 1] * e1y_Iy +
                 hfc_A[n_idx][2, 2] * e1z_Iz)
        
        # Off-diagonal (cross) terms — important for anisotropic HF
        if abs(hfc_A[n_idx][0, 1]) > 0 or abs(hfc_A[n_idx][1, 0]) > 0:
            tensor_list = [s_vec[0], qt.identity(2)]
            for j in range(n_nuc_total):
                if j == n_idx:
                    tensor_list.append(nuc_sy[j])
                else:
                    tensor_list.append(qt.identity(2))
            H_HF += 0.5 * (hfc_A[n_idx][0, 1] + hfc_A[n_idx][1, 0]) * qt.tensor(*tensor_list)
        
        if abs(hfc_A[n_idx][0, 2]) > 0 or abs(hfc_A[n_idx][2, 0]) > 0:
            tensor_list = [s_vec[0], qt.identity(2)]
            for j in range(n_nuc_total):
                if j == n_idx:
                    tensor_list.append(nuc_sz[j])
                else:
                    tensor_list.append(qt.identity(2))
            H_HF += 0.5 * (hfc_A[n_idx][0, 2] + hfc_A[n_idx][2, 0]) * qt.tensor(*tensor_list)
    
    # Couple e2 with its nuclei
    for n_idx in range(n_nuc_B):
        actual_idx = n_nuc_A + n_idx
        tensor_list = [qt.identity(2), s_vec[0]]
        for j in range(n_nuc_total):
            if j == actual_idx:
                tensor_list.append(nuc_sx[j])
            else:
                tensor_list.append(qt.identity(2))
        e2x_Ix = qt.tensor(*tensor_list)
        
        tensor_list = [qt.identity(2), s_vec[1]]
        for j in range(n_nuc_total):
            if j == actual_idx:
                tensor_list.append(nuc_sy[j])
            else:
                tensor_list.append(qt.identity(2))
        e2y_Iy = qt.tensor(*tensor_list)
        
        tensor_list = [qt.identity(2), s_vec[2]]
        for j in range(n_nuc_total):
            if j == actual_idx:
                tensor_list.append(nuc_sz[j])
            else:
                tensor_list.append(qt.identity(2))
        e2z_Iz = qt.tensor(*tensor_list)
        
        H_HF += (hfc_B[n_idx][0, 0] * e2x_Ix +
                 hfc_B[n_idx][1, 1] * e2y_Iy +
                 hfc_B[n_idx][2, 2] * e2z_Iz)
    
    return H_HF


def exchange_hamiltonian(J=0.0, dims=None):
    """Construct the exchange coupling Hamiltonian.
    
    H_Ex = -J (2 S₁·S₂ + ½)
    
    For spin-1/2:
    2 S₁·S₂ + ½ = 2(S_xS_x + S_yS_y + S_zS_z) + ½
    
    When J > 0: singlet lower (antiferromagnetic coupling in RP context).
    When J = 0: no exchange coupling.
    
    Typical values for RP in cryptochrome: |J| < 1 μeV (very weak).
    
    Parameters
    ----------
    J : float
        Exchange coupling constant [μeV]. Default: 0 (no exchange).
    dims : list of Qobj, optional
        Pre-constructed dimension list.
    
    Returns
    -------
    H_Ex : Qobj
        Exchange Hamiltonian in μeV.
    """
    if abs(J) < 1e-12:
        if dims is None:
            return qt.Qobj(np.zeros((4, 4)), dims=[[2, 2], [2, 2]])
        else:
            n_nuc = len(dims) - 2
            id_list = [qt.identity(2), qt.identity(2)]
            if n_nuc > 0:
                id_list += [qt.identity(2)] * n_nuc
            return 0 * qt.tensor(*id_list)
    
    if dims is None:
        # Simple 2-spin
        sx, sy, sz = spin_operators(2)
        s1x = qt.tensor(sx, qt.identity(2))
        s1y = qt.tensor(sy, qt.identity(2))
        s1z = qt.tensor(sz, qt.identity(2))
        s2x = qt.tensor(qt.identity(2), sx)
        s2y = qt.tensor(qt.identity(2), sy)
        s2z = qt.tensor(qt.identity(2), sz)
        
        S1_dot_S2 = s1x * s2x + s1y * s2y + s1z * s2z
        H_Ex = -J * (2 * S1_dot_S2 + 0.5 * qt.tensor(qt.identity(2), qt.identity(2)))
    else:
        # Full system
        s1x, s1y, s1z = spin_operators(2)
        s2x, s2y, s2z = spin_operators(2)
        n_nuc = len(dims) - 2
        id_nuc = qt.tensor(*([qt.identity(2)] * n_nuc)) if n_nuc > 0 else qt.identity(1)
        
        S1x = qt.tensor(s1x, qt.identity(2), id_nuc)
        S1y = qt.tensor(s1y, qt.identity(2), id_nuc)
        S1z = qt.tensor(s1z, qt.identity(2), id_nuc)
        S2x = qt.tensor(qt.identity(2), s2x, id_nuc)
        S2y = qt.tensor(qt.identity(2), s2y, id_nuc)
        S2z = qt.tensor(qt.identity(2), s2z, id_nuc)
        
        S1_dot_S2 = S1x * S2x + S1y * S2y + S1z * S2z
        ident = qt.tensor(qt.identity(2), qt.identity(2), id_nuc)
        H_Ex = -J * (2 * S1_dot_S2 + 0.5 * ident)
    
    return H_Ex


def dipolar_hamiltonian(d=0.0, r_hat=[0, 0, 1], dims=None):
    """Construct the dipolar coupling Hamiltonian.
    
    H_D = d (3(S₁·r̂)(S₂·r̂) − S₁·S₂)
    
    where d = μ₀ g₁ g₂ μ_B² / (4π r³) × (μ₀/4π).
    For r = 20 Å: d ≈ 0.1 μeV.
    For r = 10 Å: d ≈ 0.8 μeV.
    For r = 5 Å:  d ≈ 6 μeV.
    
    Parameters
    ----------
    d : float
        Dipolar coupling strength [μeV]. Default: 0.
    r_hat : array_like (3,)
        Unit vector connecting the two radicals.
    dims : list of Qobj, optional
    
    Returns
    -------
    H_D : Qobj
        Dipolar Hamiltonian in μeV.
    """
    if abs(d) < 1e-12:
        if dims is None:
            return qt.Qobj(np.zeros((4, 4)), dims=[[2, 2], [2, 2]])
        else:
            n_nuc = len(dims) - 2
            id_list = [qt.identity(2), qt.identity(2)]
            if n_nuc > 0:
                id_list += [qt.identity(2)] * n_nuc
            return 0 * qt.tensor(*id_list)
    
    r_hat = np.asarray(r_hat, dtype=float)
    r_hat = r_hat / np.linalg.norm(r_hat)
    
    if dims is None:
        sx, sy, sz = spin_operators(2)
        s1 = [qt.tensor(sx, qt.identity(2)),
              qt.tensor(sy, qt.identity(2)),
              qt.tensor(sz, qt.identity(2))]
        s2 = [qt.tensor(qt.identity(2), sx),
              qt.tensor(qt.identity(2), sy),
              qt.tensor(qt.identity(2), sz)]
        ident = qt.tensor(qt.identity(2), qt.identity(2))
    else:
        s1x, s1y, s1z = spin_operators(2)
        s2x, s2y, s2z = spin_operators(2)
        n_nuc = len(dims) - 2
        id_nuc = qt.tensor(*([qt.identity(2)] * n_nuc)) if n_nuc > 0 else qt.identity(1)
        s1 = [qt.tensor(s1x, qt.identity(2), id_nuc),
              qt.tensor(s1y, qt.identity(2), id_nuc),
              qt.tensor(s1z, qt.identity(2), id_nuc)]
        s2 = [qt.tensor(qt.identity(2), s2x, id_nuc),
              qt.tensor(qt.identity(2), s2y, id_nuc),
              qt.tensor(qt.identity(2), s2z, id_nuc)]
        ident = qt.tensor(qt.identity(2), qt.identity(2), id_nuc)
    
    S1_dot_S2 = s1[0] * s2[0] + s1[1] * s2[1] + s1[2] * s2[2]
    S1_dot_r = s1[0] * r_hat[0] + s1[1] * r_hat[1] + s1[2] * r_hat[2]
    S2_dot_r = s2[0] * r_hat[0] + s2[1] * r_hat[1] + s2[2] * r_hat[2]
    
    H_D = d * (3 * S1_dot_r * S2_dot_r - S1_dot_S2)
    return H_D


def total_hamiltonian(B=[0, 0, 1], g_A=2.0023, g_B=2.0023,
                     hfc_list_A=None, hfc_list_B=None,
                     J=0.0, d=0.0, r_hat=[0, 0, 1]):
    """Construct the full radical pair Hamiltonian.
    
    H = H_Z + H_HF + H_Ex + H_D
    
    Parameters
    ----------
    B : array_like (3,)
        Magnetic field [mT].
    g_A, g_B : float
        Electron g-factors.
    hfc_list_A : list of 3×3 arrays
        HFC tensors for radical A's nuclei.
    hfc_list_B : list of 3×3 arrays
        HFC tensors for radical B's nuclei.
    J : float
        Exchange coupling [μeV].
    d : float
        Dipolar coupling [μeV].
    r_hat : array_like (3,)
        Direction between radicals.
    
    Returns
    -------
    H : Qobj
        Total Hamiltonian.
    dims : list
        Tensor product dimensions for operator construction.
    """
    if hfc_list_A is None: hfc_list_A = []
    if hfc_list_B is None: hfc_list_B = []
    
    n_nuc = len(hfc_list_A) + len(hfc_list_B)
    
    if n_nuc > 0:
        # Build tensor product dimension list
        dims_list = [qt.identity(2), qt.identity(2)]
        for _ in range(n_nuc):
            dims_list.append(qt.identity(2))
        
        H = (zeeman_hamiltonian(B, g_A, g_B, dims_list) +
             hyperfine_hamiltonian(hfc_list_A, hfc_list_B, dims_list) +
             exchange_hamiltonian(J, dims_list) +
             dipolar_hamiltonian(d, r_hat, dims_list))
    else:
        H = (zeeman_hamiltonian(B, g_A, g_B) +
             exchange_hamiltonian(J) +
             dipolar_hamiltonian(d, r_hat))
    
    return H


def _count_nuclei(dims):
    """Count number of nuclear spins from dims specification.
    
    Handles both formats:
    1. List of Qobj: [id2, id2, id2, id2] → len-2 = 2 nuclei
    2. QuTiP 5 dims: [[2,2,2,2],[2,2,2,2]] → len(dims[0])-2 = 2 nuclei
    """
    if dims is None:
        return 0
    if isinstance(dims, (list, tuple)):
        if len(dims) > 0 and hasattr(dims[0], 'shape'):
            # List of Qobj format
            return len(dims) - 2
        elif len(dims) > 0 and isinstance(dims[0], (list, tuple)):
            # QuTiP 5 dims format [[subsys...],[subsys...]]
            return len(dims[0]) - 2
    return 0


def singlet_state(dims=None):
    """Construct the singlet state |S⟩ = (|↑↓⟩ - |↓↑⟩)/√2.
    
    Returns the density matrix ρ_S = |S⟩⟨S|.
    """
    up = qt.basis(2, 0)  # |↑⟩
    down = qt.basis(2, 1)  # |↓⟩
    
    # Singlet: (|↑↓⟩ - |↓↑⟩)/√2
    singlet = (qt.tensor(up, down) - qt.tensor(down, up)) / np.sqrt(2)
    
    n_nuc = _count_nuclei(dims)
    if n_nuc > 0:
        nuc_ground = qt.basis(2, 0)
        full_state = singlet
        for _ in range(n_nuc):
            full_state = qt.tensor(full_state, nuc_ground)
        return full_state * full_state.dag()
    
    return singlet * singlet.dag()


def triplet_states(dims=None):
    """Construct the three triplet states.
    
    |T₊⟩ = |↑↑⟩
    |T₀⟩ = (|↑↓⟩ + |↓↑⟩)/√2
    |T₋⟩ = |↓↓⟩
    
    Returns list of density matrices [ρ_{T+}, ρ_{T0}, ρ_{T-}].
    """
    up = qt.basis(2, 0)
    down = qt.basis(2, 1)
    
    tp = qt.tensor(up, up)
    t0 = (qt.tensor(up, down) + qt.tensor(down, up)) / np.sqrt(2)
    tm = qt.tensor(down, down)
    
    states = [tp, t0, tm]
    
    n_nuc = _count_nuclei(dims)
    if n_nuc > 0:
        nuc_ground = qt.basis(2, 0)
        expanded = []
        for s in states:
            full = s
            for _ in range(n_nuc):
                full = qt.tensor(full, nuc_ground)
            expanded.append(full * full.dag())
        return expanded
    else:
        return [s * s.dag() for s in states]
