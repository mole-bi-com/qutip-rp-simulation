"""
Anisotropy Analysis — Cryptochrome Orientation Model
======================================================

The magnetic field effect in cryptochrome-based radical pairs is
**strongly anisotropic** — it depends on the orientation of the
protein (and hence the radical pair) relative to the external
magnetic field.

In the avian compass:
  - Cryptochrome 4 (in the retina) is proposed as the magnetoreceptor
  - The RP quantum yield depends on the angle θ between the
    geomagnetic field and the molecular reference frame
  - This angular dependence forms the basis of the compass sense

The hyperfine tensors A_k and the dipolar axis r̂ define the
molecular frame. The external field direction in the molecular
frame is determined by rotation.

Key references:
  - Rodgers & Hore, PNAS (2009) — chemical compass model
  - Ritz, Adem & Schulten, Biophys. J. (2000) — cryptochrome model
  - Kattnig, Evans & Hore, JACS (2016) — flavin-tryptophan RP
"""

import numpy as np
import qutip as qt
from .hamiltonian import total_hamiltonian, singlet_state
from .solver import singlet_yield


def rotation_matrix(axis, angle):
    """Construct 3D rotation matrix using Rodrigues' formula.
    
    Parameters
    ----------
    axis : array_like (3,)
        Rotation axis (unit vector).
    angle : float
        Rotation angle [radians].
    
    Returns
    -------
    R : ndarray (3, 3)
        Rotation matrix.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2)
    b, c, d = -axis * np.sin(angle / 2)
    
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])


def rotate_tensor(A, R):
    """Rotate a 3×3 tensor by rotation matrix R.
    
    A_rot = R · A · R^T
    """
    return R @ A @ R.T


def anisotropic_field_sweep(B_range, theta_values, k_S, k_T=None,
                            g_A=2.0023, g_B=2.0023,
                            hfc_list_A=None, hfc_list_B=None,
                            J=0.0, d=0.0, r_hat=[0, 0, 1],
                            molecular_axis=[0, 0, 1],
                            progress=True):
    """Compute 2D map of singlet yield vs B and angle.
    
    Parameters
    ----------
    B_range : array_like
        Magnetic field strengths [mT].
    theta_values : array_like
        Angles between field and molecular axis [radians].
    k_S : float
        Singlet recombination rate [1/ns].
    g_A, g_B : float
        Electron g-factors.
    hfc_list_A, hfc_list_B : list
        HFC tensors (in molecular frame).
    J, d : float
        Exchange and dipolar coupling.
    r_hat : array_like
        Dipolar axis (in molecular frame).
    molecular_axis : array_like
        Axis of molecular frame to measure field angle from.
    progress : bool
    
    Returns
    -------
    yield_map : ndarray (n_theta, n_B)
        Yield as a function of θ and B.
    """
    if hfc_list_A is None: hfc_list_A = []
    if hfc_list_B is None: hfc_list_B = []
    
    theta_values = np.atleast_1d(theta_values)
    B_range = np.atleast_1d(B_range)
    yield_map = np.zeros((len(theta_values), len(B_range)))
    
    molecular_axis = np.asarray(molecular_axis, dtype=float)
    molecular_axis = molecular_axis / np.linalg.norm(molecular_axis)
    
    # Find a rotation axis perpendicular to molecular_axis
    if abs(molecular_axis[2]) < 0.9:
        rot_axis = np.cross(molecular_axis, [0, 0, 1])
    else:
        rot_axis = np.cross(molecular_axis, [1, 0, 0])
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    
    for i_theta, theta in enumerate(theta_values):
        # Rotate the molecular frame by θ around rot_axis
        R = rotation_matrix(rot_axis, theta)
        
        # Rotate HFC tensors
        hfc_A_rot = [rotate_tensor(np.asarray(h, dtype=float).reshape(3, 3), R)
                     for h in hfc_list_A]
        hfc_B_rot = [rotate_tensor(np.asarray(h, dtype=float).reshape(3, 3), R)
                     for h in hfc_list_B]
        
        # Rotate dipolar axis
        r_hat_rot = R @ np.asarray(r_hat, dtype=float)
        
        for i_B, B in enumerate(B_range):
            H = total_hamiltonian(
                B=[0, 0, B], g_A=g_A, g_B=g_B,
                hfc_list_A=hfc_A_rot, hfc_list_B=hfc_B_rot,
                J=J, d=d, r_hat=r_hat_rot
            )
            
            n_nuc = len(hfc_list_A) + len(hfc_list_B)
            dims = ([qt.identity(2)] * 2 +
                    [qt.identity(2)] * n_nuc) if n_nuc > 0 else None
            
            rho0 = singlet_state(dims)
            phi_S, _ = singlet_yield(H, rho0, k_S, k_T)
            yield_map[i_theta, i_B] = phi_S
    
    return yield_map


def cryptochrome_model(B_ext, theta, g_A=2.0023, g_B=2.0040,
                       k_S=0.05, k_T=0.005,
                       include_anisotropy=True):
    """Simplified cryptochrome 4 (Cry4) radical pair model.
    
    Uses experimentally motivated parameters for the
    FADH•−Trp• radical pair in avian cryptochrome 4.
    
    FADH• (radical A): 5 dominant hyperfine couplings
    - N5:  a_iso ≈ 50 μeV (isotropic)
    - N10: a_iso ≈ 25 μeV (isotropic)
    - H5:  a_iso ≈ 10 μeV (isotropic)
    - Plus weaker couplings
    
    Trp• (radical B): 3 dominant hyperfine couplings
    - Hβ: a_iso ≈ 15 μeV
    - H ring protons: a_iso ≈ 5-8 μeV
    
    Parameters
    ----------
    B_ext : array_like (3,)
        External magnetic field vector [mT].
    theta : float
        Angle between field and molecular z-axis [radians].
    g_A, g_B : float
        g-factors for FADH• and Trp•.
    k_S, k_T : float
        Recombination rates [1/ns].
    include_anisotropy : bool
        If True, use anisotropic HFC tensors.
        If False, use isotropic (averaged) HFCs.
    
    Returns
    -------
    phi_S : float
        Singlet yield.
    """
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # FADH• hyperfine couplings (μeV) — approximate values
    # From: Solov'yov et al., Biophys. J. (2012)
    nuc_A_iso = [50.0, 25.0, 11.0, 7.0, 5.0]  # a_iso [μeV]
    
    if include_anisotropy:
        # Full anisotropic tensors (axial approximation)
        # HFC tensor = a_iso × I₃ + a_aniso × diag(-1, -1, 2)
        nuc_A = []
        for a_iso in nuc_A_iso:
            a_aniso = 3.0 + np.random.uniform(-1, 1)  # ~2-4 μeV anisotropic
            tensor = np.diag([a_iso - a_aniso, a_iso - a_aniso, a_iso + 2*a_aniso])
            nuc_A.append(tensor)
    else:
        nuc_A = [np.eye(3) * a for a in nuc_A_iso]
    
    # Trp• hyperfine couplings (μeV)
    nuc_B_iso = [15.0, 8.0, 5.0, 3.0]
    if include_anisotropy:
        nuc_B = []
        for a_iso in nuc_B_iso:
            a_aniso = 1.5 + np.random.uniform(-0.5, 0.5)
            tensor = np.diag([a_iso - a_aniso, a_iso - a_aniso, a_iso + 2*a_aniso])
            nuc_B.append(tensor)
    else:
        nuc_B = [np.eye(3) * a for a in nuc_B_iso]
    
    # Apply rotation if angled
    B = np.asarray(B_ext, dtype=float)
    if len(B.shape) == 0 or B.shape == ():
        B = np.array([0, 0, B])
    B_norm = np.linalg.norm(B)
    B_dir = B / B_norm if B_norm > 0 else np.array([0, 0, 1])
    
    # Rotate molecular frame so that molecular z is at angle θ relative to B
    # Simple approach: rotate HFC tensors
    if abs(theta) > 1e-6:
        # Rotation axis perpendicular to both B_dir and [0,0,1]
        ref_axis = np.array([0, 0, 1])
        rot_axis = np.cross(ref_axis, B_dir)
        rot_norm = np.linalg.norm(rot_axis)
        if rot_norm > 1e-10:
            rot_axis = rot_axis / rot_norm
            R = rotation_matrix(rot_axis, theta)
            nuc_A = [rotate_tensor(t, R) for t in nuc_A]
            nuc_B = [rotate_tensor(t, R) for t in nuc_B]
    
    H = total_hamiltonian(
        B=B, g_A=g_A, g_B=g_B,
        hfc_list_A=nuc_A, hfc_list_B=nuc_B,
        J=0.0, d=0.0
    )
    
    n_nuc = len(nuc_A) + len(nuc_B)
    dims = [qt.identity(2)] * 2 + [qt.identity(2)] * n_nuc
    rho0 = singlet_state(dims)
    
    phi_S, _ = singlet_yield(H, rho0, k_S, k_T)
    return phi_S


def orientation_averaged_yield(B, hfc_list_A, hfc_list_B,
                              k_S, k_T=0.0, n_theta=32, n_phi=64,
                              return_map=False, progress=True):
    """Compute the orientation-averaged singlet yield.
    
    For powder/averaged samples, the yield is the average over
    all molecular orientations relative to the field:
    
    ⟨Φ_S⟩ = (1/4π) ∫₀²π∫₀^π Φ_S(θ, φ) sin θ dθ dφ
    
    Uses a Lebedev-like angular grid (θ, φ sampling).
    
    Parameters
    ----------
    B : float
        Magnetic field strength [mT].
    hfc_list_A, hfc_list_B : list
        Hyperfine tensors.
    k_S, k_T : float
        Recombination rates.
    n_theta, n_phi : int
        Angular grid resolution.
    return_map : bool
        Return the full angular map.
    progress : bool
    
    Returns
    -------
    avg_yield : float
        Orientation-averaged singlet yield.
    yield_map : ndarray (optional)
        Full angular map [θ, φ].
    """
    if hfc_list_A is None: hfc_list_A = []
    if hfc_list_B is None: hfc_list_B = []
    
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi)
    
    yield_map_3d = np.zeros((n_theta, n_phi))
    
    for i_t, theta in enumerate(thetas):
        for i_p, phi in enumerate(phis):
            # Rotation by θ around y-axis, then φ around z-axis
            R_y = rotation_matrix([0, 1, 0], theta)
            R_z = rotation_matrix([0, 0, 1], phi)
            R = R_z @ R_y
            
            hfc_A_rot = [rotate_tensor(np.asarray(h).reshape(3,3), R)
                        for h in hfc_list_A]
            hfc_B_rot = [rotate_tensor(np.asarray(h).reshape(3,3), R)
                        for h in hfc_list_B]
            
            H = total_hamiltonian(
                B=[0, 0, B],
                hfc_list_A=hfc_A_rot, hfc_list_B=hfc_B_rot,
            )
            
            n_nuc = len(hfc_list_A) + len(hfc_list_B)
            dims = [qt.identity(2)] * 2 + [qt.identity(2)] * n_nuc
            rho0 = singlet_state(dims)
            phi_S, _ = singlet_yield(H, rho0, k_S, k_T, t_max=5000)
            yield_map_3d[i_t, i_p] = phi_S
    
    # Average with sinθ weighting
    sin_theta = np.sin(thetas)
    avg_over_phi = np.mean(yield_map_3d, axis=1)  # average over φ
    avg_yield = np.trapz(avg_over_phi * sin_theta, thetas) / np.trapz(sin_theta, thetas)
    
    if return_map:
        return avg_yield, (thetas, phis, yield_map_3d)
    return avg_yield
