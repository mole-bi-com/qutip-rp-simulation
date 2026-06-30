#!/usr/bin/env python3
"""
================================================================================
  Radical Pair Spin Dynamics — Comprehensive Demonstration Suite
================================================================================
  
  GLM 풀파워 극한 심화 데모.
  
  Phases:
    1. Basic RP (no nuclei)        — Zeeman + Exchange, MFE curve
    2. Hyperfine RP                — FADH•-like HFCs, Low Field Effect
    3. Anisotropy                  — Angular dependence, Cryptochrome compass
    4. Decoherence                 — T1 relaxation, T2 dephasing comparison
    5. Biological Cryptochrome     — Full Cry4-like realistic model
    6. Publication Figure          — 2×2 composite Journal-quality panel
  
  각 Phase는:
    - Hamiltonian 분석
    - Lindblad ME 풀이
    - MFE 곡선
    - 생물학적 해석
    - Figure 저장
  
  Output: /tmp/qutip_rp_simulation/figures/*.png
  
  Reference: APL Quantum (2024) arXiv:2406.12986
             Timmel et al., Chem. Phys. Lett. (1998)
             Hore, PNAS (2007)
================================================================================
"""

import sys, os, time
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Package import ────────────────────────────────────────────────────────
sys.path.insert(0, '/tmp/qutip_rp_simulation')
from rp_sim.hamiltonian import (
    zeeman_hamiltonian, hyperfine_hamiltonian, exchange_hamiltonian,
    dipolar_hamiltonian, total_hamiltonian, singlet_state, triplet_states,
    spin_operators
)
from rp_sim.solver import (
    rp_solve, singlet_yield, field_sweep, time_trace, mfe
)
from rp_sim.decoherence import (
    relaxation_ops, dephasing_ops, recombination_ops, decoherence_channels
)
from rp_sim.anisotropy import (
    anisotropic_field_sweep, cryptochrome_model,
    orientation_averaged_yield, rotation_matrix, rotate_tensor
)
from rp_sim.visualization import (
    plot_mfe_curve, plot_anisotropy_map, plot_time_evolution,
    plot_decoherence_comparison, plot_singlet_triplet_ratio
)

FIGS_DIR = '/tmp/qutip_rp_simulation/figures'
os.makedirs(FIGS_DIR, exist_ok=True)

# Color palette (Nature-style)
COLORS = {
    'singlet': '#2166AC',
    'triplet': '#B2182B',
    'mfe': '#4DAF4A',
    'T1': '#D6604D',
    'T2': '#4393C3',
    'theta': '#762A83',
    'reference': '#333333',
}

# ── Utility ───────────────────────────────────────────────────────────────
def timer(msg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            dt = time.time() - t0
            print(f"  ⏱  {msg}: {dt:.1f}s")
            return result
        return wrapper
    return decorator


def print_header(text):
    n = len(text)
    print(f"\n{'=' * (n + 8)}")
    print(f"   {text}")
    print(f"{'=' * (n + 8)}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: Basic Radical Pair — No Hyperfine Coupling
# ═══════════════════════════════════════════════════════════════════════════
@timer("Phase 1")
def phase_1_basic_rp():
    """Pure Zeeman + Exchange — simplest RP.
    
    Physics: Only electron Zeeman splitting and exchange coupling.
    The singlet-triplet mixing occurs ONLY due to the Δg mechanism
    (difference in g-factors creating an energy mismatch).
    
    For g_A = g_B, no S-T mixing → yield is constant.
    For g_A ≠ g_B, weak mixing occurs at high fields.
    """
    print_header("PHASE 1: Basic Radical Pair (No Hyperfine)")
    
    # 1.1 Hamiltonian Analysis
    print("  Hamiltonian: H = H_Z(g_A, g_B) + H_Ex(J)")
    g_A, g_B = 2.0023, 2.0040  # Slight g-factor difference (FADH• vs Trp•)
    J = 0.1  # μeV, weak exchange
    
    H_0 = total_hamiltonian(B=[0, 0, 0], g_A=g_A, g_B=g_B, J=J)
    H_1 = total_hamiltonian(B=[0, 0, 1], g_A=g_A, g_B=g_B, J=J)
    H_10 = total_hamiltonian(B=[0, 0, 10], g_A=g_A, g_B=g_B, J=J)
    
    print(f"  B=0:   eigenvalues =", np.sort(np.round(H_0.eigenenergies(), 4)))
    print(f"  B=1:   eigenvalues =", np.sort(np.round(H_1.eigenenergies(), 4)))
    print(f"  B=10:  eigenvalues =", np.sort(np.round(H_10.eigenenergies(), 4)))
    
    # 1.2 Singlet-Triplet energy splitting
    psi_s = singlet_state()
    e_S_val = np.real((H_0 * psi_s).tr())
    print(f"  Singlet energy expectation @ B=0: {e_S_val:.4f} μeV")
    
    # 1.3 MFE Curve
    B_range = np.logspace(-2, 2, 50)  # 0.01 to 100 mT
    k_S = 0.05  # 1/ns
    
    yields = field_sweep(B_range, k_S=k_S, g_A=g_A, g_B=g_B, J=J)
    mfe_vals = mfe(yields, yields[0])
    
    print(f"  Φ_S(B=0): {yields[0]:.4f}")
    print(f"  Φ_S(B=1):  {yields[np.argmin(np.abs(B_range - 1))]:.4f}")
    print(f"  Φ_S(B=10): {yields[np.argmin(np.abs(B_range - 10))]:.4f}")
    print(f"  Φ_S(B=100): {yields[-1]:.4f}")
    
    plot_mfe_curve(B_range, yields, mfe_vals,
                   labels=['Basic RP (Δg only)'],
                   title='Phase 1: Basic Radical Pair MFE (No Hyperfine)',
                   save_path=f'{FIGS_DIR}/phase1_basic_mfe.png')
    
    return {'B_range': B_range, 'yields': yields, 'mfe': mfe_vals}


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: Hyperfine-Mediated RP — Realistic Model
# ═══════════════════════════════════════════════════════════════════════════
@timer("Phase 2")
def phase_2_hyperfine_rp():
    """Radical pair with hyperfine coupling.
    
    Physics: Hyperfine interaction between electron and nuclear spins
    drives S-T mixing even at zero field. This produces:
      - Low Field Effect (LFE): sharp change at B ~ HFC strength
      - High field leveling: yield saturates when Zeeman > HFC
    
    This is the FADH•−Trp• model relevant to cryptochrome.
    """
    print_header("PHASE 2: Hyperfine-Mediated Radical Pair")
    
    # FADH• (radical A): dominant N5 coupling (single dominant nucleus)
    hfc_A = [
        np.diag([50.0, 50.0, 50.0]),   # N5: 50 μeV isotropic
    ]
    
    # Trp• (radical B): Hβ coupling
    hfc_B = [
        np.diag([15.0, 15.0, 15.0]),   # Hβ: 15 μeV
    ]
    
    print(f"  Radical A (FADH•): {len(hfc_A)} nucleus")
    print(f"    HFC strengths: {[np.trace(h)/3 for h in hfc_A]} μeV")
    print(f"  Radical B (Trp•): {len(hfc_B)} nucleus")
    print(f"    HFC strengths: {[np.trace(h)/3 for h in hfc_B]} μeV")
    print(f"  Total Hilbert space: {4 * 2**len(hfc_A) * 2**len(hfc_B)}")
    print(f"  = 4 × 2^{len(hfc_A)+len(hfc_B)} = {4 * 2**(len(hfc_A)+len(hfc_B))}")
    
    # Hamiltonian at key field values
    B_range = np.logspace(-1, 2, 20)  # 20 points
    k_S = 0.05
    k_T = 0.005
    
    # 2.1 Isotropic HFC
    yields_iso = field_sweep(B_range, k_S=k_S, k_T=k_T,
                             hfc_list_A=hfc_A, hfc_list_B=hfc_B)
    mfe_iso = mfe(yields_iso, yields_iso[0])
    
    # 2.2 Stronger HFC (double the couplings, 1+1 nuclei)
    hfc_A_strong = [
        np.diag([100.0, 100.0, 100.0]),   # N5: 100 μeV
    ]
    hfc_B_strong = [
        np.diag([30.0, 30.0, 30.0]),      # Hβ: 30 μeV
    ]
    
    yields_strong = field_sweep(B_range, k_S=k_S, k_T=k_T,
                                hfc_list_A=hfc_A_strong,
                                hfc_list_B=hfc_B_strong)
    
    # 2.3 No HFC (for comparison)
    yields_no_hfc = field_sweep(B_range, k_S=k_S, k_T=k_T)
    
    print(f"\n  MFE comparison (B=0.5 mT):")
    idx_05 = np.argmin(np.abs(B_range - 0.5))
    print(f"    No HFC:   Φ_S = {yields_no_hfc[idx_05]:.4f}")
    print(f"    Weak HFC: Φ_S = {yields_iso[idx_05]:.4f}")
    print(f"    Strong:   Φ_S = {yields_strong[idx_05]:.4f}")
    
    # 2.4 MFE curve plot
    plot_mfe_curve(B_range,
                   [yields_no_hfc, yields_iso, yields_strong],
                   mfe_iso,
                   labels=['No HFC', 'FADH•−Trp• (iso)', '2× HFC'],
                   title='Phase 2: Hyperfine Effect on MFE',
                   save_path=f'{FIGS_DIR}/phase2_hfc_mfe.png')
    
    # 2.5 Time evolution at B=0 and B=50 μT (geomagnetic)
    print(f"\n  Time evolution at geomagnetic field (50 μT = 0.05 mT)...")
    H_geo = total_hamiltonian(B=[0, 0, 0.05],
                              hfc_list_A=hfc_A, hfc_list_B=hfc_B)
    n_nuc = len(hfc_A) + len(hfc_B)
    dims = [qt.identity(2)] * 2 + [qt.identity(2)] * n_nuc
    rho0 = singlet_state(dims)
    
    result_geo = time_trace(H_geo, rho0, k_S, k_T, t_max=200, n_steps=1000)
    
    plot_time_evolution(result_geo.times,
                        [result_geo.expect[0], result_geo.expect[1]],
                        labels=['Singlet P_S', 'Triplet P_T'],
                        title='Phase 2: Time Evolution at B = 0.05 mT (Geomagnetic)',
                        save_path=f'{FIGS_DIR}/phase2_time_evolution.png')
    
    return {
        'B_range': B_range,
        'yields_iso': yields_iso,
        'yields_strong': yields_strong,
        'yields_no_hfc': yields_no_hfc,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: Anisotropy — Angular Dependence
# ═══════════════════════════════════════════════════════════════════════════
@timer("Phase 3")
def phase_3_anisotropy():
    """Angular dependence of MFE — the compass mechanism.
    
    Physics: The hyperfine tensors are NOT isotropic in real proteins.
    Anisotropic HFCs break spherical symmetry, making the singlet yield
    depend on the angle θ between the molecular frame and B-field.
    
    This is the physical basis of the avian compass:
      - Cry4 protein in the retina has a fixed orientation
      - As the bird rotates its head, θ changes
      - The bird perceives Φ_S(θ) as color/light intensity variations
    
    Key prediction: Anisotropic MFE requires:
      (a) Anisotropic hyperfine tensors AND
      (b) Fixed molecular orientation (protein scaffold)
    """
    print_header("PHASE 3: Anisotropy — Angular Dependence of MFE")
    
    # Anisotropic HFC tensors for FADH• (axial approximation)
    # |S⟩ has p_z orbital → stronger coupling along z
    a_iso_A = [50.0]
    a_aniso_A = [5.0]
    
    hfc_A_aniso = []
    for a_iso, a_ani in zip(a_iso_A, a_aniso_A):
        tensor = np.diag([a_iso - a_ani, a_iso - a_ani, a_iso + 2*a_ani])
        hfc_A_aniso.append(tensor)
    
    # Anisotropic HFC for Trp•
    a_iso_B = [15.0]
    a_aniso_B = [2.0]
    
    hfc_B_aniso = []
    for a_iso, a_ani in zip(a_iso_B, a_aniso_B):
        tensor = np.diag([a_iso - a_ani, a_iso - a_ani, a_iso + 2*a_ani])
        hfc_B_aniso.append(tensor)
    
    print(f"  Anisotropic HFC tensors (axial: a_perp, a_parallel):")
    for i, t in enumerate(hfc_A_aniso):
        print(f"    FADH• nucleus {i+1}: diag({t[0,0]:.0f}, {t[1,1]:.0f}, {t[2,2]:.0f}) μeV")
    
    # 3.1 2D anisotropy map: yield(θ, B)
    print(f"\n  Computing 2D anisotropy map (θ × B)...")
    B_range = np.logspace(-0.5, 1.5, 20)
    theta_vals = np.linspace(0, np.pi, 15)
    
    yield_map = anisotropic_field_sweep(
        B_range, theta_vals,
        k_S=0.05, k_T=0.005,
        hfc_list_A=hfc_A_aniso, hfc_list_B=hfc_B_aniso,
    )
    
    plot_anisotropy_map(theta_vals, B_range, yield_map,
                        title='Phase 3: Anisotropy — Singlet Yield vs θ and B',
                        save_path=f'{FIGS_DIR}/phase3_anisotropy_map.png')
    
    # 3.2 Angular slices at key B values
    print(f"  Angular profiles at selected B values:")
    fig, ax = plt.subplots(figsize=(8, 5))
    theta_deg = np.degrees(theta_vals)
    
    for B_target in [0.05, 0.5, 1.0, 10.0]:
        idx_B = np.argmin(np.abs(B_range - B_target))
        yields_theta = yield_map[:, idx_B]
        ax.plot(theta_deg, yields_theta, '-', linewidth=1.8,
                label=f'B = {B_range[idx_B]:.2f} mT')
        variation = (yields_theta.max() - yields_theta.min()) / yields_theta.mean() * 100
        print(f"    B={B_range[idx_B]:.2f} mT: ΔΦ_S/⟨Φ_S⟩ = {variation:.1f}%")
    
    ax.set_xlabel('Angle $\\theta$ (degrees)')
    ax.set_ylabel('Singlet Yield $\\Phi_S$')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Phase 3: Angular Dependence of MFE')
    plt.tight_layout()
    plt.savefig(f'{FIGS_DIR}/phase3_angular_slices.png', dpi=300)
    
    # 3.3 Orientation-averaged yield (powder average)
    print(f"\n  Orientation-averaged yield (powder)...")
    n_theta = 8
    avg_yield_005 = orientation_averaged_yield(
        0.05, hfc_A_aniso, hfc_B_aniso, k_S=0.05, k_T=0.005,
        n_theta=n_theta, n_phi=16
    )
    avg_yield_1 = orientation_averaged_yield(
        1.0, hfc_A_aniso, hfc_B_aniso, k_S=0.05, k_T=0.005,
        n_theta=n_theta, n_phi=32
    )
    avg_yield_10 = orientation_averaged_yield(
        10.0, hfc_A_aniso, hfc_B_aniso, k_S=0.05, k_T=0.005,
        n_theta=n_theta, n_phi=32
    )
    
    print(f"    ⟨Φ_S⟩(B=0.05 mT) = {avg_yield_005:.4f}")
    print(f"    ⟨Φ_S⟩(B=1.0 mT)  = {avg_yield_1:.4f}")
    print(f"    ⟨Φ_S⟩(B=10 mT)   = {avg_yield_10:.4f}")
    
    return {
        'theta_vals': theta_vals,
        'B_range': B_range,
        'yield_map': yield_map,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: Decoherence Analysis
# ═══════════════════════════════════════════════════════════════════════════
@timer("Phase 4")
def phase_4_decoherence():
    """Effect of decoherence on radical pair yield.
    
    Physics: Two key decoherence channels:
    
    T₁ (spin-lattice relaxation):
      - Energy exchange with environment
      - Destroys spin polarization
      - Rate: 1/T₁
      - In proteins: T₁ ~ 1-10 μs
    
    T₂ (spin-spin dephasing):
      - Phase randomization without energy loss
      - Destroys quantum coherence
      - Rate: 1/T₂ (pure dephasing)
      - In proteins: T₂ ~ 100 ns - 1 μs
    
    Key question: How much decoherence can the RP tolerate
    while still showing a measurable MFE?
    """
    print_header("PHASE 4: Decoherence Analysis")
    
    # Model system: FADH•−Trp• (1+1 nuclei)
    hfc_A = [np.diag([50.0, 50.0, 50.0])]
    hfc_B = [np.diag([15.0, 15.0, 15.0])]
    
    n_nuc = len(hfc_A) + len(hfc_B)
    dims = [qt.identity(2)] * 2 + [qt.identity(2)] * n_nuc
    
    B_geo = 0.05  # mT — Earth's field
    
    H = total_hamiltonian(B=[0, 0, B_geo],
                          hfc_list_A=hfc_A, hfc_list_B=hfc_B)
    rho0 = singlet_state(dims)
    
    # 4.1 T₁ relaxation sweep
    print(f"  T₁ relaxation sweep (B = {B_geo} mT)...")
    T1_values = np.logspace(1, 5, 20)  # 10 ns to 100 μs
    
    ref_yield = singlet_yield(H, rho0, k_S=0.05, k_T=0.005, t_max=5000)[0]
    print(f"    Reference (no decoherence): Φ_S = {ref_yield:.4f}")
    
    P_S_op = singlet_state(dims)
    
    yields_T1 = []
    for T1 in T1_values:
        result = rp_solve(H, rho0, k_S=0.05, k_T=0.005, t_max=5000, n_steps=2000,
                          T1=T1, e_ops=[P_S_op])
        P_S = result.expect[0]
        phi_S = 0.05 * np.trapezoid(P_S, result.times)
        yields_T1.append(phi_S)
    
    # 4.2 T₂ dephasing sweep
    print(f"  T₂ dephasing sweep...")
    T2_values = np.logspace(1, 5, 20)
    
    yields_T2 = []
    for T2 in T2_values:
        result = rp_solve(H, rho0, k_S=0.05, k_T=0.005, t_max=5000, n_steps=2000,
                          T2=T2, e_ops=[P_S_op])
        P_S = result.expect[0]
        phi_S = 0.05 * np.trapezoid(P_S, result.times)
        yields_T2.append(phi_S)
    
    # 4.3 Combined decoherence analysis
    print(f"\n  Critical decoherence thresholds:")
    for i, (T1, y_T1, y_T2) in enumerate(zip(T1_values, yields_T1, yields_T2)):
        if i % 4 == 0:
            print(f"    T = {T1:.0f} ns:  Φ_S(T₁)={y_T1:.4f}  Φ_S(T₂)={y_T2:.4f}")
    
    # Find where yield drops by 50%
    half_idx_T1 = np.argmin(np.abs(np.array(yields_T1) - ref_yield * 0.5))
    half_idx_T2 = np.argmin(np.abs(np.array(yields_T2) - ref_yield * 0.5))
    print(f"    T₁ for 50% yield loss: {T1_values[half_idx_T1]:.0f} ns")
    print(f"    T₂ for 50% yield loss: {T2_values[half_idx_T2]:.0f} ns")
    
    # 4.4 Decoherence comparison figure
    plot_decoherence_comparison(
        T1_values, yields_T1, T2_values, yields_T2,
        title=f'Phase 4: Decoherence Effect on RP Yield (B={B_geo} mT)',
        save_path=f'{FIGS_DIR}/phase4_decoherence.png'
    )
    
    return {
        'T1_values': T1_values,
        'T2_values': T2_values,
        'yields_T1': yields_T1,
        'yields_T2': yields_T2,
        'ref_yield': ref_yield,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: Biological Cryptochrome Model
# ═══════════════════════════════════════════════════════════════════════════
@timer("Phase 5")
def phase_5_cryptochrome():
    """Full cryptochrome compass model.
    
    Simulates the avian cryptochrome 4 (Cry4) radical pair as proposed
    by Ritz, Schulten, Hore and others.
    
    The compass mechanism:
      1. Blue light activates Cry4 → FADH•−Trp• radical pair
      2. RP evolves under geomagnetic field (~50 μT)
      3. Singlet/triplet yield depends on field orientation
      4. Yield difference → neural signal → compass sense
    
    Key predictions tested:
      - Anisotropic MFE at geomagnetic field (50 μT)
      - Angular sensitivity (how fine can the bird distinguish?)
      - Effect of inclination (equator vs pole)
    """
    print_header("PHASE 5: Biological Cryptochrome Compass Model")
    
    # Realistic Cry4 parameters
    g_A = 2.0040   # FADH• radical
    g_B = 2.0023   # Trp• radical (close to free electron)
    k_S = 0.05     # ns⁻¹ (singlet recombination)
    k_T = 0.005    # ns⁻¹ (triplet recombination, slower)
    
    # Anisotropic HFC tensor — FADH• (single dominant nucleus N5)
    hfc_A = [
        np.diag([45.0, 47.0, 58.0]),   # N5: strongly anisotropic
    ]
    
    # Trp• anisotropic HFC (single dominant Hβ)
    hfc_B = [
        np.diag([14.0, 15.0, 16.0]),   # Hβ
    ]
    
    n_nuc = len(hfc_A) + len(hfc_B)
    print(f"  Cryptochrome 4 model:")
    print(f"    Radical pair: FADH•−Trp•")
    print(f"    Total nuclear spins: {n_nuc}")
    print(f"    Hilbert space dim: {2**(2 + n_nuc)}")
    print(f"    Recombination: k_S={k_S}, k_T={k_T} ns⁻¹")
    
    # 5.1 Angular sweep at geomagnetic field (50 μT = 0.05 mT)
    print(f"\n  Angular sweep at B = 0.05 mT (geomagnetic field)...")
    theta_vals = np.linspace(0, np.pi, 25)
    
    # Isotropic vs anisotropic comparison
    hfc_A_iso = [np.eye(3) * np.trace(t)/3 for t in hfc_A]
    hfc_B_iso = [np.eye(3) * np.trace(t)/3 for t in hfc_B]
    
    yields_iso = []
    yields_aniso = []
    
    for theta in theta_vals:
        # Isotropic case
        B_vec = np.array([np.sin(theta), 0, np.cos(theta)]) * 0.05
        H_iso = total_hamiltonian(B=B_vec, g_A=g_A, g_B=g_B,
                                  hfc_list_A=hfc_A_iso, hfc_list_B=hfc_B_iso)
        dims = [qt.identity(2)] * 2 + [qt.identity(2)] * n_nuc
        rho0 = singlet_state(dims)
        phi_iso, _ = singlet_yield(H_iso, rho0, k_S, k_T, t_max=5000)
        yields_iso.append(phi_iso)
        
        # Anisotropic case
        R = rotation_matrix([0, 1, 0], theta)
        hfc_A_rot = [rotate_tensor(t, R) for t in hfc_A]
        hfc_B_rot = [rotate_tensor(t, R) for t in hfc_B]
        H_aniso = total_hamiltonian(B=[0, 0, 0.05], g_A=g_A, g_B=g_B,
                                    hfc_list_A=hfc_A_rot, hfc_list_B=hfc_B_rot)
        phi_aniso, _ = singlet_yield(H_aniso, rho0, k_S, k_T, t_max=5000)
        yields_aniso.append(phi_aniso)
    
    yields_iso = np.array(yields_iso)
    yields_aniso = np.array(yields_aniso)
    
    # 5.2 Sensitivity analysis
    aniso_contrast = (yields_aniso.max() - yields_aniso.min())
    iso_contrast = (yields_iso.max() - yields_iso.min())
    
    print(f"\n  Compass sensitivity analysis:")
    print(f"    Isotropic HFC:    ΔΦ_S = {iso_contrast:.4f}  ({iso_contrast/yields_iso.mean()*100:.1f}%)")
    print(f"    Anisotropic HFC:  ΔΦ_S = {aniso_contrast:.4f}  ({aniso_contrast/yields_aniso.mean()*100:.1f}%)")
    print(f"    Enhancement:      {aniso_contrast/iso_contrast:.1f}×")
    
    # 5.3 Angular resolution
    # Minimum detectable change: need significant yield difference
    noise_level = 0.001  # Estimated neural noise in yield detection
    theta_deg = np.degrees(theta_vals)
    dPhi_dtheta = np.gradient(yields_aniso, theta_deg)
    min_resolvable = noise_level / np.max(np.abs(dPhi_dtheta))
    print(f"    Estimated angular resolution: {min_resolvable:.1f}°")
    
    # 5.4 The compass figure
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(theta_deg, yields_iso, '--', color='#AAAAAA', linewidth=1.5,
            label='Isotropic HFC (no compass)')
    ax.plot(theta_deg, yields_aniso, '-', color=COLORS['theta'],
            linewidth=2.5, label='Anisotropic HFC (compass)')
    
    # Shading for functional range
    ax.fill_between([0, 180], yields_aniso.min(), yields_aniso.max(),
                    alpha=0.08, color=COLORS['theta'])
    
    ax.set_xlabel('Angle $\\theta$ (degrees)', fontsize=13)
    ax.set_ylabel('Singlet Yield $\\Phi_S$', fontsize=13)
    ax.set_title('Phase 5: Avian Compass — Angular Sensitivity at 50 μT',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 180)
    ax.set_ylim(yields_aniso.min() * 0.95, yields_aniso.max() * 1.05)
    plt.tight_layout()
    plt.savefig(f'{FIGS_DIR}/phase5_cryptochrome_compass.png', dpi=300)
    plt.close()
    
    # 5.5 MFE curves at θ=0 and θ=90°
    print(f"\n  MFE curves at key orientations...")
    B_range = np.logspace(-1, 1.5, 20)
    
    yields_theta0 = []
    yields_theta90 = []
    
    for B in B_range:
        # θ=0: field along molecular z
        hfc_rot_0 = [rotate_tensor(t, np.eye(3)) for t in hfc_A]
        hfc_B_rot_0 = [rotate_tensor(t, np.eye(3)) for t in hfc_B]
        H_0 = total_hamiltonian(B=[0, 0, B], g_A=g_A, g_B=g_B,
                                hfc_list_A=hfc_rot_0, hfc_list_B=hfc_B_rot_0)
        phi_0, _ = singlet_yield(H_0, rho0, k_S, k_T, t_max=5000)
        yields_theta0.append(phi_0)
        
        # θ=90°: field ⊥ molecular z
        R90 = rotation_matrix([0, 1, 0], np.pi/2)
        hfc_rot_90 = [rotate_tensor(t, R90) for t in hfc_A]
        hfc_B_rot_90 = [rotate_tensor(t, R90) for t in hfc_B]
        H_90 = total_hamiltonian(B=[0, 0, B], g_A=g_A, g_B=g_B,
                                 hfc_list_A=hfc_rot_90, hfc_list_B=hfc_B_rot_90)
        phi_90, _ = singlet_yield(H_90, rho0, k_S, k_T, t_max=5000)
        yields_theta90.append(phi_90)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(B_range, yields_theta0, '-', color=COLORS['singlet'],
                linewidth=2.0, label='$\\theta = 0°$ (parallel)')
    ax.semilogx(B_range, yields_theta90, '-', color=COLORS['triplet'],
                linewidth=2.0, label='$\\theta = 90°$ (perpendicular)')
    ax.axvline(x=0.05, color='gray', linestyle=':', alpha=0.7)
    ax.text(0.055, 0.92, 'Geomagnetic\nfield', fontsize=9, color='gray')
    ax.set_xlabel('Magnetic Field $B_0$ (mT)', fontsize=13)
    ax.set_ylabel('Singlet Yield $\\Phi_S$', fontsize=13)
    ax.set_title('Phase 5: MFE at Different Orientations', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{FIGS_DIR}/phase5_orientation_mfe.png', dpi=300)
    plt.close()
    
    return {
        'theta_vals': theta_vals,
        'yields_iso': yields_iso,
        'yields_aniso': yields_aniso,
        'angular_resolution': min_resolvable,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 6: Publication-Quality Composite Figure
# ═══════════════════════════════════════════════════════════════════════════
@timer("Phase 6")
def phase_6_publication_figure():
    """Generate a composite 2×2 figure suitable for journal publication.
    
    Panels:
      (a) MFE curve: HFC strength comparison
      (b) Anisotropy map: yield(θ, B)
      (c) Decoherence: T1 vs T2 effect
      (d) Compass: angular profile at geomagnetic field
    """
    print_header("PHASE 6: Publication-Quality Composite Figure")
    
    # ── Recompute data for the panels ──
    # Panel A: MFE (1+1 nuclei)
    hfc_A = [np.diag([50.0, 50.0, 50.0])]
    hfc_B = [np.diag([15.0, 15.0, 15.0])]
    B_range = np.logspace(-1, 2, 25)
    yields_s = field_sweep(B_range, k_S=0.05, k_T=0.005,
                           hfc_list_A=hfc_A, hfc_list_B=hfc_B)
    B_0 = yields_s[0]
    mfe_vals = (yields_s - B_0) / B_0
    
    # Panel B: Anisotropy map (1+1 nuclei)
    hfc_A_aniso = [np.diag([45.0, 47.0, 58.0])]
    hfc_B_aniso = [np.diag([14.0, 15.0, 16.0])]
    B_aniso = np.logspace(-0.5, 1.5, 15)
    theta_vals = np.linspace(0, np.pi, 12)
    yield_map = anisotropic_field_sweep(
        B_aniso, theta_vals, k_S=0.05, k_T=0.005,
        hfc_list_A=hfc_A_aniso, hfc_list_B=hfc_B_aniso,
    )
    
    # Panel C: Decoherence — simpler calc
    n_nuc = len(hfc_A_aniso) + len(hfc_B_aniso)
    dims = [qt.identity(2)] * 2 + [qt.identity(2)] * n_nuc
    H_dec = total_hamiltonian(B=[0, 0, 0.05],
                              hfc_list_A=hfc_A_aniso, hfc_list_B=hfc_B_aniso)
    rho0 = singlet_state(dims)
    
    T1_vals = np.logspace(1.5, 4.5, 15)
    P_S_pub = singlet_state(dims)
    yields_T1_pub = []
    for T1 in T1_vals:
        result = rp_solve(H_dec, rho0, k_S=0.05, k_T=0.005, t_max=5000, n_steps=2000,
                          T1=T1, e_ops=[P_S_pub])
        yields_T1_pub.append(0.05 * np.trapezoid(result.expect[0], result.times))
    
    T2_vals = np.logspace(1.5, 4.5, 15)
    yields_T2_pub = []
    for T2 in T2_vals:
        result = rp_solve(H_dec, rho0, k_S=0.05, k_T=0.005, t_max=5000, n_steps=2000,
                          T2=T2, e_ops=[P_S_pub])
        yields_T2_pub.append(0.05 * np.trapezoid(result.expect[0], result.times))
    
    # Panel D: Compass
    theta_d = np.linspace(0, np.pi, 35)
    yields_compass = []
    for theta in theta_d:
        R = rotation_matrix([0, 1, 0], theta)
        hfc_A_rot = [rotate_tensor(t, R) for t in hfc_A_aniso]
        hfc_B_rot = [rotate_tensor(t, R) for t in hfc_B_aniso]
        H_c = total_hamiltonian(B=[0, 0, 0.05], g_A=2.004, g_B=2.0023,
                                hfc_list_A=hfc_A_rot, hfc_list_B=hfc_B_rot)
        phi_c, _ = singlet_yield(H_c, rho0, 0.05, 0.005, t_max=5000)
        yields_compass.append(phi_c)
    
    # ── Composite figure ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel (a): MFE curve
    ax_a = axes[0, 0]
    ax_a.plot(B_range, yields_s, '-', color='#2166AC', linewidth=2.0)
    ax_a_twin = ax_a.twinx()
    ax_a_twin.plot(B_range, mfe_vals * 100, '--', color='#4DAF4A',
                   linewidth=1.5, alpha=0.7)
    ax_a.set_xlabel('$B_0$ (mT)')
    ax_a.set_ylabel('$\\Phi_S$', color='#2166AC')
    ax_a_twin.set_ylabel('MFE (%)', color='#4DAF4A')
    ax_a.set_xscale('log')
    ax_a.set_title('(a) Magnetic Field Effect', fontweight='bold')
    ax_a.grid(True, alpha=0.3)
    ax_a.axhline(y=0.25, color='gray', linestyle=':', alpha=0.3)
    ax_a.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    
    # Panel (b): Anisotropy map
    ax_b = axes[0, 1]
    theta_deg = np.degrees(theta_vals)
    T, B = np.meshgrid(theta_deg, B_aniso)
    im = ax_b.pcolormesh(T, B, yield_map.T, shading='auto',
                         cmap='RdBu_r', vmin=0, vmax=1)
    ax_b.set_xlabel('$\\theta$ (deg)')
    ax_b.set_ylabel('$B_0$ (mT)')
    ax_b.set_title('(b) Anisotropy $\\Phi_S(\\theta, B)$', fontweight='bold')
    plt.colorbar(im, ax=ax_b)
    
    # Panel (c): Decoherence
    ax_c = axes[1, 0]
    ax_c.semilogx(T1_vals, yields_T1_pub, 'o-', color='#D6604D',
                  linewidth=2.0, markersize=5, label='$T_1$ relax.')
    ax_c.semilogx(T2_vals, yields_T2_pub, 's-', color='#4393C3',
                  linewidth=2.0, markersize=5, label='$T_2$ deph.')
    ax_c.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4)
    ax_c.set_xlabel('Decoherence time (ns)')
    ax_c.set_ylabel('$\\Phi_S$')
    ax_c.set_title('(c) Decoherence Effect', fontweight='bold')
    ax_c.legend(loc='best', framealpha=0.9)
    ax_c.grid(True, alpha=0.3)
    
    # Panel (d): Compass
    ax_d = axes[1, 1]
    ax_d.plot(np.degrees(theta_d), yields_compass, '-', color='#762A83',
              linewidth=2.5)
    ax_d.fill_between(np.degrees(theta_d), yields_compass,
                      min(yields_compass), alpha=0.15, color='#762A83')
    ax_d.set_xlabel('$\\theta$ (deg)')
    ax_d.set_ylabel('$\\Phi_S$ @ 50 μT')
    ax_d.set_title('(d) Avian Compass Sensitivity', fontweight='bold')
    ax_d.set_xlim(0, 180)
    ax_d.grid(True, alpha=0.3)
    
    plt.suptitle('Radical Pair Spin Dynamics — Comprehensive Analysis',
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{FIGS_DIR}/phase6_publication_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{FIGS_DIR}/phase6_publication_figure.pdf', bbox_inches='tight')
    plt.close()
    
    print(f"  Published to: {FIGS_DIR}/phase6_publication_figure.png/.pdf")
    
    return {
        'B_range': B_range,
        'yields_s': yields_s,
        'mfe_vals': mfe_vals,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    total_t0 = time.time()
    
    print("=" * 70)
    print("  RADICAL PAIR SPIN DYNAMICS — COMPREHENSIVE SIMULATION SUITE")
    print("  QuTiP 5.3 | Lindblad Master Equation | Publication-Quality")
    print("=" * 70)
    
    results = {}
    
    results['phase1'] = phase_1_basic_rp()
    results['phase2'] = phase_2_hyperfine_rp()
    results['phase3'] = phase_3_anisotropy()
    results['phase4'] = phase_4_decoherence()
    results['phase5'] = phase_5_cryptochrome()
    results['phase6'] = phase_6_publication_figure()
    
    total_time = time.time() - total_t0
    
    print("\n" + "=" * 70)
    print(f"  ✅ ALL PHASES COMPLETE — Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 70)
    print(f"\n  Figures saved to: {FIGS_DIR}/")
    print(f"  Files:")
    for f in sorted(os.listdir(FIGS_DIR)):
        fpath = os.path.join(FIGS_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s} {size_kb:7.1f} KB")
    print()
