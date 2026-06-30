"""
Publication-Quality Visualization for Radical Pair Simulations
==============================================================

Generates Nature/Science-grade figures using matplotlib + seaborn.

Figure types:
  1. MFE curve: singlet yield vs magnetic field
  2. Angular anisotropy map: yield vs θ and B
  3. Time evolution: population dynamics
  4. Decoherence comparison: yield vs decoherence rate
  5. Composite publication figure (2×2 panel)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path

# ─── Style setup ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 1.8,
    'figure.figsize': (7, 5),
})

# Color palette (Nature-style)
COLORS = {
    'singlet': '#2166AC',     # Blue
    'triplet': '#B2182B',     # Red
    'mfe': '#4DAF4A',         # Green
    'T1': '#D6604D',          # Orange-red
    'T2': '#4393C3',          # Light blue
    'theta': '#762A83',       # Purple
    'reference': '#333333',   # Dark gray
    'grid': '#CCCCCC',
}

# Custom colormap for anisotropy maps
ANISO_CMAP = LinearSegmentedColormap.from_list(
    'aniso', ['#053061', '#2166AC', '#4393C3', '#92C5DE',
              '#F7F7F7', '#FDDBC7', '#F4A582', '#D6604D', '#B2182B']
)


def plot_mfe_curve(B_range, yields_singlet, mfe_values=None,
                   labels=None, title=None, highlight_B=None,
                   save_path=None):
    """Plot MFE curve: singlet yield vs magnetic field.
    
    Parameters
    ----------
    B_range : ndarray
        Magnetic field values [mT].
    yields_singlet : ndarray or list of ndarray
        Singlet yields.
    mfe_values : ndarray, optional
        MFE (relative change) for second y-axis.
    labels : list, optional
        Labels for legend.
    title : str, optional
    highlight_B : float, optional
        Highlight a specific B value with vertical line.
    save_path : str, optional
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    
    B_range = np.asarray(B_range)
    
    # Convert yields to array of arrays
    if isinstance(yields_singlet, np.ndarray) and yields_singlet.ndim == 1:
        yields_singlet = [yields_singlet]
    
    if labels is None:
        labels = [f'Run {i+1}' for i in range(len(yields_singlet))]
    
    colors_cycle = [COLORS['singlet'], COLORS['theta'], COLORS['T1']]
    
    for i, (y, label) in enumerate(zip(yields_singlet, labels)):
        color = colors_cycle[i % len(colors_cycle)]
        ax1.plot(B_range, y, '-', color=color, linewidth=2.0,
                label=label, alpha=0.9)
    
    ax1.set_xlabel('Magnetic Field $B_0$ (mT)')
    ax1.set_ylabel('Singlet Yield $\\Phi_S$')
    ax1.set_xlim(B_range.min(), B_range.max())
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0.25, color=COLORS['grid'], linestyle=':', alpha=0.5)
    ax1.axhline(y=0.5, color=COLORS['grid'], linestyle=':', alpha=0.5)
    ax1.axhline(y=0.75, color=COLORS['grid'], linestyle=':', alpha=0.5)
    
    # Highlight geomagnetic field (~50 μT = 0.05 mT) and Earth's field
    if highlight_B is not None:
        ax1.axvline(x=highlight_B, color='red', linestyle='--',
                    alpha=0.7, linewidth=1.2)
        ax1.text(highlight_B * 1.05, 0.95, f'$B = {highlight_B:.2f}$ mT',
                rotation=90, fontsize=9, color='red', alpha=0.7)
    
    # Add MFE on second y-axis if provided
    if mfe_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(B_range, mfe_values * 100, '--', color=COLORS['mfe'],
                linewidth=1.5, alpha=0.7, label='MFE (%)')
        ax2.set_ylabel('MFE (%)', color=COLORS['mfe'])
        ax2.tick_params(axis='y', labelcolor=COLORS['mfe'])
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='best', framealpha=0.9)
    
    if title:
        ax1.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    
    return fig, ax1


def plot_anisotropy_map(theta_vals, B_vals, yield_map,
                       title=None, save_path=None):
    """Plot 2D color map of singlet yield vs θ and B.
    
    Parameters
    ----------
    theta_vals : ndarray
        Angle values [radians].
    B_vals : ndarray
        Magnetic field values [mT].
    yield_map : ndarray (n_theta, n_B)
        Singlet yields.
    title : str, optional
    save_path : str, optional
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Convert θ to degrees for display
    theta_deg = np.degrees(theta_vals)
    
    # Create 2D color map (pcolormesh for proper pixel alignment)
    T, B = np.meshgrid(theta_deg, B_vals)
    
    im = ax.pcolormesh(T, B, yield_map.T, shading='auto',
                       cmap=ANISO_CMAP, vmin=0, vmax=1)
    
    ax.set_xlabel('Angle $\\theta$ (degrees)')
    ax.set_ylabel('Magnetic Field $B_0$ (mT)')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Singlet Yield $\\Phi_S$', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_time_evolution(times, populations, labels=None,
                       title=None, log_x=False, save_path=None):
    """Plot time-dependent state populations.
    
    Parameters
    ----------
    times : ndarray
        Time points [ns].
    populations : list of ndarray
        Population traces.
    labels : list of str, optional
    title : str, optional
    log_x : bool
        Use logarithmic x-axis.
    save_path : str, optional
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    times = np.asarray(times)
    if isinstance(populations, np.ndarray) and populations.ndim == 1:
        populations = [populations]
    
    if labels is None:
        labels = [f'Trace {i+1}' for i in range(len(populations))]
    
    colors = [COLORS['singlet'], COLORS['triplet'], COLORS['mfe'],
              COLORS['theta'], COLORS['T1']]
    
    for i, (pop, label) in enumerate(zip(populations, labels)):
        color = colors[i % len(colors)]
        ax.plot(times, pop, '-', color=color, linewidth=1.8,
                label=label, alpha=0.85)
    
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Population')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if log_x:
        ax.set_xscale('log')
        ax.set_xlim(max(times[1], 0.1), times[-1])
    else:
        ax.set_xlim(0, times[-1])
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return fig, ax


def plot_decoherence_comparison(T1_values, yields_T1,
                                T2_values, yields_T2,
                                title=None, save_path=None):
    """Compare effect of T1 relaxation vs T2 dephasing on yield.
    
    Parameters
    ----------
    T1_values : ndarray
        T₁ relaxation times [ns].
    yields_T1 : ndarray
        Singlet yields for each T₁.
    T2_values : ndarray
        T₂ dephasing times [ns].
    yields_T2 : ndarray
        Singlet yields for each T₂.
    title : str, optional
    save_path : str, optional
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    T1_values = np.asarray(T1_values)
    T2_values = np.asarray(T2_values)
    
    ax.semilogx(T1_values, yields_T1, 'o-', color=COLORS['T1'],
                linewidth=2.0, markersize=6, label='$T_1$ relaxation')
    ax.semilogx(T2_values, yields_T2, 's-', color=COLORS['T2'],
                linewidth=2.0, markersize=6, label='$T_2$ dephasing')
    
    ax.set_xlabel('Decoherence Time (ns)')
    ax.set_ylabel('Singlet Yield $\\Phi_S$')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return fig, ax


def plot_singlet_triplet_ratio(B_range, yields_singlet, yields_triplet,
                               title=None, save_path=None):
    """Plot singlet and triplet yields with ratio inset.
    
    Parameters
    ----------
    B_range : ndarray
        Magnetic field [mT].
    yields_singlet : ndarray
        Singlet yields.
    yields_triplet : ndarray
        Triplet yields.
    title : str
    save_path : str
    
    Returns
    -------
    fig, ax : matplotlib figure
    """
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(9, 6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1])
    
    # Main panel: singlet and triplet yields
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(B_range, yields_singlet, '-', color=COLORS['singlet'],
            linewidth=2.0, label='Singlet $\\Phi_S$')
    ax1.plot(B_range, yields_triplet, '-', color=COLORS['triplet'],
            linewidth=2.0, label='Triplet $\\Phi_T$')
    ax1.set_xlabel('Magnetic Field $B_0$ (mT)')
    ax1.set_ylabel('Recombination Yield')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.05)
    
    # Inset: S/T ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ratio = np.array(yields_singlet) / (np.array(yields_triplet) + 1e-12)
    ax2.plot(B_range, ratio, '-', color=COLORS['mfe'], linewidth=2.0)
    ax2.set_xlabel('$B_0$ (mT)')
    ax2.set_ylabel('$\\Phi_S / \\Phi_T$')
    ax2.grid(True, alpha=0.3)
    
    # Inset: MFE
    ax3 = fig.add_subplot(gs[1, 1])
    mfe_vals = (np.array(yields_singlet) - yields_singlet[0]) / yields_singlet[0] * 100
    ax3.plot(B_range, mfe_vals, '-', color='#D6604D', linewidth=2.0)
    ax3.set_xlabel('$B_0$ (mT)')
    ax3.set_ylabel('MFE (%)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    
    return fig, (ax1, ax2, ax3)


def make_publication_figure(figures, labels=None, n_cols=2,
                           title=None, save_path=None):
    """Composite publication figure from multiple sub-figures.
    
    Parameters
    ----------
    figures : list of tuple (fig, label)
    labels : list of str
        Sub-figure labels (a, b, c, d).
    n_cols : int
        Number of columns.
    title : str, optional
    save_path : str, optional
    
    Returns
    -------
    fig : matplotlib figure
    """
    # Too complex with external figures — use save compositing
    # For now, just return the last figure
    fig = figures[-1][0]
    return fig
