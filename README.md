# Radical Pair Spin Dynamics — QuTiP Simulation Suite

Publication-quality simulation of **radical pair (RP) spin dynamics** using QuTiP 5.3, covering the full pipeline from basic magnetic field effects (MFE) to avian cryptochrome compass models.

## 🔬 What This Does

| Phase | Physics | Key Output |
|-------|---------|------------|
| 1 | Basic MFE (Zeeman + Exchange, no HFC) | Φ_S vs B field |
| 2 | Hyperfine-mediated S–T mixing | HFC strength comparison |
| 3 | Anisotropic HFC angular dependence | 2D θ×B anisotropy map |
| 4 | Decoherence (T₁/T₂ relaxation) | Critical decoherence thresholds |
| 5 | Cryptochrome compass (FADH•–Trp•) | Angular sensitivity at 50 μT |
| 6 | Publication composite figure | 2×2 Nature-style figure |

## 🧮 Core Physics

The radical pair Hamiltonian:

```
H = H_Z + H_HFC + H_Ex + H_Dip
```

Non-trace-preserving Liouvillian (key innovation — standard `mesolve` doesn't work for RP recombination):

```
L[ρ] = -i[H, ρ] - (k_S/2){P_S, ρ} - (k_T/2){P_T, ρ}

Φ_S = k_S ∫₀^∞ Tr[P_S · ρ(t)] dt
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install qutip numpy scipy matplotlib

# Run all 6 phases
python demo_comprehensive.py
```

Output: 10 figures (PNG + PDF) in `figures/`

## 📁 Structure

```
├── demo_comprehensive.py     # 6-phase orchestrator
├── rp_sim/
│   ├── hamiltonian.py        # Zeeman, HFC, Exchange, Dipolar
│   ├── solver.py             # Non-trace-preserving Liouvillian solver
│   ├── decoherence.py        # T₁/T₂ Lindblad dissipators
│   ├── anisotropy.py         # Angular dependence + Cryptochrome model
│   └── visualization.py      # Publication-quality plotting
└── figures/                  # Generated output (10 files)
```

## 📊 Key Results

- **Phase 1:** Φ_S(B=0) = 1.005 → confirms singlet-selective recombination
- **Phase 2:** HFC reduces Φ_S to 0.91 (weak) / 0.93 (strong) — HFC drives S–T mixing
- **Phase 3:** Angular variation ΔΦ_S/⟨Φ_S⟩ = 2–12% depending on B field
- **Phase 4:** T₁ critical threshold ~10 ns; T₂ threshold ~100 μs
- **Phase 5:** Anisotropic HFC → 31% angular contrast at geomagnetic field (50 μT)

## 🔗 References

1. Hore & Mouritsen, *The Radical Pair Mechanism and the Avian Chemical Compass*, [arXiv:1502.00671](https://arxiv.org/abs/1502.00671)
2. Kattnig & Hore, *How quantum is radical pair magnetoreception?*, [Faraday Discuss. (2020)](https://pubs.rsc.org/en/content/articlelanding/2020/fd/c9fd00049f)
3. *Simulating spin biology using a digital quantum computer*, [arXiv:2406.12986](https://arxiv.org/abs/2406.12986) — next-step target (quantum circuit implementation)

## 📜 License

MIT
