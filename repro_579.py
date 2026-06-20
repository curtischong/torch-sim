"""Reproduce torch-sim issue #579.

Batched NVT Nose-Hoover MD with SevenNet produces NaN energy/temperature in some
batch elements while others stay finite, with no diagnostic warning.

https://github.com/TorchSim/torch-sim/issues/579

Setup mirrors the report:
  - 16 independent amorphous Na16Ta16Cl96 structures (128 atoms each)
  - fixed-cell NVT, Nose-Hoover chain thermostat, 300 K
  - dt = 2 fs, thermostat damping tau = 200 fs
  - 200 warmup + 1000 measured steps
  - SevenNet (mf-ompa checkpoint, modal="omat24"); optional D3(BJ)-PBE
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_masses

import sevenn
import sevenn.util

import torch_sim as ts
from torch_sim.integrators import Integrator
from torch_sim.models.interface import SumModel
from torch_sim.models.sevennet import SevenNetModel

from d3_params import build_pbe_d3_model


def make_amorphous(seed: int, min_dist: float = 2.26, box: float = 14.5,
                   strain: float = 0.0) -> Atoms:
    """Random amorphous Na16Ta16Cl96 (NaTaCl6) via rejection sampling."""
    rng = np.random.default_rng(seed)
    numbers = [11] * 16 + [73] * 16 + [17] * 96  # Na, Ta, Cl
    pos: list[np.ndarray] = []
    while len(pos) < len(numbers):
        cand = rng.uniform(0.0, box, size=3)
        if pos:
            d = np.array(pos) - cand
            d -= box * np.round(d / box)  # min-image
            if np.linalg.norm(d, axis=1).min() < min_dist:
                continue
        pos.append(cand)
    scale = 1.0 - strain  # isotropic compression -> denser, more strained
    return Atoms(
        numbers=numbers,
        positions=np.array(pos) * scale,
        cell=[box * scale, box * scale, box * scale],
        pbc=True,
    )


def min_nn(atoms: Atoms) -> float:
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-systems", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-d3", action="store_true", help="disable D3(BJ)-PBE")
    ap.add_argument("--min-dist", type=float, default=2.26)
    ap.add_argument("--box", type=float, default=14.5)
    ap.add_argument("--strain", type=float, default=0.0,
                    help="isotropic compression fraction (denser quench)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    torch.manual_seed(args.seed)

    print(f"sevenn={sevenn.__version__} torch={torch.__version__} device={device}")

    # Graded strain: system 0 is relaxed (stable alone), the last is most
    # compressed (unstable alone). Lets us test cross-system NaN contamination.
    strains = [args.strain * i / max(args.n_systems - 1, 1)
               for i in range(args.n_systems)]
    structures = [
        make_amorphous(seed=i, min_dist=args.min_dist, box=args.box, strain=s)
        for i, s in enumerate(strains)
    ]
    nn = [min_nn(a) for a in structures]
    print(f"built {len(structures)} structures, min-NN range "
          f"{min(nn):.2f}-{max(nn):.2f} Angstrom (graded strain)")

    cp = sevenn.util.load_checkpoint("sevennet-mf-ompa")
    raw = cp.build_model()
    raw.set_is_batch_data(True)
    sevennet = SevenNetModel(model=raw.to(device), modal="omat24",
                             device=torch.device(device), dtype=dtype)
    if args.no_d3:
        model = sevennet
        print("model: SevenNet (no D3)")
    else:
        d3 = build_pbe_d3_model(device=device, dtype=dtype, compute_stress=True)
        model = SumModel(sevennet, d3)
        print("model: SevenNet + D3(BJ)-PBE")

    n_steps = args.warmup + args.steps
    print(f"running batched nvt_nose_hoover: {n_steps} steps, dt=2fs, 300K, tau=200fs")
    final = ts.integrate(
        system=structures,
        model=model,
        integrator=Integrator.nvt_nose_hoover,
        n_steps=n_steps,
        temperature=300.0,
        timestep=0.002,  # ps == 2 fs
        init_kwargs={"tau": 0.2},  # ps == 200 fs (thermostat damping)
        pbar=True,
    )

    # Per-system diagnostics
    e = final.energy.detach().cpu()
    temps = ts.calc_kT(
        masses=final.masses, momenta=final.momenta, system_idx=final.system_idx
    ).detach().cpu() / ts.units.MetalUnits.temperature
    pos_finite = torch.zeros(final.n_systems, dtype=torch.bool)
    for i in range(final.n_systems):
        mask = final.system_idx == i
        pos_finite[i] = torch.isfinite(final.positions[mask]).all()

    print("\nper-system final state:")
    print(f"{'sys':>3} {'energy(eV)':>14} {'T(K)':>10} {'pos_finite':>11}")
    n_bad = 0
    for i in range(final.n_systems):
        e_nan = not torch.isfinite(e[i])
        t_nan = not torch.isfinite(temps[i])
        bad = e_nan or t_nan or not pos_finite[i]
        n_bad += bad
        flag = "  <-- NaN" if bad else ""
        print(f"{i:>3} {e[i].item():>14.3f} {temps[i].item():>10.1f} "
              f"{str(bool(pos_finite[i])):>11}{flag}")

    print(f"\n{n_bad}/{final.n_systems} systems ended with NaN "
          f"(energy/temperature/positions).")
    if n_bad:
        print("REPRODUCED: batched MD silently returned partial-batch NaNs "
              "(no warning/exception was raised).")
    else:
        print("No NaNs this run; try more steps or a different --seed.")


if __name__ == "__main__":
    main()
