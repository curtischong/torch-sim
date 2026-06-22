"""Minimal reproduction of torch-sim issue #579 (Nose-Hoover `tau` units).

Run:  uv run python repro_579_min.py

The reporter ran batched NVT MD (16 amorphous NaTaCl6 systems, SevenNet-OMNI,
NVT Nose-Hoover, 2 fs, tdamp 200 fs) and got NaN energy/temperature in *some*
batch elements -- with D3 on AND off, and with MatterSim too. ASE stayed finite
under the same physical settings, so it is a torch-sim integrator bug, not a
potential bug.

ROOT CAUSE
    ``ts.integrate`` converts ``timestep`` to internal units but used to forward
    the Nose-Hoover relaxation time ``tau`` unconverted. The reporter passed
    ``tau = 200 fs = 0.2 ps`` and ``timestep = 2 fs = 0.002 ps`` (same
    convention). In metal units time is scaled ~98x, so ``dt`` became ~0.196
    while ``tau`` stayed 0.2 -- i.e. ``tau ~ dt``. A Nose-Hoover chain with
    relaxation time ~= one step has a near-zero thermostat mass and is
    *marginally* unstable: depending on the initial velocity draw, some batch
    systems tip into NaN -- exactly the "some batch elements end with NaN"
    symptom. The fix converts ``tau``/``b_tau`` with the same factor as
    ``timestep`` (torch_sim/runners.py).

Two demos, each comparing BUGGY tau (pre-fix, tau~dt) vs FIXED tau (converted):
  1. Lennard-Jones, single system -- fast, no download, fully deterministic.
  2. SevenNet-OMNI (the reporter's exact model, modal="omat24") on their real 16
     structures, batched -- a 100% faithful repro; downloads a ~100 MB checkpoint
     and reproduces the subset-NaN symptom. Requires sevenn>=0.13.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms

import torch_sim as ts
from torch_sim.integrators import nvt_nose_hoover_init, nvt_nose_hoover_step
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.units import MetalUnits

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
DT_PS = 0.002  # 2 fs
TAU_PS = 0.2  # 200 fs
TIME = float(MetalUnits.time)  # ps -> internal (~98.23)
STRUCT_NPZ = Path(__file__).with_name("repro_579_structures.npz")


def reporter_structures() -> list[Atoms]:
    """The reporter's 16 packmol NaTaCl6 structures (128 atoms, 2.9 g/cm^3)."""
    d = np.load(STRUCT_NPZ)
    numbers = d["numbers"]
    return [
        Atoms(numbers=numbers, positions=p, cell=c, pbc=True)
        for p, c in zip(d["pos"], d["cells"], strict=True)
    ]


def lj_single_structure() -> Atoms:
    """Author-like NaTaCl6 (128 atoms, box 15.62, min spacing 2.26 A, no overlap)."""
    box, rng = 15.62, np.random.default_rng(11001)
    numbers = [11] * 16 + [73] * 16 + [17] * 96
    pos: list[np.ndarray] = []
    while len(pos) < len(numbers):
        cand = rng.uniform(0.0, box, size=3)
        if pos:
            d = np.array(pos) - cand
            d -= box * np.round(d / box)
            if np.linalg.norm(d, axis=1).min() < 2.26:
                continue
        pos.append(cand)
    return Atoms(numbers=numbers, positions=np.array(pos), cell=[box] * 3, pbc=True)


def bad_systems(state: ts.state.SimState) -> list[int]:
    return ts.runners.nonfinite_systems(state).nonzero().flatten().tolist()


def run_integrate(
    systems: list[Atoms], model, *, tau_ps: float, steps: int, seed: int = 0
) -> list[int]:
    torch.manual_seed(seed)  # fix the initial velocity draw (same for buggy vs fixed)
    final = ts.integrate(
        system=systems,
        model=model,
        integrator=ts.Integrator.nvt_nose_hoover,
        n_steps=steps,
        temperature=300.0,
        timestep=DT_PS,
        init_kwargs={"tau": tau_ps, "chain_length": 3, "chain_steps": 1, "sy_steps": 3},
        pbar=False,
    )
    return bad_systems(final)


def lj_demo() -> None:
    print("[1] Lennard-Jones, single system (deterministic, no download)")
    atoms = lj_single_structure()
    # sigma > 2.26 A spacing -> every ion in the repulsive wall (strong forces),
    # a stand-in for the dense ionic melt; divergence is distributed, not a contact.
    model = LennardJonesModel(sigma=3.0, epsilon=0.3, cutoff=7.5, device=DEV, dtype=DTYPE)
    dt = torch.tensor(DT_PS * TIME)
    kT = torch.tensor(300.0 * MetalUnits.temperature)
    # buggy: tau forwarded UNCONVERTED -> 0.2 internal ~= dt
    torch.manual_seed(0)
    s = nvt_nose_hoover_init(
        ts.io.atoms_to_state([atoms], device=DEV, dtype=DTYPE),
        model,
        kT=kT,
        dt=dt,
        tau=TAU_PS,
    )
    for _ in range(200):
        s = nvt_nose_hoover_step(s, model, dt=dt, kT=kT)
    buggy = bad_systems(s)
    # fixed: ts.integrate converts tau like timestep
    fixed = run_integrate([atoms], model, tau_ps=TAU_PS, steps=200)
    print(f"    buggy (tau~dt)   diverged systems = {buggy}  (expect [0])")
    print(f"    fixed (tau*time) diverged systems = {fixed}  (expect [])\n")


def sevennet_demo() -> None:
    print("[2] SevenNet-OMNI (modal=omat24), reporter's 16 real structures, batched")
    from torch_sim.models.sevennet import SevenNetModel

    systems = reporter_structures()
    model = SevenNetModel(model="7net-omni", modal="omat24", device=DEV, dtype=DTYPE)
    # Reporter's MD length (1000 steps). The instability is marginal, so the EXACT
    # diverged set is not bit-reproducible on GPU (SevenNet scatter ops are
    # non-deterministic) -- but the buggy path reliably tips >=1 system to NaN over
    # this length, while the fixed path stays finite. This is the reporter's exact
    # "some batch elements end with NaN" symptom.
    # buggy: reconstruct pre-fix internal tau (0.2) by undoing integrate's conversion
    buggy = run_integrate(systems, model, tau_ps=TAU_PS / TIME, steps=1000)
    fixed = run_integrate(systems, model, tau_ps=TAU_PS, steps=1000)
    print(f"    buggy (tau~dt)   diverged systems = {buggy}  (>=1 system NaNs; set varies)")
    print(f"    fixed (tau*time) diverged systems = {fixed}  (expect [])\n")


def main() -> None:
    print(f"device={DEV}  timestep=2 fs  tau=200 fs  tau/dt(buggy)~1\n")
    lj_demo()
    sevennet_demo()


if __name__ == "__main__":
    main()
