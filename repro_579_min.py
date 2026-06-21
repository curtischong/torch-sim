"""Minimal reproduction of torch-sim issue #579 (Nose-Hoover `tau` units).

Run:  uv run python repro_579_min.py

The reporter saw batched NVT MD return NaN energy/temperature on physically
reasonable amorphous structures (real packmol NaTaCl6, min pair distance
~2.26 A, no close contacts) and confirmed it happens with D3 *off* and with a
different model (MatterSim) -- so it is an integrator bug, not a potential bug.

Root cause: ``ts.integrate`` converts ``timestep`` to internal units but used
to forward the Nose-Hoover relaxation time ``tau`` unconverted. The user passed
``tau = 200 fs = 0.2 ps`` and ``timestep = 2 fs = 0.002 ps`` (same convention).
In metal units time is scaled by ~98x, so ``dt`` became ~0.196 while ``tau``
stayed 0.2 -- i.e. ``tau ~ dt``. A Nose-Hoover chain with relaxation time of one
step has a near-zero thermostat mass and is numerically unstable: it blows up to
NaN over a normal trajectory, NO close contact required.

This script reproduces that with a cheap Lennard-Jones potential (the bug is
model-independent) on an author-equivalent structure: NaTaCl6, 128 atoms, box
15.62 A, random packing rejected below 2.26 A -- the same geometry the reporter
used, minus any planted overlap. It prints:
  - buggy path (raw tau, pre-fix behavior)      -> diverges to NaN
  - fixed path (ts.integrate, tau converted)     -> stays finite
"""

from __future__ import annotations

import numpy as np
import torch
from ase import Atoms

import torch_sim as ts
from torch_sim.integrators import nvt_nose_hoover_init, nvt_nose_hoover_step
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.units import MetalUnits

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
BOX = 15.62  # A, matches the reporter's cell
MIN_DIST = 2.26  # A, matches the reporter's min pair distance (packmol tol 2 A)
DT_PS = 0.002  # 2 fs
TAU_PS = 0.2  # 200 fs
STEPS = 200


def author_like_structure(seed: int = 11001) -> Atoms:
    """Random NaTaCl6 (128 atoms) at the reporter's density, no close contacts."""
    rng = np.random.default_rng(seed)
    numbers = [11] * 16 + [73] * 16 + [17] * 96  # Na, Ta, Cl
    pos: list[np.ndarray] = []
    while len(pos) < len(numbers):
        cand = rng.uniform(0.0, BOX, size=3)
        if pos:
            d = np.array(pos) - cand
            d -= BOX * np.round(d / BOX)  # minimum image
            if np.linalg.norm(d, axis=1).min() < MIN_DIST:
                continue
        pos.append(cand)
    return Atoms(numbers=numbers, positions=np.array(pos), cell=[BOX] * 3, pbc=True)


def finite(state: ts.state.SimState) -> bool:
    return not bool(ts.runners.nonfinite_systems(state).any())


def main() -> None:
    atoms = author_like_structure()
    # sigma > the 2.26 A spacing puts every ion in the repulsive wall, i.e. strong
    # forces everywhere -- a stand-in for the reporter's dense (2.9 g/cm^3) ionic
    # melt. The divergence is driven by distributed forces, not a single overlap.
    model = LennardJonesModel(
        sigma=3.0, epsilon=0.3, cutoff=7.5, device=DEV, dtype=DTYPE
    )
    dt = torch.tensor(DT_PS * float(MetalUnits.time))
    kT = torch.tensor(300.0 * float(MetalUnits.temperature))

    print(f"device={DEV}  box={BOX} A  min_dist>={MIN_DIST} A  "
          f"timestep=2 fs  tau=200 fs  steps={STEPS}\n")

    # --- buggy path: tau forwarded UNCONVERTED (interpreted in internal units) ---
    torch.manual_seed(0)
    s = ts.io.atoms_to_state([atoms], device=DEV, dtype=DTYPE)
    s = nvt_nose_hoover_init(s, model, kT=kT, dt=dt, tau=TAU_PS)  # raw 0.2 -> tau~dt
    for _ in range(STEPS):
        s = nvt_nose_hoover_step(s, model, dt=dt, kT=kT)
    buggy_ok = finite(s)

    # --- fixed path: ts.integrate converts tau (ps) like timestep ---------------
    torch.manual_seed(0)
    final = ts.integrate(
        system=[atoms], model=model, integrator=ts.Integrator.nvt_nose_hoover,
        n_steps=STEPS, temperature=300.0, timestep=DT_PS,
        init_kwargs={"tau": TAU_PS}, pbar=False,
    )
    fixed_ok = finite(final)

    print(f"buggy (raw tau,  tau~dt)        finite = {buggy_ok}   "
          f"<- expected False (diverges to NaN)")
    print(f"fixed (ts.integrate converts tau) finite = {fixed_ok}   "
          f"<- expected True  (stays finite)")
    temp = float(final.calc_temperature().flatten()[0])
    print(f"\nfixed tau internal = {float(final.chain.tau.flatten()[0]):.3f} "
          f"(= 0.2 * {float(MetalUnits.time):.2f}); buggy used 0.2 ~ dt "
          f"{float(dt):.3f}")
    print(f"fixed final temperature = {temp:.1f} K (finite and bounded)")
    print("\nREPRODUCED" if (not buggy_ok and fixed_ok) else "\nNOT REPRODUCED")


if __name__ == "__main__":
    main()
