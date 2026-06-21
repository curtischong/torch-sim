"""Reproduction + fix verification for torch-sim issue #579.

Just run it:  uv run python repro_579.py

Root cause: ``ts.integrate`` converts ``timestep`` to internal units but used to
forward the Nose-Hoover relaxation time ``tau`` unconverted. In metal units that
factor is ~98x, so asking for tau = 200 fs actually gave tau ~ 2 fs (about one
step) -- a thermostat ~100x too stiff. It survives gentle dynamics but massively
overcorrects a force spike (e.g. a close contact) into float overflow -> NaN.
That is "torch-sim diverges where ASE stays finite under identical settings"
from the report.

On identical structures and across several seeds this prints, for each seed:
  - ASE (SevenNet, Nose-Hoover NVT, physical 2 fs / 200 fs) stays finite
  - torch-sim with the OLD unconverted tau diverges on the planted system
  - torch-sim via the fixed ``ts.integrate`` (tau in ps, now converted) is finite
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from ase import Atoms
from ase import units as aseu
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import sevenn
import sevenn.util
from sevenn.calculator import SevenNetCalculator

import torch_sim as ts
from torch_sim.integrators import Integrator, nvt_nose_hoover_init, nvt_nose_hoover_step
from torch_sim.models.sevennet import SevenNetModel
from torch_sim.units import MetalUnits

# --- fixed settings (mirror the report) -------------------------------------
CKPT = "sevennet-mf-ompa"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = 4
STEPS = 6
CONTACT = 0.8  # Angstrom: planted hard overlap that makes divergence deterministic
DT_PS = 0.002  # 2 fs
TAU_PS = 0.2  # 200 fs


def make_system(seed: int, *, contact: float | None, box: float = 14.5) -> Atoms:
    """Random amorphous Na16Ta16Cl96; if ``contact`` is set, plant one overlap."""
    rng = np.random.default_rng(seed)
    numbers = [11] * 16 + [73] * 16 + [17] * 96  # Na, Ta, Cl
    pos: list[np.ndarray] = []
    while len(pos) < len(numbers):
        cand = rng.uniform(0.0, box, size=3)
        if pos:
            d = np.array(pos) - cand
            d -= box * np.round(d / box)
            if np.linalg.norm(d, axis=1).min() < 2.26:
                continue
        pos.append(cand)
    positions = np.array(pos)
    if contact is not None:
        positions[1] = positions[0] + np.array([contact, 0.0, 0.0])
    return Atoms(numbers=numbers, positions=positions, cell=[box] * 3, pbc=True)


def ase_finite(atoms: Atoms, calc: SevenNetCalculator, seed: int) -> bool:
    """ASE Nose-Hoover NVT at physical 2 fs / 200 fs; return whether it stays finite."""
    a = atoms.copy()
    a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=300, rng=np.random.default_rng(seed))
    dyn = NoseHooverChainNVT(a, timestep=2.0 * aseu.fs, temperature_K=300,
                             tdamp=200 * aseu.fs)
    for _ in range(STEPS):
        dyn.run(1)
        if not np.isfinite(a.get_positions()).all():
            return False
    return True


def torchsim_buggy_bad(systems: list[Atoms], model, seed: int) -> list[bool]:
    """Pre-fix path: tau forwarded UNCONVERTED (interpreted in internal units)."""
    torch.manual_seed(seed)
    state = ts.io.atoms_to_state(systems, device=DEV, dtype=torch.float32)
    dt = torch.tensor(DT_PS * float(MetalUnits.time))
    kT = torch.tensor(300.0 * float(MetalUnits.temperature))
    s = nvt_nose_hoover_init(state, model, kT=kT, dt=dt, tau=TAU_PS)  # bug: raw tau
    for _ in range(STEPS):
        s = nvt_nose_hoover_step(s, model, dt=dt, kT=kT)
    return ts.runners.nonfinite_systems(s).tolist()


def torchsim_fixed_bad(systems: list[Atoms], model, seed: int) -> list[bool]:
    """Fixed path: ts.integrate converts tau (ps) to internal units like timestep."""
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = ts.integrate(
            system=systems, model=model, integrator=Integrator.nvt_nose_hoover,
            n_steps=STEPS, temperature=300.0, timestep=DT_PS,
            init_kwargs={"tau": TAU_PS}, pbar=False,
        )
    return ts.runners.nonfinite_systems(final).tolist()


def main() -> None:
    print(f"sevenn={sevenn.__version__} torch={torch.__version__} device={DEV}")
    print(f"SevenNet; NVT Nose-Hoover, 300 K, 2 fs, tau=200 fs, {STEPS} steps, "
          f"contact={CONTACT} A\n")

    cp = sevenn.util.load_checkpoint(CKPT)
    raw = cp.build_model()
    raw.set_is_batch_data(True)
    model = SevenNetModel(model=raw.to(DEV), modal="omat24", device=DEV,
                          dtype=torch.float32)
    ase_calc = SevenNetCalculator(model=CKPT, modal="omat24", device=str(DEV))

    n_ok = 0
    for seed in range(SEEDS):
        clean = make_system(2 * seed, contact=None)
        planted = make_system(2 * seed + 1, contact=CONTACT)
        systems = [clean, planted]  # system 0 clean, system 1 planted

        ase_ok = ase_finite(planted, ase_calc, seed)
        buggy = torchsim_buggy_bad(systems, model, seed)
        fixed = torchsim_fixed_bad(systems, model, seed)

        # Expected: ASE finite; buggy diverges only the planted (sys 1); fix all finite
        ok = ase_ok and buggy == [False, True] and fixed == [False, False]
        n_ok += ok
        print(f"seed={seed}: ASE_planted_finite={ase_ok}  "
              f"torchsim_buggy_diverged={[i for i, b in enumerate(buggy) if b]}  "
              f"torchsim_fixed_diverged={[i for i, b in enumerate(fixed) if b]}  "
              f"-> {'OK' if ok else 'UNEXPECTED'}")

    print(f"\n{n_ok}/{SEEDS} seeds matched the expected pattern:")
    print("  - ASE stays finite (the issue #579 control)")
    print("  - pre-fix torch-sim (unconverted tau) diverges where ASE does not")
    print("  - fixed torch-sim (tau converted in integrate) matches ASE: finite")


if __name__ == "__main__":
    main()
