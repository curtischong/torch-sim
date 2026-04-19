# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ase",
#   "matbench-discovery",
#   "numpy",
#   "pandas",
#   "pymatgen",
#   "torch",
# ]
# ///
"""Autobatcher throughput benchmark on WBM initial structures.

Compares torch-sim's autobatching strategies for geometry optimization:
- ``no_batcher``: ``optimize(autobatcher=False)`` — everything in one batch,
  converged structures are still hot-swapped out.
- ``in_flight``: ``InFlightAutoBatcher`` with auto-estimated memory scaler —
  swaps converged states for pending ones to maximize GPU utilization.
- ``binning``: ``BinningAutoBatcher`` packs states into fixed bins, and each
  bin is optimized to completion before the next starts (no hot-swap).

Results are saved to benchmark_results/autobatcher-<model>-<optimizer>-<ts>.csv.

Example:
    uv run --with ".[mace]" examples/benchmarking/autobatcher-throughput.py \
        --model mace --optimizer lbfgs --n-structures 100
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from torch_sim.state import SimState
    from torch_sim.typing import MemoryScaling

import numpy as np
import pandas as pd
import torch


MAX_ATOMS = {"mace": 5000}
STRATEGIES = ("no_batcher", "in_flight", "binning")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Autobatcher throughput benchmark on WBM structures."
    )
    parser.add_argument("--model", choices=["mace"], default="mace")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to model checkpoint. Uses MACE-MP-0 small if omitted.",
    )
    parser.add_argument("--optimizer", choices=["lbfgs", "fire"], default="lbfgs")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
        help="Autobatcher strategies to benchmark.",
    )
    parser.add_argument("--n-structures", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--f-max", type=float, default=0.05)
    parser.add_argument(
        "--cell-filter",
        choices=["frechet", "exp", "none"],
        default="frechet",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float64"
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Max atoms per batch (overrides per-model default).",
    )
    parser.add_argument(
        "--steps-between-swaps",
        type=int,
        default=5,
        help="Steps between convergence checks in InFlightAutoBatcher.",
    )
    return parser.parse_args()


def clear_gpu_memory() -> None:
    """Empty CUDA cache and run Python GC."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


def _sample_wbm_structures(n_structures: int, seed: int) -> list[dict[str, Any]]:
    """Fetch random WBM structures as pymatgen dicts."""
    from matbench_discovery.data import DataFiles, ase_atoms_from_zip
    from pymatgen.io.ase import AseAtomsAdaptor

    all_atoms = ase_atoms_from_zip(DataFiles.wbm_initial_atoms.path)
    if n_structures > len(all_atoms):
        raise RuntimeError(
            f"Requested {n_structures} structures but WBM only has {len(all_atoms)}."
        )
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(all_atoms), size=n_structures, replace=False)
    sampled = [all_atoms[int(i)] for i in indices.tolist()]
    adaptor = AseAtomsAdaptor()
    return [adaptor.get_structure(a).as_dict() for a in sampled]


def _structures_to_sim_state(
    structures: list[dict[str, Any]],
    dtype: torch.dtype,
    device: torch.device,
) -> SimState:
    """Convert pymatgen structure dicts to a batched SimState."""
    from pymatgen.core import Structure

    from torch_sim.io import structures_to_state

    return structures_to_state(
        [Structure.from_dict(s) for s in structures],
        device=device,
        dtype=dtype,
    )


def load_model(
    model_type: str,
    model_path: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, MemoryScaling]:
    """Return (torchsim_model, memory_scales_with)."""
    if model_type == "mace":
        from mace.calculators.foundations_models import download_mace_mp_checkpoint

        from torch_sim.models.mace import MaceModel, MaceUrls

        path = model_path or MaceUrls.mace_mp_small
        local_path = download_mace_mp_checkpoint(path)
        model = MaceModel(model=local_path, device=device, dtype=dtype, enable_cueq=False)
        return model, "n_atoms_x_density"

    raise ValueError(f"Unknown model type: {model_type}")


def _count_converged(final_states: list[SimState], model: Any, f_max: float) -> int:
    """Count structures below the force-convergence threshold."""
    converged = 0
    for state in final_states:
        forces = model(state)["forces"]
        if float(torch.linalg.norm(forces, dim=1).max()) <= f_max:
            converged += 1
    return converged


def run_strategy(
    strategy: str,
    sim_state: SimState,
    model: Any,
    memory_scales_with: MemoryScaling,
    optimizer_name: str,
    cell_filter_name: str,
    max_steps: int,
    f_max: float,
    max_atoms: int,
    steps_between_swaps: int,
) -> dict[str, Any]:
    """Run one autobatching strategy and return timing + convergence metrics."""
    import torch_sim as ts
    from torch_sim.autobatching import BinningAutoBatcher, InFlightAutoBatcher
    from torch_sim.optimizers import Optimizer

    optimizer = Optimizer[optimizer_name]
    convergence_fn = ts.generate_force_convergence_fn(force_tol=f_max)
    init_kwargs: dict[str, Any] = {}
    if cell_filter_name != "none":
        init_kwargs["cell_filter"] = ts.CellFilter[cell_filter_name]

    optimize_kwargs = dict(
        model=model,
        optimizer=optimizer,
        max_steps=max_steps,
        convergence_fn=convergence_fn,
        steps_between_swaps=steps_between_swaps,
        init_kwargs=init_kwargs or None,
    )

    n_batches = 1
    t0 = time.perf_counter()
    if strategy == "no_batcher":
        final_state = ts.optimize(system=sim_state, autobatcher=False, **optimize_kwargs)
        final_states = final_state.split()
    elif strategy == "in_flight":
        batcher = InFlightAutoBatcher(
            model=model,
            memory_scales_with=memory_scales_with,
            max_memory_scaler=max_atoms,
        )
        final_state = ts.optimize(
            system=sim_state, autobatcher=batcher, **optimize_kwargs
        )
        final_states = final_state.split()
    elif strategy == "binning":
        batcher = BinningAutoBatcher(
            model=model,
            memory_scales_with=memory_scales_with,
            max_memory_scaler=max_atoms,
        )
        batcher.load_states(sim_state)
        n_batches = len(batcher.index_bins)
        bin_results: list[SimState] = []
        for bin_state, _ in batcher:
            bin_final = ts.optimize(
                system=bin_state, autobatcher=False, **optimize_kwargs
            )
            bin_results.append(bin_final)
        ordered = batcher.restore_original_order(bin_results)
        final_states = [s for group in ordered for s in group.split()]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    elapsed = time.perf_counter() - t0

    n_relaxed = len(final_states)
    converged = _count_converged(final_states, model, f_max)

    return {
        "strategy": strategy,
        "n_batches": n_batches,
        "n_relaxed": n_relaxed,
        "n_converged": converged,
        "converged_pct": round(100 * converged / n_relaxed, 1) if n_relaxed else 0,
        "total_s": round(elapsed, 3),
        "s_per_structure": round(elapsed / n_relaxed, 4) if n_relaxed else 0,
        "structures_per_min": round(n_relaxed / elapsed * 60, 2) if elapsed > 0 else 0,
    }


def _save_and_print(rows: list[dict[str, Any]], tag: str) -> None:
    """Save results CSV and print summary."""
    df = pd.DataFrame(rows)
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_path = Path(f"benchmark_results/{tag}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))
    print(json.dumps(rows, indent=2, default=str))


def main() -> None:
    """Entry point."""
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    max_atoms = args.max_atoms or MAX_ATOMS.get(args.model, 3000)

    print(
        f"Autobatcher throughput: {args.model} / {args.optimizer} "
        f"on {args.n_structures} WBM structures ({device}, {args.dtype})"
    )
    print(f"Strategies: {args.strategies}")

    print("Loading WBM structures...")
    t0 = time.perf_counter()
    structures = _sample_wbm_structures(args.n_structures, args.seed)
    load_s = time.perf_counter() - t0
    print(f"  Loaded {len(structures)} structures in {load_s:.2f}s")

    from pymatgen.core import Structure

    n_atoms_list = [len(Structure.from_dict(s)) for s in structures]
    print(
        f"  Atom count: min={min(n_atoms_list)}, "
        f"max={max(n_atoms_list)}, mean={np.mean(n_atoms_list):.1f}"
    )

    clear_gpu_memory()
    print(f"Loading {args.model} model...")
    model, memory_scales_with = load_model(
        args.model, args.model_path, device, dtype
    )

    rows: list[dict[str, Any]] = []
    common_row = {
        "model": args.model,
        "optimizer": args.optimizer,
        "cell_filter": args.cell_filter,
        "dtype": args.dtype,
        "device": str(device),
        "n_structures": args.n_structures,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "f_max": args.f_max,
        "max_atoms": max_atoms,
        "steps_between_swaps": args.steps_between_swaps,
        "n_atoms_mean": round(float(np.mean(n_atoms_list)), 1),
        "n_atoms_max": max(n_atoms_list),
    }

    for strategy in args.strategies:
        print(f"\n--- strategy: {strategy} ---")
        sim_state = _structures_to_sim_state(structures, dtype=dtype, device=device)
        metrics = run_strategy(
            strategy=strategy,
            sim_state=sim_state,
            model=model,
            memory_scales_with=memory_scales_with,
            optimizer_name=args.optimizer,
            cell_filter_name=args.cell_filter,
            max_steps=args.max_steps,
            f_max=args.f_max,
            max_atoms=max_atoms,
            steps_between_swaps=args.steps_between_swaps,
        )
        print(
            f"  {strategy}: {metrics['n_converged']}/{metrics['n_relaxed']} converged "
            f"— {metrics['structures_per_min']} structures/min "
            f"({metrics['n_batches']} batch(es), {metrics['total_s']}s)"
        )
        rows.append({**common_row, **metrics})
        clear_gpu_memory()

    ref = next((r for r in rows if r["strategy"] == "no_batcher"), rows[0])
    ref_s = ref["s_per_structure"]
    for row in rows:
        row["speedup_vs_" + ref["strategy"]] = (
            round(ref_s / row["s_per_structure"], 2)
            if row["s_per_structure"] > 0
            else None
        )

    del model
    clear_gpu_memory()

    _save_and_print(rows, f"autobatcher-{args.model}-{args.optimizer}")


if __name__ == "__main__":
    main()
