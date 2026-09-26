"""Compare fixed-work FIRE timings with a small MACE model on CUDA.

Example (save the reference before editing FIRE):
    git show HEAD:torch_sim/optimizers/fire.py > /tmp/fire_original.py
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python \
        examples/benchmarking/fire-steps.py --reference-fire /tmp/fire_original.py

Runs seven repetitions of 100 steps on eight rattled 32-atom copper cells,
including model evaluation. Alternates reference/current order, synchronizes CUDA
at timing boundaries, profiles separately, and compares final states. Model
loading, initialization and warmup are excluded from fixed-step timings.
"""

# ruff: noqa: INP001

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

import torch_sim as ts


def load_module(name, path):
    """Load a benchmark helper or a saved reference implementation."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(base, model, steps, repeats, flavor, cell_filter, implementations):  # noqa: C901
    """Time identical work in alternating order and collect separate profiles."""

    def init():
        return ts.fire_init(
            base.clone(), model, fire_flavor=flavor, cell_filter=cell_filter
        )

    for step in implementations.values():
        state = init()
        for _ in range(20):
            step(state, model, fire_flavor=flavor)

    times = {label: [] for label in implementations}
    snapshots = {}
    for repeat in range(repeats):
        order = list(implementations)
        if repeat % 2:
            order.reverse()
        for label in order:
            step = implementations[label]
            state = init()
            torch.cuda.synchronize(model.device)
            start = time.perf_counter()
            for _ in range(steps):
                step(state, model, fire_flavor=flavor)
            torch.cuda.synchronize(model.device)
            times[label].append((time.perf_counter() - start) * 1000 / steps)
            snapshots[label] = {
                name: getattr(state, name).detach().cpu()
                for name in (
                    "positions",
                    "cell",
                    "energy",
                    "forces",
                    "velocities",
                    "dt",
                    "alpha",
                    "n_pos",
                )
            }

    differences = {}
    if "before" in snapshots:
        for name, before in snapshots["before"].items():
            differences[name] = (before - snapshots["after"][name]).abs().max().item()

    counts = {}
    for label, step in implementations.items():
        state = init()
        for _ in range(20):
            step(state, model, fire_flavor=flavor)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            for _ in range(3):
                step(state, model, fire_flavor=flavor)
            torch.cuda.synchronize(model.device)
        counts[label] = {
            event.key: event.count / 3
            for event in prof.key_averages()
            if any(
                term in event.key
                for term in (
                    "item",
                    "_local_scalar_dense",
                    "nonzero",
                    "cudaStreamSynchronize",
                    "unique",
                )
            )
        }
    medians = {label: statistics.median(values) for label, values in times.items()}
    return {
        "ms_per_step": times,
        "median_ms": medians,
        "speedup": medians["before"] / medians["after"] if "before" in medians else None,
        "max_abs_difference": differences,
        "events_per_step": counts,
    }


def main():
    """Run fixed-step benchmarks and save raw measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-fire", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark_results/fire-paired.json")
    )
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    torch.set_num_threads(4)
    bench = load_module("opt_bench", Path(__file__).with_name("opt-throughput.py"))
    model, _, _ = bench.load_model("mace", None, torch.device("cuda:0"), torch.float32)
    base = bench._structures_to_sim_state(  # noqa: SLF001
        bench._copper_structures(8, 0),  # noqa: SLF001
        torch.float32,
        model.device,
    )
    implementations = {}
    if args.reference_fire:
        implementations["before"] = load_module(
            "fire_reference", args.reference_fire
        ).fire_step
    implementations["after"] = ts.fire_step
    results = {}
    for flavor, cell_filter in (
        ("ase_fire", None),
        ("ase_fire", ts.CellFilter.unit),
        ("ase_fire", ts.CellFilter.frechet),
        ("vv_fire", None),
    ):
        key = f"{flavor}/{cell_filter}"
        results[key] = run_case(
            base, model, args.steps, args.repeats, flavor, cell_filter, implementations
        )
        print(key, json.dumps(results[key]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
