"""Compare historical and working-tree ASE FIRE steps on the same MACE workload.

Run from the repository root. Requires CUDA and MACE. Only _ase_fire_step changes
between variants; initialization, workflow, model, and math helpers are shared.
"""

# ruff: noqa: INP001, S607, SLF001

import importlib
import importlib.util
import json
import statistics
import subprocess
import types
from pathlib import Path

import torch

import torch_sim as ts


def load_source(name, source):
    """Load a historical implementation without changing the checkout."""
    module = types.ModuleType(name)
    exec(compile(source, name, "exec"), module.__dict__)  # noqa: S102
    return module


fire = importlib.import_module("torch_sim.optimizers.fire")
old = load_source(
    "old_fire",
    subprocess.check_output(
        ["git", "show", "5540d89:torch_sim/optimizers/fire.py"],
        text=True,
    ),
)
before = load_source(
    "before_fire",
    subprocess.check_output(
        ["git", "show", "2e414a1:torch_sim/optimizers/fire.py"],
        text=True,
    ),
)
variants = {
    "old_ase_step": old._ase_fire_step,
    "current_ase_step": before._ase_fire_step,
    "refactored_ase_step": fire._ase_fire_step,
}
spec = importlib.util.spec_from_file_location(
    "benchmark", "examples/benchmarking/opt-throughput.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
torch.set_num_threads(4)
model, _, scaling = bench.load_model("mace", None, torch.device("cuda:0"), torch.float32)
base = bench._structures_to_sim_state(
    bench._copper_structures(8, 0), torch.float32, model.device
)
for step in variants.values():
    fire._ase_fire_step = step
    state = ts.fire_init(base.clone(), model)
    for _ in range(10):
        ts.fire_step(state, model)

results = {label: [] for label in variants}
labels = list(variants)
for repeat in range(6):
    order = labels[repeat % 3 :] + labels[: repeat % 3]
    if repeat >= 3:
        order = list(reversed(order))
    for label in order:
        fire._ase_fire_step = variants[label]
        metrics = bench.run_torchsim_optimization(
            base.clone(), model, scaling, "fire", "none", 100, 0.05, 5000
        )
        results[label].append(metrics)
        print(label, repeat, metrics, flush=True)
medians = {
    label: statistics.median(run["total_s"] for run in runs)
    for label, runs in results.items()
}
report = {
    "comparison": "Only _ase_fire_step is replaced; all other code is shared.",
    "runs": results,
    "median_s": medians,
}
Path("benchmark_results/fire-readability-paired.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print("medians", medians, flush=True)
