# FIRE synchronization benchmark

Reference commit: `5540d893a849db9772081229a86d5f68537ad037`.
Hardware/software: NVIDIA H200, PyTorch 2.10.0+cu128, MACE-MP-0 small
(`2023-12-10-mace-128-L0_energy_epoch-249.model`), float32, eager execution,
no cuEquivariance acceleration. Four CPU threads; GPU 0 otherwise idle.

The input is eight independently rattled 32-atom FCC copper cells (256 atoms,
0.1 Å displacement, seed 0). This avoids downloading WBM and emphasizes overhead
on a small workload. These results should not be generalized to large batches or
other models without measuring them.

## End-to-end relaxation

All performance numbers below come from
`examples/benchmarking/opt-throughput.py`. Five alternating old/new calls to its
`run_torchsim_optimization` function in one process gave **4.710 → 4.582 seconds**,
a **2.8% throughput gain**. All eight
structures converged in every run (`f_max=0.05`, at most 100 steps, fixed cells).
Raw results: `fire-workflow-paired.json`. The optimizer registry was switched
between the saved original FIRE functions and the new functions for each run;
the model, starting structures and all settings were shared. Both implementations
received ten warmup steps, and no tests or other benchmarks ran concurrently.

The first separate-process baseline and rerun produced medians of 5.370 and 4.665
seconds (the two `opt-mace-fire_*.csv` files). That larger apparent gain did not
persist in the alternating comparison. Use the approximately 3% controlled gain
as the conclusion, rather than the separate-process difference.

## Reproduction

Run the existing relaxation-throughput benchmark before and after editing FIRE:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  examples/benchmarking/opt-throughput.py --optimizer fire --dataset copper \
  --n-structures 8 --dtype float32 --cell-filter none --max-steps 100 \
  --repeats 5 --skip-ase
```

The throughput benchmark includes initialization, convergence checks and
in-flight autobatching; it excludes model loading and ten warmup steps.

## Validation

The optimizer/state/math suites and strict FIRE-versus-ASE trajectory tests passed:
95 tests, with 18 unrelated general ASE comparison cases deselected. Added tests
cover the initial all-NaN step, mixed-sign powers, ASE/VV counter ordering, known
batched-dot output sizes, and dispatch guards rejecting scalar reads and boolean
indexing from FIRE with model evaluation stubbed out.

```bash
.venv/bin/python -m pytest tests/test_optimizers.py tests/test_optimizer_states.py \
  tests/test_math.py tests/test_optimizers_vs_ase.py \
  -k 'not optimizer_vs_ase_parametrized and not bfgs_vs_ase_parametrized and not lbfgs_vs_ase_parametrized' -q
```

Ruff checks and formatting passed for all changed Python files.

A separate float64 CPU comparison against the saved original implementation
matched at `rtol=atol=1e-12` after every step for 30 steps, across ASE/VV FIRE and
fixed/unit/Fréchet cells, with NaN-velocity replacements at steps 8 and 17.

## ASE step isolation and readability refactor

Replacing only `_ase_fire_step`, with all other code shared, gave these median
end-to-end relaxation times on the same H200/MACE/copper workload:

| ASE step implementation | Median seconds |
| --- | ---: |
| Before synchronization changes (`5540d89`) | 4.5875 |
| After synchronization changes (`2e414a1`) | 4.4515 |
| After extracting velocity mixing and cell updates into helpers | 4.4585 |

Each variant ran six times in one process, with the order rotated and reversed
to cover all six permutations. All eight structures converged in every run.
Each variant received ten warmup steps; tests ran before the benchmark.
Raw results are in `fire-readability-paired.json`.

This supports a modest, approximately 3% throughput gain from the ASE step
changes on this workload. The readability refactor differs by only 0.16% from
the preceding implementation, smaller than the observed run-to-run variation.
These measurements cover fixed cells and do not isolate the benefit of each
individual mask or establish a speedup on CPU or larger batches.

Reproduce from the repository root (requires both reference commits locally):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  examples/benchmarking/fire-step-comparison.py
```

The refactor passed the same 95-test command above. A separate 30-step float64
comparison against `2e414a1` matched at `rtol=atol=1e-12` after every step for
ASE/VV FIRE and fixed/unit/Fréchet cells, including replacements at steps 8 and 17.
