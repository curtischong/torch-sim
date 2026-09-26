# FIRE synchronization benchmark

Reference commit: `5540d893a849db9772081229a86d5f68537ad037`.
Hardware/software: NVIDIA H200, PyTorch 2.10.0+cu128, MACE-MP-0 small
(`2023-12-10-mace-128-L0_energy_epoch-249.model`), float32, eager execution,
no cuEquivariance acceleration. Four CPU threads; GPU 0 otherwise idle.

The input is eight independently rattled 32-atom FCC copper cells (256 atoms,
0.1 Å displacement, seed 0). This avoids downloading WBM and emphasizes overhead
on a small workload. These results should not be generalized to large batches or
other models without measuring them.

## Fixed-work comparison

Model evaluation is included. Each repetition starts from the same initial state
and executes 100 FIRE steps, with seven repetitions per implementation. Old/new
order alternates in one process. Initialization, model loading and warmup are
excluded. Wall-clock timers synchronize CUDA at both boundaries. Profiling is
performed separately from timing. Full measurements: `fire-paired.json`.

| FIRE variant | Before (ms/step) | After (ms/step) | Throughput gain |
| --- | ---: | ---: | ---: |
| ASE, fixed cell | 16.408 | 15.931 | 3.0% |
| ASE, unit cell filter | 20.348 | 19.782 | 2.9% |
| ASE, Fréchet cell filter | 28.739 | 27.951 | 2.8% |
| Velocity-Verlet, fixed cell | 18.224 | 17.593 | 3.6% |

Values are medians. This is a modest improvement, not a large speedup.
The separate-process exploratory after run overlapped CPU tests and produced
inconsistent timing; it is excluded from this comparison. Alternating measurements
above were run without overlapping tests or benchmarks.

The profiler counted 58 → 37 `cudaStreamSynchronize` calls per fixed-cell ASE
step, 34 → 22 scalar extractions, and 15 → 5 `aten::nonzero` calls. Counts include
the MLIP and shared helpers, which retain synchronization outside FIRE's core.
Fréchet-filter matrix operations also retain substantial synchronization.

After 100 steps, old/new timestep, alpha and positive-step counters matched in all
four cases. Maximum position difference was 4.3e-6 Å and force difference was
1.4e-5 eV/Å (GPU float32). Raw differences are recorded in the JSON.

## End-to-end relaxation

Five alternating old/new runs of the existing `run_torchsim_optimization` benchmark
in one process gave **4.710 → 4.582 seconds**, a **2.8% throughput gain**. All eight
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

Save the old FIRE implementation before changing it, then run the fixed-work
comparison using the final benchmark script:

```bash
git show 5540d893a849db9772081229a86d5f68537ad037:torch_sim/optimizers/fire.py > /tmp/fire_original.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  examples/benchmarking/fire-steps.py --reference-fire /tmp/fire_original.py
```

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

Ruff checks and formatting passed for all changed Python files. The fixed-step
benchmark script also completed a short run against the saved reference.

A separate float64 CPU comparison against the saved original implementation
matched at `rtol=atol=1e-12` after every step for 30 steps, across ASE/VV FIRE and
fixed/unit/Fréchet cells, with NaN-velocity replacements at steps 8 and 17.
