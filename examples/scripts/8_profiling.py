"""Profiling Example - Identify performance bottlenecks in simulations.

This script demonstrates how to use the profiling utilities to:
- Profile optimization runs and export Chrome Trace format for flame graphs
- Use labeled sections to identify specific bottlenecks
- View results in Chrome DevTools (chrome://tracing)
"""

# /// script
# dependencies = []
# ///

import os
import tempfile

import torch
from ase.build import bulk

import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.profiling import Profiler, profiling_section


# Set up the device and data type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# Number of steps to run
SMOKE_TEST = os.getenv("CI") is not None
N_steps = 5 if SMOKE_TEST else 50

# Output directory for profile traces
OUTPUT_DIR = tempfile.gettempdir() if SMOKE_TEST else "./profiles"

# ============================================================================
# SECTION 1: Basic Profiling
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 1: Basic Profiling of FIRE Optimization")
print("=" * 70)

# Create a Lennard-Jones model for Argon
lj_model = LennardJonesModel(
    sigma=3.405,
    epsilon=0.0104,
    cutoff=2.5 * 3.405,
    device=device,
    dtype=dtype,
)

# Create an FCC Argon structure
atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat([3, 3, 3])
state = ts.io.atoms_to_state(atoms, device=device, dtype=dtype)

# Add some perturbation to create a non-equilibrium structure
state.positions = state.positions + 0.1 * torch.randn_like(state.positions)

# Profile the optimization
print(f"\nProfiling FIRE optimization for {N_steps} steps...")
print(f"Output directory: {OUTPUT_DIR}")

with Profiler("fire_optimization", output_dir=OUTPUT_DIR) as prof:
    opt_state = ts.fire_init(state=state, model=lj_model, dt_start=0.005)
    for step in range(N_steps):
        opt_state = ts.fire_step(state=opt_state, model=lj_model, dt_max=0.01)

print(f"\nProfile saved to: {prof.trace_path}")
print("\nTo view the flame graph:")
print("  1. Open Chrome and go to: chrome://tracing")
print("  2. Click 'Load' and select the JSON file above")
print("  3. Use WASD keys to navigate, scroll to zoom")

# Print a summary table
print("\n--- CPU Time Summary (top 10 operations) ---")
prof.print_summary(sort_by="cpu_time_total", row_limit=10)


# ============================================================================
# SECTION 2: Profiling with Labeled Sections
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 2: Profiling with Labeled Sections")
print("=" * 70)

# Reset the state
state = ts.io.atoms_to_state(atoms, device=device, dtype=dtype)
state.positions = state.positions + 0.1 * torch.randn_like(state.positions)

print("\nProfiling with labeled sections for finer granularity...")

with Profiler("labeled_sections", output_dir=OUTPUT_DIR) as prof:
    # Label the initialization phase
    with profiling_section("fire_initialization"):
        opt_state = ts.fire_init(state=state, model=lj_model, dt_start=0.005)

    # Label the optimization loop
    with profiling_section("optimization_loop"):
        for step in range(N_steps):
            # You can even label individual steps or groups of steps
            with profiling_section(f"step_{step}"):
                opt_state = ts.fire_step(state=opt_state, model=lj_model, dt_max=0.01)

    # Label any post-processing
    with profiling_section("final_energy_calc"):
        final_output = lj_model(opt_state)

print(f"\nProfile with sections saved to: {prof.trace_path}")
print("\nLabeled sections will appear as named regions in the flame graph.")


# ============================================================================
# SECTION 3: Comparing Different Configurations
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Profile Memory Usage")
print("=" * 70)

from torch_sim.profiling import ProfilerConfig

# Configure profiler to track memory
config = ProfilerConfig(
    profile_memory=True,
    with_stack=True,  # Enable for better flame graphs
    with_flops=True,  # Estimate FLOPs
)

state = ts.io.atoms_to_state(atoms, device=device, dtype=dtype)
state.positions = state.positions + 0.1 * torch.randn_like(state.positions)

print("\nProfiling with memory tracking enabled...")

with Profiler("memory_profile", output_dir=OUTPUT_DIR, config=config) as prof:
    opt_state = ts.fire_init(state=state, model=lj_model, dt_start=0.005)
    for step in range(N_steps):
        opt_state = ts.fire_step(state=opt_state, model=lj_model, dt_max=0.01)

print(f"\nMemory profile saved to: {prof.trace_path}")

# Show memory usage summary
print("\n--- Memory Usage Summary ---")
print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))


print("\n" + "=" * 70)
print("Profiling complete!")
print("=" * 70)
print(f"\nAll profile traces saved to: {OUTPUT_DIR}")
print("\nYou can also view profiles at:")
print("  - https://ui.perfetto.dev/ (drag and drop JSON file)")
print("  - https://www.speedscope.app/ (for stack traces)")
