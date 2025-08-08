import numpy as np
import torch
from ase.build import bulk
from mace.calculators.foundations_models import mace_mp
from pymatgen.core import Composition

import torch_sim as ts
from examples.scripts.create_zn_structure import composition_to_random_structure
from torch_sim.autobatching import (
    InFlightAutoBatcher,
    calculate_memory_scaler,
    estimate_max_memory_scaler,
)
from torch_sim.models.mace import MaceModel


scale_volume = 1.0

# Generate 200 random structures
compound_init = [
    composition_to_random_structure(
        Composition(f"Zn{np.random.randint(2, 5)}"), scale_volume=scale_volume
    )
    for _ in range(200)
]

# --- Placeholder definitions ---
# Please replace these with your actual model and device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
mace_model = mace_mp(
    model="https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model",
    return_raw_model=True,
    default_dtype=dtype,
)

# Option 2: Load the compiled model from the local file
# MODEL_PATH = "../../../checkpoints/MACE/mace-mpa-0-medium.model"
# loaded_model = torch.load(MODEL_PATH, map_location=device)

# Create diamond cubic Silicon
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
atoms_list = [si_dc, si_dc]

batched_model = MaceModel(
    # Pass the raw model
    model=mace_model,
    # Or load from compiled model
    # model=compiled_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)


state = ts.initialize_state(compound_init, device=device, dtype=torch.float64)

convergence_fn = ts.generate_force_convergence_fn(0.025)
relaxed_state = ts.optimize(
    system=state,
    model=batched_model,
    optimizer=ts.frechet_cell_fire,
    max_steps=1000,
    convergence_fn=convergence_fn,
)
print("Optimization finished.")
print(relaxed_state)
