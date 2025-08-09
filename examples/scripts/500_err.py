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

# Note: the random int must have 2 atoms right now because
# of the ambiguous attribute scope handling (which will be fixed in https://github.com/Radical-AI/torch-sim/pull/228)
compound_init = [
    composition_to_random_structure(
        Composition(f"Zn{np.random.randint(2, 5)}"), scale_volume=scale_volume
    )
    for _ in range(5000)
]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Option 1: Load the raw model from the downloaded model
mace_model = mace_mp(
    model="https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0b/mace_agnesi_small.model",
    return_raw_model=True,
    default_dtype=dtype,
)

batched_model = MaceModel(
    model=mace_model,
    device=device,
    compute_forces=True,
    compute_stress=True,
    dtype=dtype,
    enable_cueq=False,
)


state = ts.initialize_state(compound_init, device=device, dtype=dtype)
state_list = state.split()
memory_metric_values = [
    calculate_memory_scaler(s, memory_scales_with="n_atoms_x_density") for s in state_list
]
max_memory_metric = estimate_max_memory_scaler(
    batched_model, state_list, metric_values=memory_metric_values
)
print("Max memory metric", max_memory_metric)

batcher = InFlightAutoBatcher(
    batched_model, max_memory_padding=1, max_memory_scaler=max_memory_metric * 0.8
)

convergence_fn = ts.generate_force_convergence_fn(0.025)
relaxed_state = ts.optimize(
    system=state,
    model=batched_model,
    optimizer=ts.frechet_cell_fire,
    autobatcher=batcher,
    max_steps=10,
    convergence_fn=convergence_fn,
)
