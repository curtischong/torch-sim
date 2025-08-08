from torch_sim.models.orb import OrbModel
import numpy as np
from ase.build import bulk
import torch
from orb_models.forcefield import pretrained


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

orb_ff = pretrained.orb_v3_conservative_inf_omat(
    device=device,
    precision="float32-high",
)

model = OrbModel(
    model=orb_ff,
    conservative=True,
    compute_stress=True,
    compute_forces=True,
)
si_dc = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
atoms_list = [si_dc, si_dc]

positions_numpy = np.concatenate([atoms.positions for atoms in atoms_list])

# stack cell vectors into a (2, 3, 3) array where the first index is batch dimension
cell_numpy = np.stack([atoms.cell.array for atoms in atoms_list])

# concatenate atomic numbers into a (16,) array
atomic_numbers_numpy = np.concatenate(
    [atoms.get_atomic_numbers() for atoms in atoms_list]
)

# convert to tensors
positions = torch.tensor(positions_numpy, device=device, dtype=dtype)
cell = torch.tensor(cell_numpy, device=device, dtype=dtype)
atomic_numbers = torch.tensor(atomic_numbers_numpy, device=device, dtype=torch.int)


# create system idx array of shape (16,) which is 0 for first 8 atoms, 1 for last 8 atoms
atoms_per_system = torch.tensor(
    [len(atoms) for atoms in atoms_list], device=device, dtype=torch.int
)
system_idx = torch.repeat_interleave(
    torch.arange(len(atoms_per_system), device=device), atoms_per_system
)

# You can see their shapes are as expected
print(f"Positions: {positions.shape}")
print(f"Cell: {cell.shape}")
print(f"Atomic numbers: {atomic_numbers.shape}")
print(f"System indices: {system_idx.shape}")

# Now we can pass them to the model
results = model(
    dict(
        positions=positions,
        cell=cell,
        atomic_numbers=atomic_numbers,
        system_idx=system_idx,
        pbc=True,
    )
)

# The energy has shape (n_systems,) as the structures in a batch
print(f"Energy: {results['energy'].shape}")

# The forces have shape (n_atoms, 3) same as positions
print(f"Forces: {results['forces'].shape}")

# The stress has shape (n_systems, 3, 3) same as cell
print(f"Stress: {results['stress'].shape}")

# Check if the energy, forces, and stress are the same for the Si system across the batch
print(torch.max(torch.abs(results["energy"][0] - results["energy"][1])))
print(torch.max(torch.abs(results["forces"][0] - results["forces"][1])))
print(torch.max(torch.abs(results["stress"][0] - results["stress"][1])))