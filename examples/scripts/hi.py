import torch_sim as ts
from ase.build import bulk
import torch

atoms_list = [
    bulk("Si", "diamond", a=5.43, cubic=True),
    bulk("Fe", "bcc", a=2.8665, cubic=True),
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

state = ts.initialize_state(atoms_list, device=device, dtype=dtype)

print(state)
