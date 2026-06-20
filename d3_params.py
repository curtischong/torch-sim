"""Build nvalchemiops D3Parameters from the Grimme reference data (via tad-dftd3).

Run directly to validate the converted params against tad-dftd3's own D3(BJ)-PBE
dispersion energy on a small cluster.

Usage (params only): import build_pbe_d3_model
Usage (validation):  uv run --with tad-dftd3 python d3_params.py
"""

from __future__ import annotations

import torch

# PBE BJ-damping params (simple-dftd3 parameters.toml)
PBE_BJ = {"a1": 0.4289, "s8": 0.7875, "a2": 4.4407, "s6": 1.0}


def _reference_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (rcov, r4r2, c6ab, cn_ref) in nvalchemiops layout (max_z=103, mesh=7)."""
    from tad_dftd3 import data, reference
    from tad_mctc.data import COV_D3

    ref = reference.Reference()  # c6: (104,104,7,7), cn: (104,7)
    n_elem, mesh = ref.cn.shape  # 104, 7

    rcov = COV_D3(dtype=torch.float64)[:n_elem].clone()
    r4r2 = data.R4R2(dtype=torch.float64)[:n_elem].clone()
    c6ab = ref.c6.clone().to(torch.float64)  # (104,104,7,7), padded refs are 0.0

    # cn_ref[zi,zj,i,j] = reference CN of element zi at ref i (broadcast over zj,j).
    # The kernel reads the partner's CN via the symmetric entry [zj,zi,j,i].
    cn = ref.cn.clone().to(torch.float64)  # (104,7); padded entries are -1.0
    # Push padded references far away so their Gaussian weight exp(-4*dCN^2) -> 0
    cn[cn < 0] = -1.0e6
    cn_ref = cn[:, None, :, None].expand(n_elem, n_elem, mesh, mesh).contiguous()

    return rcov, r4r2, c6ab, cn_ref


def build_pbe_d3_model(
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    cutoff: float = 40.0,  # Bohr
    compute_stress: bool = True,
):
    from nvalchemiops.torch.interactions.dispersion import D3Parameters

    from torch_sim.models.dispersion import D3DispersionModel

    rcov, r4r2, c6ab, cn_ref = _reference_tensors()
    params = D3Parameters(
        rcov=rcov.to(device),
        r4r2=r4r2.to(device),
        c6ab=c6ab.to(device),
        cn_ref=cn_ref.to(device),
        interp_mesh=c6ab.shape[-1],
    )
    return D3DispersionModel(
        **PBE_BJ,
        d3_params=params,
        cutoff=cutoff,
        device=torch.device(device),
        dtype=dtype,
        compute_forces=True,
        compute_stress=compute_stress,
    )


def _validate() -> None:
    """Compare nvalchemiops D3 energy vs tad-dftd3 on a small molecular cluster."""
    import tad_dftd3 as d3
    from tad_mctc.units import AU2EV

    import torch_sim as ts

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    # A small Na/Ta/Cl cluster (Angstrom)
    numbers = torch.tensor([11, 17, 17, 73, 17, 17], device=device)
    pos_ang = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.4, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [3.0, 3.0, 0.0],
            [3.0, 3.0, 2.5],
            [5.2, 3.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )

    # tad-dftd3 reference (non-periodic), positions in Bohr
    AA2AU = 1.0 / 0.52917721067
    param = {k: torch.tensor(v, dtype=dtype, device=device) for k, v in PBE_BJ.items()}
    e_tad = d3.dftd3(numbers, pos_ang * AA2AU, param).sum().item() * AU2EV

    # nvalchemiops D3 via torch-sim model (non-periodic, big box)
    model = build_pbe_d3_model(device=device, dtype=dtype, cutoff=60.0,
                               compute_stress=False)
    state = ts.SimState(
        positions=pos_ang,
        masses=torch.ones(len(numbers), dtype=dtype, device=device),
        cell=torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 100.0,
        pbc=False,
        atomic_numbers=numbers,
    )
    e_nv = model(state)["energy"].item()

    print(f"tad-dftd3 D3(BJ)-PBE energy : {e_tad:.6f} eV")
    print(f"nvalchemiops D3 (torch-sim): {e_nv:.6f} eV")
    print(f"abs diff: {abs(e_tad - e_nv):.3e} eV  "
          f"rel: {abs(e_tad - e_nv) / abs(e_tad):.2%}")


if __name__ == "__main__":
    _validate()
