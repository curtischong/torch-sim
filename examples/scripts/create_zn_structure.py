"""
This script creates a random structure for a given composition and relaxes it
using a combination of soft-sphere potential and the ORB model.

The process involves:
1. Generating a random structure with a given composition.
2. Relaxing the structure using a soft-sphere potential to remove overlaps.
3. Further relaxing the structure using the ORB model with the FIRE optimizer.
4. Printing the final energy, forces, and trajectory frames.
"""

from typing import Final

import numpy as np
import torch
from pymatgen.core import Composition, Structure

import torch_sim as ts
from torch_sim.models.soft_sphere import SoftSphereModel
from torch_sim.optimizers import fire, unit_cell_fire
from torch_sim.state import SimState


VEGARDS_VOLUME_PER_ATOM: Final[dict[str, float]] = {
    "H": 46.47654994,
    "He": 17.29396027,
    "Li": 20.40188669,
    "Be": 7.94909257,
    "B": 7.22909694,
    "C": 10.90829421,
    "N": 24.58000466,
    "O": 17.28781129,
    "F": 13.35799978,
    "Ne": 18.45618751,
    "Na": 36.93819600,
    "Mg": 22.39949816,
    "Al": 16.47171763,
    "Si": 20.16476342,
    "P": 23.03718174,
    "S": 27.51223970,
    "Cl": 28.21528937,
    "Ar": 38.56579000,
    "K": 76.52889532,
    "Ca": 43.36096382,
    "Sc": 25.01223695,
    "Ti": 17.01971989,
    "V": 13.26378015,
    "Cr": 13.08447886,
    "Mn": 13.02268306,
    "Fe": 11.73408423,
    "Co": 10.84154263,
    "Ni": 10.49202034,
    "Cu": 11.44599856,
    "Zn": 14.42255445,
    "Ga": 19.17560525,
    "Ge": 22.84410461,
    "As": 24.63744306,
    "Se": 33.36400842,
    "Br": 31.67578476,
    "Kr": 47.23822778,
    "Rb": 90.26387157,
    "Sr": 56.43301959,
    "Y": 33.77781304,
    "Zr": 23.49965943,
    "Nb": 18.25806661,
    "Mo": 15.89162790,
    "Tc": 14.25169050,
    "Ru": 13.55117683,
    "Rh": 13.78278033,
    "Pd": 15.02799459,
    "Ag": 17.36423754,
    "Cd": 22.30751928,
    "In": 26.39251473,
    "Sn": 35.44875317,
    "Sb": 31.44050547,
    "I": 44.80605045,
    "Xe": 85.78650724,
    "Cs": 123.10401593,
    "Ba": 64.99909048,
    "La": 37.57215894,
    "Ce": 25.50164500,
    "Pr": 37.28132207,
    "Nd": 35.46125938,
    "Pm": 34.52341053,
    "Sm": 33.80194237,
    "Eu": 44.06303669,
    "Gd": 33.89555135,
    "Tb": 32.71278409,
    "Dy": 32.00431174,
    "Ho": 31.36464775,
    "Er": 30.88840453,
    "Tm": 30.19899786,
    "Lu": 29.68884455,
    "Hf": 22.26362174,
    "Ta": 18.47902812,
    "W": 15.93226352,
    "Re": 14.81078265,
    "Os": 14.12740026,
    "Ir": 14.31038277,
    "Pt": 15.32745405,
    "Au": 18.14473805,
    "Hg": 27.00558105,
    "Tl": 29.90613832,
    "Pb": 31.05371528,
    "Bi": 34.32738211,
    "Ac": 46.13635019,
    "Th": 32.14427668,
    "Pa": 24.46037090,
    "U": 19.65107607,
    "Np": 18.10940441,
    "Pu": 21.86327923,
}


def get_volume_per_atom(comp: Composition, volume_per_atom: dict) -> float:
    """Get the volume per atom for a given composition using Vegard's law.
    Args:
        comp (pymatgen composition): Composition object
        volume_per_atom (dict): Dictionary of volume per atom for each element
    Returns:
        volume_per_atom (float): Volume per atom for the given composition
    """
    return sum(
        volume_per_atom[elem.name] * comp.get_atomic_fraction(elem.name)
        for elem in comp.elements
    )


def get_box_side_length_from_density_or_comp(
    composition: Composition,
    # n_atoms: int,
    density: float | None = None,
    volume_per_atom: dict = VEGARDS_VOLUME_PER_ATOM,
) -> float:
    """Calculate cubic box side length from composition and density.
    Args:
        composition (pymatgen Composition object) : Target formula with actual atom counts
            e.g. Cr80Ti20
        num_atoms (int): Total number of atoms
        density (float): Target density in g/cm^3. If None, uses average MP volume.
        method (str): Method to use to calculate volume per atom.
            'mp-morph' uses the average volume from the mp-morph workflow.
            'vegards' uses the volume per atom from Vegard's law.
            'covalent' uses the covalent radii to estimate the volume.
        volume_per_atom (dict): Dictionary of volume per atom for each element if using
            Vegard's law
    Returns:
        Side length of cubic box in Angstroms
    """
    n_atoms = composition.num_atoms
    if density:
        avogadro = 6.02214076e23
        density_in_ang = (density * avogadro) / (composition.weight * 1e24)
        volume = 1 / density_in_ang
        box_side = volume ** (1 / 3)
    else:
        vol_per_atom = get_volume_per_atom(composition, volume_per_atom)
        volume = vol_per_atom * n_atoms
        box_side = volume ** (1 / 3)

    return box_side


def composition_to_random_structure(
    composition: Composition,
    density: float | None = None,
    scale_volume: float = 1.0,
    seed: int | None = None,
) -> Structure:
    """Generates random packed structure in a box, optionally minimizing overlap.
    Modified version of the jax-md a2c workflow in ASE.
    See: https://arxiv.org/abs/2310.01117 and https://github.com/jax-md/jax-md/pull/327
    Args:
        composition (pymatgen Composition object): Target formula with actual atom counts:
            e.g. Cr80Ti20
        density (float): target material density of the structure in g/cm^3
        lattice (Sequence[Sequence[float]]): lattice vectors of the structure if
            predefined
        side_length_scale (float): scale factor for the box side lengths in x, y, z
        seed (int): random seed
        diameter (float): interatomic distances below this value considered overlapping
        auto_diameter (bool): if True, attempts to calculate diameter for soft-sphere pot
        max_iter (int): number of fire decent steps applied in overlap minimization
        distance_tolerance (float): used to mask out identical atoms in distance matrix
        return_min_distance (bool): if True, returns the minimum distance between atoms
        method (str): Method to use to calculate volume per atom.
            'mp-morph' uses the average volume from the mp-morph workflow.
            'vegards' uses the volume per atom from Vegard's law.
            'covalent' uses the covalent radii to estimate the volume.
        volume_per_atom (dict): Dictionary of volume per atom for each element if using
            Vegard's law
    Returns:
        Structure: Randomly packed structure with minimum distance between atoms if
            requested.
    """
    element_symbols, element_counts = zip(*composition.as_dict().items(), strict=False)
    element_counts = [int(i) for i in element_counts]
    species = [
        el for i, el in enumerate(element_symbols) for _ in range(element_counts[i])
    ]
    box_side = get_box_side_length_from_density_or_comp(composition, density=density)

    lattice = np.eye(3) * box_side * scale_volume ** (1 / 3)

    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 1.0, size=(sum(element_counts), 3))

    positions_cart = np.dot(positions, lattice)

    return Structure(lattice, species, positions_cart, coords_are_cartesian=True)


def pack_soft_sphere(
    comp: Composition,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    sigma: float = 2.5,
    scale_volume: float = 1.0,
) -> SimState:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    struct = composition_to_random_structure(comp, scale_volume=scale_volume)

    ss_model = SoftSphereModel(
        sigma=sigma,
        device=device,
        dtype=dtype,
        use_neighbor_list=True,
        compute_stress=True,
    )
    return ts.optimize(
        system=struct,
        model=ss_model,
        optimizer=unit_cell_fire,
    )
