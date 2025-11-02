"""Types used across TorchSim."""

import inspect
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, TypedDict, Union, get_type_hints

import torch


if TYPE_CHECKING:
    from ase import Atoms
    from phonopy.structure.atoms import PhonopyAtoms
    from pymatgen.core import Structure

    from torch_sim.state import SimState


MemoryScaling = Literal["n_atoms_x_density", "n_atoms"]
StateKey = Literal["positions", "masses", "cell", "pbc", "atomic_numbers", "system_idx"]
StateDict = dict[StateKey, torch.Tensor]


class BravaisType(StrEnum):
    """Enumeration of the seven Bravais lattice types in 3D crystals.

    These lattice types represent the distinct crystal systems classified
    by their symmetry properties, from highest symmetry (cubic) to lowest
    symmetry (triclinic).

    Each type has specific constraints on lattice parameters and angles,
    which determine the number of independent elastic constants.
    """

    cubic = "cubic"
    hexagonal = "hexagonal"
    trigonal = "trigonal"
    tetragonal = "tetragonal"
    orthorhombic = "orthorhombic"
    monoclinic = "monoclinic"
    triclinic = "triclinic"


StateLike = Union[
    "Atoms",
    "Structure",
    "PhonopyAtoms",
    list["Atoms"],
    list["Structure"],
    list["PhonopyAtoms"],
    "SimState",
]


def typed_dict_from_init(cls: type) -> type[TypedDict]:
    """Generate a TypedDict describing the __init__ parameters of a class."""
    # Get the __init__ signature
    sig = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)

    fields: dict = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        # Get annotation (or default to Any)
        annotation = hints.get(name, object)

        # Mark as optional if there's a default
        if param.default is not inspect.Parameter.empty:
            annotation = annotation | None

        fields[name] = annotation

    # Dynamically create a TypedDict class
    typed_dict_name = f"{cls.__name__}InitDict"
    return TypedDict(typed_dict_name, fields)
