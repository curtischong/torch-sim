"""Optimizer state classes."""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any

import torch

from torch_sim.state import SimState


@dataclass(kw_only=True)
class OptimState(SimState):
    """Unified state class for optimization algorithms.

    This class extends SimState to store and track the evolution of system state
    during optimization. It maintains the energies, forces, and optional cell
    optimization state needed for structure relaxation.
    """

    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor

    _atom_attributes = SimState._atom_attributes | {"forces"}  # noqa: SLF001
    _system_attributes = SimState._system_attributes | {"energy", "stress"}  # noqa: SLF001

    def __init__(
        self,
        *,
        forces: torch.Tensor,
        energy: torch.Tensor,
        stress: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.forces = forces
        self.energy = energy
        self.stress = stress

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _ensure_init_calls_parent(cls)


@dataclass(kw_only=True)
class FireState(OptimState):
    """State class for FIRE optimization.

    Extends OptimState with FIRE-specific parameters for velocity-based optimization.
    """

    velocities: torch.Tensor
    dt: torch.Tensor
    alpha: torch.Tensor
    n_pos: torch.Tensor

    _atom_attributes = OptimState._atom_attributes | {"velocities"}  # noqa: SLF001
    _system_attributes = OptimState._system_attributes | {"dt", "alpha", "n_pos"}  # noqa: SLF001

    def __init__(
        self,
        *,
        velocities: torch.Tensor,
        dt: torch.Tensor,
        alpha: torch.Tensor,
        n_pos: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.velocities = velocities
        self.dt = dt
        self.alpha = alpha
        self.n_pos = n_pos


# there's no GradientDescentState, it's the same as OptimState


class _SuperInitCallVisitor(ast.NodeVisitor):
    """AST visitor that detects calls to parent __init__ implementations."""

    def __init__(self, base_names: set[str]) -> None:
        self._base_names = base_names
        self.found = False

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "__init__":
            # Detect super().__init__(...)
            if isinstance(node.func.value, ast.Call):
                func = node.func.value.func
                if isinstance(func, ast.Name) and func.id == "super":
                    self.found = True
            # Detect direct BaseClass.__init__(...)
            elif isinstance(node.func.value, ast.Name) and node.func.value.id in self._base_names:
                self.found = True
        self.generic_visit(node)


def _ensure_init_calls_parent(cls: type) -> None:
    """Ensure subclasses define __init__ and call a parent initializer."""
    if "__init__" not in cls.__dict__:
        raise TypeError(
            f"{cls.__name__} must define an __init__ method that calls its parent."
        )

    init_fn = cls.__dict__["__init__"]
    if not callable(init_fn):
        raise TypeError(f"{cls.__name__}.__init__ must be callable.")

    try:
        source = inspect.getsource(init_fn)
    except (OSError, TypeError) as exc:
        msg = (
            f"Cannot verify that {cls.__name__}.__init__ calls its parent. "
            "Define __init__ in source code and ensure it invokes super().__init__."
        )
        raise TypeError(msg) from exc

    tree = ast.parse(textwrap.dedent(source))
    base_names = {base.__name__ for base in cls.__mro__[1:] if hasattr(base, "__name__")}
    visitor = _SuperInitCallVisitor(base_names)
    visitor.visit(tree)

    if not visitor.found:
        raise TypeError(
            f"{cls.__name__}.__init__ must call super().__init__ (or a parent class __init__)."
        )
