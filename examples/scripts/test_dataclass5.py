from dataclasses import InitVar, dataclass, field
from typing import cast


@dataclass(kw_only=True)
class BaseState:
    system_idx: bool | None

    def __post_init__(self) -> None:
        """Post-initialize the BaseState."""
        if self.system_idx is None:
            self.system_idx = True

    @property
    def system_idx(self) -> bool:
        return cast("bool", self.system_idx)

    @system_idx.setter
    def system_idx(self, value: bool) -> None:
        self.system_idx = value


@dataclass(kw_only=True)
class ChildState(BaseState):
    child_attr: int


if __name__ == "__main__":
    base_state = BaseState(system_idx=None)
    child_state = ChildState(system_idx=None, child_attr=1)
    print(base_state)
    print(child_state)
