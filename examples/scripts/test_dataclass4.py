from dataclasses import InitVar, dataclass, field


@dataclass(kw_only=True, init=False)
class BaseState:
    system_idx: bool

    def __init__(self, system_idx: bool | None = None):  # noqa: FBT001
        """Initialize the BaseState."""
        self.system_idx = system_idx

    def __post_init__(self) -> None:
        """Post-initialize the BaseState."""
        if self.system_idx is None:
            self.system_idx = True


@dataclass(kw_only=True)
class ChildState(BaseState):
    child_attr: int


if __name__ == "__main__":
    base_state = BaseState(system_idx=None)
    child_state = ChildState(system_idx=None, child_attr=1)
    print(base_state)
    print(child_state)
