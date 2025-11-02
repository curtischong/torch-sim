from dataclasses import InitVar, dataclass


@dataclass(kw_only=True)
class BaseState:
    pbc: InitVar[bool]

    def __post_init__(self, pbc: bool | None):
        if pbc is None:
            pbc = True
        self.pbc = pbc


@dataclass(kw_only=True)
class ChildState(BaseState):
    child_attr: int


if __name__ == "__main__":
    base_state = BaseState(pbc=None)
    child_state = ChildState(pbc=None, child_attr=1)
    print(base_state)
    print(child_state)
