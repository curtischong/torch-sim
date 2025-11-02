from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, cast, overload


@dataclass
class Hi:
    sys_idx: int

    if TYPE_CHECKING:

        @overload
        def __init__(self, sys_idx: int | None) -> None: ...

    def __post_init__(self):
        if self.sys_idx is None:
            self.sys_idx = 0


@dataclass
class Yes(Hi):
    yo: int


hi = Hi(sys_idx=None)
print(hi)

hi = Hi(sys_idx=2)
print(hi)

y = Yes(yo=2)
