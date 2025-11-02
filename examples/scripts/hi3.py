from dataclasses import Field, InitVar, dataclass, field
from typing import TYPE_CHECKING, cast, overload


@dataclass
class Hi:
    sys_idx: int | None = field(default=None)

    # if TYPE_CHECKING:

    #     @overload
    #     def __init__(self, sys_idx: int | None) -> None: ...

    def __post_init__(self):
        if self.sys_idx is None:
            self.sys_idx = 0

    if TYPE_CHECKING:

        @property
        def sys_idx(self) -> int:
            return self.sys_idx


@dataclass(kw_only=True)
class Yes(Hi):
    yo: int


hi = Hi()
print(hi)

hi = Hi(sys_idx=2)
print(hi)

y = Yes(yo=2, sys_idx=3)

if y.sys_idx == 3:
    print("Yes")
