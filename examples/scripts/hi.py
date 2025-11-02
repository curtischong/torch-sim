from dataclasses import InitVar, dataclass, field


@dataclass
class Hi:
    sys_idx: InitVar[int | None]
    _sys_idx_attr: int = field(init=False)

    def __post_init__(self, sys_idx: int | None):
        if sys_idx is None:
            sys_idx = 0
        self._sys_idx_attr = sys_idx

    @property
    def _sys_idx(self):
        return self._sys_idx_attr

    @_sys_idx.setter
    def _sys_idx(self, sys_idx):
        # print('Setter Called with Value ', uploaded_by)
        self._sys_idx_attr = sys_idx

    def yes(self):
        return self.sys_idx


Hi.sys_idx = Hi._sys_idx

hi = Hi(sys_idx=None)
print(hi)
print(hi.yes())
