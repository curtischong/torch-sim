from dataclasses import InitVar, dataclass, field


@dataclass(kw_only=True)
class BaseState:
    # Accept the same public name during init, but do not store it directly
    system_idx: InitVar[bool | None] = None
    _system_idx: bool = field(init=False, repr=False)

    def __post_init__(self, system_idx: bool | None) -> None:
        self._system_idx = True if system_idx is None else system_idx

    @property
    def system_idx(self) -> bool:
        return self._system_idx

    @system_idx.setter
    def system_idx(self, value: bool) -> None:
        self._system_idx = value

    def __repr__(self) -> str:
        import dataclasses as dc
        from dataclasses import fields

        def is_initvar(field_obj: dc.Field) -> bool:
            return getattr(field_obj, "_field_type", None) is getattr(
                dc, "_FIELD_INITVAR", None
            )

        names_acc: list[str] = []
        for cls in type(self).__mro__:
            annotations = getattr(cls, "__annotations__", None)
            if annotations is None:
                continue
            for name, typ in annotations.items():
                if type(typ) is InitVar:
                    names_acc.append(name)
        initvar_names = tuple(names_acc)

        parts: list[str] = []
        for f in fields(self):
            field_name = f.name
            paired_with_initvar = (
                field_name.startswith("_") and field_name[1:] in initvar_names
            )
            if not paired_with_initvar and not f.repr:
                continue
            display_name = field_name[1:] if paired_with_initvar else field_name
            value = getattr(self, field_name)
            parts.append(f"{display_name}={value!r}")
        return f"{self.__class__.__name__}(" + ", ".join(parts) + ")"


@dataclass(kw_only=True, repr=False)
class ChildState(BaseState):
    child_attr: int


@dataclass(kw_only=True)
class Yo:
    """A class with a var"""

    what: bool | None


@dataclass(kw_only=True)
class Yay(Yo):
    yay: bool | None


if __name__ == "__main__":
    base_state = BaseState(system_idx=None)
    child_state = ChildState(system_idx=None, child_attr=1)
    # print("ANN", BaseState.__annotations__)
    print(base_state)
    print(child_state)

    print(Yo(what=None))
    print(Yay(what=None, yay=None))
