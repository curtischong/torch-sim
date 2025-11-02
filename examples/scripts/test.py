from dataclasses import InitVar, dataclass, field


@dataclass
class Foo:
    bar: InitVar[int | None]
    _bar: int = field(init=False)

    def __post_init__(self, bar: int | None):
        if bar is None:
            bar = 1
        self._bar = bar

    @property
    def bar(self):
        return self._bar

    @bar.setter
    def bar(self, value: int):
        self._bar = value


if __name__ == "__main__":
    foo = Foo(bar=None)
    print(foo)
    assert foo.bar is not None
