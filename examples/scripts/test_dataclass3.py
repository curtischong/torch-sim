from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def normalize_to_array(x: Sequence[float] | np.ndarray | bool) -> FloatArray:
    if isinstance(x, bool):
        return np.array([1.0] if x else [0.0], dtype=float)
    if isinstance(x, np.ndarray):
        return x.astype(float, copy=False)
    return np.array(x, dtype=float)


@dataclass
class Foo:
    # Callers pass `arr=...` with these types:
    arr: InitVar[Sequence[float] | np.ndarray | bool]

    # What we actually store:
    _arr: FloatArray = field(init=False, repr=False)

    def __post_init__(self, arr):
        self._arr = normalize_to_array(arr)

    # Public attribute: always ndarray after init
    @property
    def arr(self) -> FloatArray:
        return self._arr

    # Optional: allow reassignment with the same normalization
    @arr.setter
    def arr(self, value: Sequence[float] | np.ndarray | bool) -> None:
        print("self.arr", self.arr)
        self._arr = normalize_to_array(value)


if __name__ == "__main__":
    foo = Foo(arr=[1, 3])
    print(foo.arr)
    foo.arr = [2, 4]
    print(foo.arr)
