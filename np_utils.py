from numpy.typing import NDArray


def lock(a: NDArray) -> None:
    if a is None:
        return
    a.flags.writeable = False
