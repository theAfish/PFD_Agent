import os
from contextlib import (
    contextmanager,
)
from functools import (
    wraps,
)
from pathlib import (
    Path,
)
from typing import (
    Callable,
)

from dflow.python import (
    OPIO,
)

@contextmanager
def set_directory(path: Path):
    """Sets the current working path within the context.

    Parameters
    ----------
    path : Path
        The path to the cwd

    Yields
    ------
    None

    Examples
    --------
    >>> with set_directory("some_path"):
    ...    do_something()
    """
    cwd = Path().absolute()
    path.mkdir(exist_ok=True, parents=True)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)