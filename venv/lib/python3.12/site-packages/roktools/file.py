from functools import wraps
from typing import IO


def process_filename_or_file_handler(mode):
    def decorator(func):
        @wraps(func)
        def wrapper(input, *args, **kwargs):
            if isinstance(input, str):
                with open(input, mode) as fh:
                    return func(fh, *args, **kwargs)
            else:
                return func(input, *args, **kwargs)
        return wrapper
    return decorator


def grep_lines(filename: str, pattern_string: str):
    """
    Generator function used to grep lines from a file. Can be used in methods
    such as numpy.genfromtxt, ...

    >>> generator = grep_lines(filename, "pattern")
    >>> data = numpy.loadtxt(generator)
    """

    with open(filename, 'r') as fh:
        for line in fh:
            if pattern_string in line:
                yield line


def skip_lines(fh: IO, n_lines: int):

    for _ in range(n_lines):
        fh.readline()
