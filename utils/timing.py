# utils/timing.py

import time
from functools import wraps
from utils.logging import setup_logger

logger = setup_logger()


def timeit(func):
    """
    Decorator to measure execution time of a function.

    Logs execution duration in seconds.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()

        result = func(*args, **kwargs)

        end = time.perf_counter()
        duration = end - start

        logger.info(f"{func.__name__} executed in {duration:.4f} seconds")

        return result

    return wrapper

from contextlib import contextmanager


@contextmanager
def time_block(label: str):
    """
    Context manager to time a block of code.

    Args:
        label (str): Name of the operation
    """
    start = time.perf_counter()
    yield
    end = time.perf_counter()

    logger.info(f"{label} took {end - start:.4f} seconds")
