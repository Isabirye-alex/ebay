# utils/retry.py

import time
import logging
from functools import wraps


def retry(max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Retry decorator with exponential backoff.

    Args:
        max_retries (int): Number of attempts before failing
        delay (int | float): Initial delay in seconds
        backoff (int | float): Multiplier for delay
        exceptions (tuple): Exceptions to catch

    Returns:
        function: Wrapped function with retry logic
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    logging.warning(
                        f"[Retry {attempt}/{max_retries}] {func.__name__} failed: {e}"
                    )

                    if attempt == max_retries:
                        logging.error(f"{func.__name__} failed permanently.")
                        raise

                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator
