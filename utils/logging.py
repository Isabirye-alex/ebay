# utils/logging.py

import logging


def setup_logger(name="Logger", level=logging.INFO):
    """
    Configure and return a standardized logger.

    Args:
        name (str): Logger name
        level (int): Logging level

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger
