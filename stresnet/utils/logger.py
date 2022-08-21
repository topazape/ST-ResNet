import logging
from typing import Optional


def get_logger(filename: Optional[str] = None) -> logging.Logger:
    fmt = logging.Formatter(
        fmt="[%(asctime)s] :%(name)s: [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    if filename is not None:
        handler = logging.FileHandler(filename=filename)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger
