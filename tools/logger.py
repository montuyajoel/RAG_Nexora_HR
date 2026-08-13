import logging


LOG_FORMAT = (
    "%(asctime)s - "
    "%(name)s - "
    "%(levelname)s - "
    "%(message)s"
)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        LOG_FORMAT
    )

    handler.setFormatter(
        formatter
    )

    logger.addHandler(
        handler
    )

    logger.propagate = False

    return logger