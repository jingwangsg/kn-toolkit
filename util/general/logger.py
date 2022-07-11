import logging
from pytorch_lightning.utilities import rank_zero_only
import os
import textwrap


class MultiLineFormatter(logging.Formatter):
    def format(self, record):
        message = record.msg
        record.msg = ""
        header = super().format(record)
        msg = textwrap.indent(message, " " * len(header)).lstrip()
        record.msg = message
        return header + msg


def get_logger(name):
    """Initializes multi-GPU-friendly python command line logger."""
    # multi-line indentation: https://stackoverflow.com/questions/58590731/how-to-indent-multiline-message-printed-by-python-logger
    formatter = MultiLineFormatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logging.basicConfig(
        # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        # datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[log_handler],
    )
    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
