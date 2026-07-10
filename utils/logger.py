"""
Logger utility module.
"""
import logging
import sys


def initialize_logger(loglevel='INFO'):
    """
    Initializes and configures the root logger with a specific format and console handler.

    Args:
        loglevel (str): The logging level to set (e.g., 'INFO', 'DEBUG'). Defaults to 'INFO'.
    """
    formatter = logging.Formatter('%(asctime)-15s %(levelname)s %(message)s')

    consolehandler = logging.StreamHandler(sys.stdout)
    consolehandler.setFormatter(formatter)

    log = logging.getLogger()

    for hdlr in log.handlers[:]:
        log.removeHandler(hdlr)

    log.addHandler(consolehandler)
    log.setLevel(loglevel)
