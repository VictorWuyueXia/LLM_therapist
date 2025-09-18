# This script provides a utility for logging with colored output and file logging.
# It ensures that a log directory exists, and provides a get_logger function to create a logger with both stream and file handlers.
# The coloredlogs package is used to enhance the readability of logs in the terminal.

import os
import datetime
import coloredlogs
import logging
from scripts.config_loader import LOG_DIR

LOG_FOLDER = LOG_DIR

if not os.path.exists(LOG_FOLDER):
    try:
        original_umask = os.umask(0)
        os.makedirs(LOG_FOLDER, 0o777)
    finally:
        os.umask(original_umask)

def get_logger(name, file=None, file_handler=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    logger.addHandler(c_handler)
    if file_handler is not None:
        f_handler = file_handler
    else:
        if file is None:
            file = os.path.join(LOG_FOLDER, datetime.datetime.now().strftime(f"output_%y%m%d_%H%M%S.log"))
        f_handler = logging.FileHandler(file)
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
    logger.addHandler(f_handler)
    coloredlogs.install(level='DEBUG', logger=logger)
    return logger


