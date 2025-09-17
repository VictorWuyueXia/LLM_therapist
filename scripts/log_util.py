# This script provides a utility for logging with colored output and file logging.
# It ensures that a log directory exists, and provides a get_logger function to create a logger with both stream and file handlers.
# The coloredlogs package is used to enhance the readability of logs in the terminal.

import os
import datetime
import coloredlogs
import logging

# Define the folder where log files will be stored
LOG_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log')

# Ensure the log directory exists, create it if it does not
if not os.path.exists(LOG_FOLDER):
    try:
        original_umask = os.umask(0)  # Set umask to 0 so files are created with the desired permissions
        os.makedirs(LOG_FOLDER, 0o777)  # Create the log directory with full permissions
    finally:
        os.umask(original_umask)  # Restore the original umask

def get_logger(name, file=None, file_handler=None):
    """
    Create and return a logger with colored console output and file logging.
    - name: logger name (usually module or subject id)
    - file: optional, path to the log file. If not provided, a new file is created in LOG_FOLDER.
    - file_handler: optional, a pre-configured file handler. If provided, it is used instead of creating a new one.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set logger to debug level to capture all messages

    # Create a stream handler for console output
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    # Note: coloredlogs will override the formatter for the stream handler
    logger.addHandler(c_handler)

    # Set up file handler for logging to a file
    if file_handler is not None:
        f_handler = file_handler
    else:
        if file is None:
            # If no file is specified, create a new log file with a timestamp in the name
            file = os.path.join(LOG_FOLDER, datetime.datetime.now().strftime(f"output_%y%m%d_%H%M%S.log"))
        f_handler = logging.FileHandler(file)
        f_handler.setLevel(logging.DEBUG)
        # Set a formatter for the file handler to include time, level, logger name, and message
        f_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
    logger.addHandler(f_handler)
    
    # Install coloredlogs to enhance the console output with colors
    coloredlogs.install(level='DEBUG', logger=logger)
    return logger

# If this script is run directly, demonstrate the logger output
if __name__ == "__main__":
    logger = get_logger("LogUtil")
    logger.debug("test debug")
    logger.info("test info")
    logger.warning("test warning")
    logger.error("test error")
    logger.critical("test critical")
    logger.info((1,2))