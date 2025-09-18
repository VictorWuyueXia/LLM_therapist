# pip install coloredlogs
# To test coloredlogs: coloredlogs --demo
import os
import datetime
import coloredlogs, logging

LOG_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log')
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
    # coloredlogs overrides the formatter
    # c_handler.setFormatter(logging.Formatter('[%(levelname)s] %(name)s - %(message)s'))
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

if __name__ == "__main__":
    logger = get_logger("LogUtil")
    logger.debug("test debug")
    logger.info("test info")
    logger.warning("test warning")
    logger.error("test error")
    logger.critical("test critical")
    logger.info((1,2))