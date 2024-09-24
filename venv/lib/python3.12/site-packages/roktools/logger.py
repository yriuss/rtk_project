
import logging
import os

# TODO Move logging configuration to log.ini
# TODO Use slack handler for notifying error to slack rokubun group

FORMAT = '%(asctime)s - %(levelname)-8s - %(message)s'
EPOCH_FORMAT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)
logger.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))
console_handler = logging.StreamHandler()
formatter = logging.Formatter(FORMAT)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class LevelLogFilter(object):
    def __init__(self, levels):
        self.__levels = levels

    def filter(self, record):
        return record.levelno in self.__levels


def debug(message):
    logger.debug(message)


def info(message):
    logger.info(message)


def warning(message):
    logger.warning(message)


def error(message):
    logger.error(message)


def critical(message, exception=None):
    logger.critical(message, exc_info=exception)


def exception(message, exception):
    logger.critical(message, exc_info=exception)
    raise exception


def log(level, message):
    logger.log(logging._nameToLevel[level], message)


def set_level(level):
    logger.setLevel(level=level)


def setFileHandler(filename):
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    return handler


def unsetHandler(handler):
    handler.close()
    logger.removeHandler(handler)
