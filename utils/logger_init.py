import logging
import os
import sys

class LogColors:
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: LogColors.BLUE + "%(message)s" + LogColors.RESET,
        logging.INFO: LogColors.GREEN + "%(message)s" + LogColors.RESET,
        logging.WARNING: LogColors.YELLOW + "%(message)s" + LogColors.RESET,
        logging.ERROR: LogColors.RED + "%(message)s" + LogColors.RESET,
        logging.CRITICAL: LogColors.RED + "%(message)s" + LogColors.RESET,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, "%(message)s")
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] ' + log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

logger = logging.getLogger('YSHLogger')

def init_logger(log_path):
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_dir = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(log_dir, exist_ok=True)

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    try:
        fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"Failed to initialize file logger: {e}")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColoredFormatter())
    logger.addHandler(ch)
    
    return logger