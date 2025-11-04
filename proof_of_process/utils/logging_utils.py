import logging, os, sys

def get_logger(name="PoP", level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
