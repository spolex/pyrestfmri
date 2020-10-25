import logging


def init(name='logger', fh_level=logging.INFO, ch_level=logging.INFO):
    logger = logging.getLogger(name)
    # create logger with 'spam_application'
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(name + '.log')
    fh.setLevel(fh_level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(ch_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
