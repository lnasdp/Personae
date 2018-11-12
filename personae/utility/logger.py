# coding=utf-8

import logging
import os


loggers_map = {

}


def get_logger(module_name,
               log_dir='/tmp/',
               enable_fh=False,
               sh_level=logging.WARNING,
               fh_level=logging.INFO,
               **kwargs):

    # 1. If logger exist, return.
    if module_name in loggers_map:
        return loggers_map[module_name]

    # 2. Make dir path.
    dir_path = os.path.join(log_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 3. Make log file path.
    log_path = os.path.join(dir_path, '{}.log'.format(module_name))

    # 4. Get logger.
    logger_name = '{}_logger'.format(module_name)
    logger = logging.getLogger(logger_name)
    logger.setLevel(fh_level)
    # logger.propagate = False

    # 5. Get logger stream handler.
    log_sh = logging.StreamHandler()
    log_sh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
    log_sh.setLevel(sh_level)
    logger.addHandler(log_sh)

    # 6. Get logger file handler.
    if enable_fh:
        log_fh = logging.FileHandler(log_path)
        log_fh.setLevel(fh_level)
        log_fh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
        logger.addHandler(log_fh)
    return logger



