# coding=utf-8

import logging
import sys
import os

import config

from time import time

loggers_map = {

}


def get_logger(module_name,
               log_dir=config.DEFAULT_LOG_DIR,
               enable_fh=True,
               sh_level=logging.WARNING,
               fh_level=logging.INFO):
    # If logger exist, return.
    if module_name in loggers_map:
        return loggers_map[module_name]
    # Make log and dir path.
    dir_path = os.path.join(log_dir, config.DATETIME_LOAD_MODULE)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    log_path = os.path.join(dir_path, '{}.log'.format(module_name))
    # Get logger.
    logger_name = '{}_logger'.format(module_name)
    logger = logging.getLogger(logger_name)
    logger.setLevel(fh_level)
    # logger.propagate = False
    # Get logger stream handler.
    log_sh = logging.StreamHandler()
    log_sh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
    log_sh.setLevel(sh_level)
    # Add handler.
    logger.addHandler(log_sh)
    # Get logger file handler.
    if enable_fh:
        log_fh = logging.FileHandler(log_path)
        log_fh.setLevel(fh_level)
        log_fh.setFormatter(logging.Formatter('[{}] {}'.format('%(asctime)s', '%(message)s')))
        # Add handler.
        logger.addHandler(log_fh)
    return logger


class TimeInspector(object):

    time_marks = []

    logger = get_logger('Timer', enable_fh=False)

    @classmethod
    def set_time_mark(cls):
        _time = time()
        cls.time_marks.append(_time)
        return _time

    @classmethod
    def pop_time_mark(cls):
        cls.time_marks.pop()

    @classmethod
    def get_cost_time(cls):
        cost_time = time() - cls.time_marks.pop()
        return cost_time

    @classmethod
    def log_cost_time(cls, info):
        cost_time = time() - cls.time_marks.pop()
        cls.logger.warning('Time cost: {0:.2f} | {1}'.format(cost_time, info))

