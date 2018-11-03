# coding=utf-8

from time import time

from personae.utility import logger


class TimeInspector(object):

    time_marks = []

    logger = logger.get_logger('Timer', enable_fh=False)

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
