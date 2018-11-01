# coding=utf-8

from abc import abstractmethod


class BaseStrategy(object):

    @abstractmethod
    def before_trading(self, date):
        pass

    @abstractmethod
    def handle_bar(self, bar_df, date):
        pass

    @abstractmethod
    def after_trading(self, date):
        pass
