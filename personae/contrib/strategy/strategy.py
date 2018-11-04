# coding=uf-8

from abc import abstractmethod


class BaseStrategy(object):

    @abstractmethod
    def before_trading(self):
        pass

    @abstractmethod
    def handle_bar(self, bar):
        pass

    @abstractmethod
    def after_trading(self):
        pass
