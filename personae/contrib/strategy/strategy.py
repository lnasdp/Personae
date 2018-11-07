# coding=utf-8

import numpy as np
import pandas as pd

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


class SampleStrategy(BaseStrategy):

    def before_trading(self):
        pass

    def handle_bar(self, bar):
        return pd.Series(index=bar.index, data=np.random.randint(-10, 1, size=(len(bar.index, ))))

    def after_trading(self):
        pass
