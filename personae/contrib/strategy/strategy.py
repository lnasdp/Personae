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

        amount = [1000 if np.random.randint(0, 10) % 2 == 0 else -1000] * len(bar.index)

        return pd.Series(index=bar.index, data=amount)

    def after_trading(self):
        pass
