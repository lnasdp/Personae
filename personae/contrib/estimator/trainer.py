# coding=utf-8

import pandas as pd

from abc import abstractmethod

from personae.contrib.model.model import BaseModel
from personae.contrib.data.handler import BaseDataHandler

from personae.utility.profiler import TimeInspector


class BaseTrainer(object):

    def __init__(self, model_class, model_params: dict, data_handler: BaseDataHandler):

        # Model.
        self.model_class = model_class
        self.model_params = model_params
        self.model = None

        # Data handler.
        self.data_handler = data_handler

    @abstractmethod
    def train(self):
        raise NotImplementedError('Implement this method to train models.')

    @abstractmethod
    def load(self):
        raise NotImplementedError('Implement this method to load models.')

    @abstractmethod
    def predict(self, **kwargs):
        raise NotImplementedError('Implement this method to get prediction')


class StaticTrainer(BaseTrainer):

    def train(self):
        TimeInspector.set_time_mark()
        # Get model.
        model = self.model_class(**self.model_params)  # type: BaseModel
        model.fit(
            x_train=self.data_handler.x_train,
            y_train=self.data_handler.y_train,
            x_validation=self.data_handler.x_validation,
            y_validation=self.data_handler.y_validation,
            w_train=self.data_handler.w_train,
            w_validation=self.data_handler.w_validation
        )
        # Save model.
        model.save()
        # Set model.
        self.model = model
        TimeInspector.log_cost_time('Finished training model. (Static)')

    def load(self):
        TimeInspector.set_time_mark()
        # Get model.
        model = self.model_class(**self.model_params)  # type: BaseModel
        model.load()
        # Set model.
        self.model = model
        TimeInspector.log_cost_time('Finished loading model. (Static)')

    def predict(self):
        return self.model.predict(x_test=self.data_handler.x_test)


class RollingTrainer(BaseTrainer):

    def train(self):
        self.date_model_map = dict.fromkeys(seq=[self.data_handler.test_end_date], value=[model])

    def load(self):
        pass

    def predict(self):
        pass
