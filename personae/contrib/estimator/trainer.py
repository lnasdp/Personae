# coding=utf-8

import pandas as pd

from abc import abstractmethod

from personae.contrib.model.model import BaseModel
from personae.contrib.data.handler import BaseDataHandler

from personae.utility.profiler import TimeInspector
from personae.utility.logger import get_logger


class BaseTrainer(object):

    def __init__(self, model_class, model_params: dict, data_handler: BaseDataHandler):

        # Model.
        self.model_class = model_class
        self.model_params = model_params
        self.model = None

        # Data handler.
        self.data_handler = data_handler

        # Logger.
        self.logger = get_logger('TRAINER')

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

    def __init__(self, model_class, model_params: dict, data_handler: BaseDataHandler):
        super(RollingTrainer, self).__init__(model_class, model_params, data_handler)
        self.date_model_map = dict()

    def train(self):
        # 1. Get total data parts.
        total_data_parts = self.data_handler.rolling_total_parts
        info = 'Total numbers of model are: {}, start training models...'
        self.logger.warning(info.format(total_data_parts))
        # 2. Rolling train.
        for index, _ in enumerate(total_data_parts):
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
            model.name = '{}_{}'.format(model.name, index)
            model.save()
            # Build date - model map.
            self.date_model_map[self.data_handler.rolling_test_end_dates[index]] = model
            info = 'Total numbers of model are: {},' ' finished training model: {}.'
            TimeInspector.log_cost_time(info.format(total_data_parts, index + 1))

    def load(self):
        pass

    def predict(self):
        pass
