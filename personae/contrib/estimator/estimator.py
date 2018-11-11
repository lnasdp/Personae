# coding=utf-8

import yaml
import argparse
import importlib

from importlib import util

from personae.contrib.backtest.engine import PredictorEngine
from personae.contrib.data.handler import BaseDataHandler
from personae.contrib.model.model import BaseModel


class ConfigManager(object):

    def __init__(self, config_path):
        # Config path.
        self.config_path = config_path
        # Load config.
        with open(self.config_path, 'r') as fp:
            self.config = yaml.load(fp)
        # Set config.
        self.data_config = DataConfig(self.config.get('data', dict()))
        self.model_config = ModelConfig(self.config.get('model', dict()))
        self.strategy_config = StrategyConfig(self.config.get('strategy', dict()))
        self.backtest_engine_config = BacktestEngineConfig(self.config.get('backtest', dict()))


class DataConfig(object):

    def __init__(self, config: dict):
        self.handler_class = config.get('class', 'BaseDataHandler')
        self.handler_module = config.get('module', None)
        self.handler_params = config


class ModelConfig(object):

    def __init__(self, config: dict):
        self.model_class = config.get('class', 'BaseModel')
        self.model_module = config.get('module', None)
        self.model_rolling = config.get('rolling', False)
        self.model_params = config


class StrategyConfig(object):

    def __init__(self, config: dict):
        self.strategy_class = config.get('class', 'TopKStrategy')
        self.strategy_module = config.get('module', None)
        self.strategy_params = config


class BacktestEngineConfig(object):

    def __init__(self, config: dict):
        self.backtest_engine_class = config.get('class', 'BaseEngine')
        self.backtest_engine_module = config.get('module', None)
        self.backtest_engine_params = config


class Estimator(object):

    def __init__(self,
                 data_config: DataConfig,
                 model_config: ModelConfig,
                 strategy_config: StrategyConfig,
                 backtest_engine_config: BacktestEngineConfig):

        # Configs.
        self.data_config = data_config
        self.model_config = model_config
        self.strategy_config = strategy_config
        self.backtest_engine_config = backtest_engine_config

        # Init data handler.
        self.data_handler = None
        self._init_data_handler()

        # Init model.
        self.model_class = None
        self.models = []
        self._init_model()

        # Init strategy.
        self.strategy = None
        self._init_strategy()

        # Init backtest engine.
        self.backtest_engine = None
        self._init_backtest_engine()

    def _init_data_handler(self):

        handler_class = self._load_class_with_module(self.data_config.handler_class,
                                                     self.data_config.handler_module,
                                                     'personae.contrib.data.handler')

        self.data_handler = handler_class(**self.data_config.handler_params)

    def _init_model(self):

        model_class = self._load_class_with_module(self.model_config.model_class,
                                                   self.model_config.model_module,
                                                   'personae.contrib.model.model')

        self.model_class = model_class

    def _init_strategy(self):

        strategy_class = self._load_class_with_module(self.strategy_config.strategy_class,
                                                      self.strategy_config.strategy_module,
                                                      'personae.contrib.strategy.strategy')

        self.strategy = strategy_class(**self.strategy_config.strategy_params)

    def _init_backtest_engine(self):

        backtest_class = self._load_class_with_module(self.backtest_engine_config.backtest_engine_class,
                                                      self.backtest_engine_config.backtest_engine_module,
                                                      'personae.contrib.backtest.engine')

        self.backtest_engine = backtest_class(**self.backtest_engine_config.backtest_engine_params)

    @staticmethod
    def _load_class_with_module(class_name, module_path, default_module_path):

        if module_path:
            module_spec = util.spec_from_file_location(name='Estimator', location=module_path)
            module = util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(default_module_path)

        object_class = getattr(module, class_name)

        return object_class

    def _train(self):

        if not self.model_config.model_rolling:
            # Get data handler.
            data_handler = self.data_handler  # type: BaseDataHandler

            # Get model.
            model = self.model_class(**self.model_config.model_params)  # type: BaseModel
            model.fit(
                x_train=data_handler.x_train,
                y_train=data_handler.y_train,
                x_validation=data_handler.x_validation,
                y_validation=data_handler.y_validation,
                w_train=data_handler.w_train,
                w_validation=data_handler.w_validation
            )

            self.models.append(model)

    def run(self):
        # Train model.
        pass


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser(prog='estimator')
    args_parser.add_argument('-c',
                             '--config_path',
                             required=True,
                             type=str,
                             help="yml config path indicates where to load config.")

    args = args_parser.parse_args()

    # Config path.
    config_path = args.config_path

    # Config manager.
    config_manager = ConfigManager(config_path)

    # Estimator.
    estimator = Estimator(config_manager.data_config,
                          config_manager.model_config,
                          config_manager.strategy_config,
                          config_manager.backtest_engine_config)

    # Run.
    estimator.run()
