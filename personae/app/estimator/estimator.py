# coding=utf-8

import yaml
import copy
import argparse
import importlib

from importlib import util

from personae.contrib.trainer.trainer import StaticTrainer, RollingTrainer


class ConfigManager(object):

    def __init__(self, config_path):
        # Config path.
        self.config_path = config_path
        # Load config.
        with open(self.config_path, 'r') as fp:
            self.config = yaml.load(fp)

        # Run.
        self.run_config = RunConfig(self.config.get('run', dict()))

        # Loader.
        self.loader_config = LoaderConfig(self.config.get('loader', dict()))

        #  Handler.
        self.handler_config = HandlerConfig(self.config.get('handler', dict()))

        # Model.
        self.model_config = ModelConfig(self.config.get('model', dict()))

        # Strategy.
        self.strategy_config = StrategyConfig(self.config.get('strategy', dict()))

        # Backtest.
        self.backtest_engine_config = BacktestEngineConfig(self.config.get('backtest', dict()))


class RunConfig(object):

    def __init__(self, config: dict):
        self.mode = config.get('mode', 'train')
        self.rolling = config.get('rolling', False)


class LoaderConfig(object):

    def __init__(self, config: dict):
        self.loader_class = config.get('class', 'BaseDataLoader')
        self.loader_module = config.get('module', None)
        self.loader_params = config.get('params', dict())


class HandlerConfig(object):

    def __init__(self, config: dict):
        self.handler_class = config.get('class', 'BaseDataHandler')
        self.handler_module = config.get('module', None)
        self.handler_params = config.get('params', dict())


class ModelConfig(object):

    def __init__(self, config: dict):
        self.model_class = config.get('class', 'GBMModel')
        self.model_module = config.get('module', None)
        self.model_params = config.get('params', dict())


class StrategyConfig(object):

    def __init__(self, config: dict):
        self.strategy_class = config.get('class', 'MLTopKEqualWeightStrategy')
        self.strategy_module = config.get('module', None)
        self.strategy_params = config.get('params', dict())


class BacktestEngineConfig(object):

    def __init__(self, config: dict):
        self.backtest_engine_class = config.get('class', 'BaseEngine')
        self.backtest_engine_module = config.get('module', None)
        self.backtest_engine_params = config.get('params', dict())


class Estimator(object):

    def __init__(self,
                 run_config: RunConfig,
                 loader_config: LoaderConfig,
                 handler_config: HandlerConfig,
                 model_config: ModelConfig,
                 strategy_config: StrategyConfig,
                 backtest_engine_config: BacktestEngineConfig):

        # Configs.
        self.run_config = run_config
        self.loader_config = loader_config
        self.handler_config = handler_config
        self.model_config = model_config
        self.strategy_config = strategy_config
        self.backtest_engine_config = backtest_engine_config

        # Loader.
        self.data_loader_class = self._load_class_with_module(self.loader_config.loader_class,
                                                              self.loader_config.loader_module,
                                                              'personae.contrib.data.loader')

        # Data handler.
        self.data_handler = None
        self.data_handler_class = self._load_class_with_module(self.handler_config.handler_class,
                                                               self.handler_config.handler_module,
                                                               'personae.contrib.data.handler')

        # Model.
        self.model_class = self._load_class_with_module(self.model_config.model_class,
                                                        self.model_config.model_module,
                                                        'personae.contrib.model.model')

        # Trainer.
        self.trainer = None

        # Strategy.
        self.strategy = None
        self.strategy_class = self._load_class_with_module(self.strategy_config.strategy_class,
                                                           self.strategy_config.strategy_module,
                                                           'personae.contrib.strategy.strategy')

        # Backtest engine.
        self.backtest_engine = None
        self.backtest_engine_class = self._load_class_with_module(self.backtest_engine_config.backtest_engine_class,
                                                                  self.backtest_engine_config.backtest_engine_module,
                                                                  'personae.contrib.backtest.engine')

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

    def run(self):
        # Data loader.
        # Data.
        stock_df = self.data_loader_class.load_data(
            data_type='stock',
            **self.loader_config.loader_params
        )

        bench_loader_params = copy.deepcopy(self.loader_config.loader_params)
        bench_loader_params.update({
            'market_type': 'all',
        })

        # Bench df.
        bench_df = self.data_loader_class.load_data(
            data_type='index',
            **bench_loader_params,
        )

        # Data handler.
        self.data_handler = self.data_handler_class(
            processed_df=stock_df,
            **self.handler_config.handler_params
        )

        # Trainer.
        if self.run_config.rolling:
            self.trainer = RollingTrainer(
                self.model_class,
                self.model_config.model_params,
                self.data_handler
            )
        else:
            self.trainer = StaticTrainer(
                self.model_class,
                self.model_config.model_params,
                self.data_handler
            )

        if self.run_config.mode == 'train':
            self.trainer.train()
        else:
            self.trainer.load()

        # Prediction.
        tar_position_scores = self.trainer.predict()

        # Strategy.
        self.strategy = self.strategy_class(
            tar_position_scores=tar_position_scores,
            **self.strategy_config.strategy_params
        )

        # Backtest engine.
        self.backtest_engine = self.backtest_engine_class(
            stock_df=stock_df,
            bench_df=bench_df,
            start_date=self.data_handler.test_start_date,
            end_date=self.data_handler.test_end_date,
            **self.backtest_engine_config.backtest_engine_params
        )

        # Start backtest.
        self.backtest_engine.run(self.strategy)
        self.backtest_engine.analyze()


