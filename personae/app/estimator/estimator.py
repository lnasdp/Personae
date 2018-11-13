# coding=utf-8

import yaml
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
        # Set config.
        self.run_config = RunConfig(self.config.get('run', dict()))
        self.data_config = DataConfig(self.config.get('data', dict()))
        self.model_config = ModelConfig(self.config.get('model', dict()))
        self.strategy_config = StrategyConfig(self.config.get('strategy', dict()))
        self.backtest_engine_config = BacktestEngineConfig(self.config.get('backtest', dict()))


class RunConfig(object):

    def __init__(self, config: dict):
        self.mode = config.get('mode', 'train')
        self.rolling = config.get('rolling', False)


class DataConfig(object):

    def __init__(self, config: dict):
        self.handler_class = config.get('class', 'BaseDataHandler')
        self.handler_module = config.get('module', None)
        self.handler_params = config.get('params', dict())


class ModelConfig(object):

    def __init__(self, config: dict):
        self.model_class = config.get('class', 'LightGBMModel')
        self.model_module = config.get('module', None)
        self.model_params = config.get('params', dict())


class StrategyConfig(object):

    def __init__(self, config: dict):
        self.strategy_class = config.get('class', 'MLTopKStrategy')
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
                 data_config: DataConfig,
                 model_config: ModelConfig,
                 strategy_config: StrategyConfig,
                 backtest_engine_config: BacktestEngineConfig):

        # Configs.
        self.run_config = run_config
        self.data_config = data_config
        self.model_config = model_config
        self.strategy_config = strategy_config
        self.backtest_engine_config = backtest_engine_config

        # Data handler.
        self.data_handler = None
        self.data_handler_class = self._load_class_with_module(self.data_config.handler_class,
                                                               self.data_config.handler_module,
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
        # Data handler.
        self.data_handler = self.data_handler_class(**self.data_config.handler_params)

        # Trainer.
        if self.run_config.rolling:
            self.trainer = RollingTrainer(self.model_class,
                                          self.model_config.model_params,
                                          self.data_handler)
        else:
            self.trainer = StaticTrainer(self.model_class,
                                         self.model_config.model_params,
                                         self.data_handler)

        if self.run_config.mode == 'train':
            self.trainer.train()
        else:
            self.trainer.load()

        # Prediction.
        tar_positions_se = self.trainer.predict()

        # Strategy.
        self.strategy = self.strategy_class(tar_positions_se=tar_positions_se,
                                            **self.strategy_config.strategy_params)

        # Backtest engine.
        self.backtest_engine = self.backtest_engine_class(processed_data_dir=self.data_handler.processed_data_dir,
                                                          start_date=self.data_handler.test_start_date,
                                                          end_date=self.data_handler.test_end_date,
                                                          **self.backtest_engine_config.backtest_engine_params)

        # Start backtest.
        self.backtest_engine.run(self.strategy)
        self.backtest_engine.analyze()
        self.backtest_engine.plot()


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
    estimator = Estimator(
        config_manager.run_config,
        config_manager.data_config,
        config_manager.model_config,
        config_manager.strategy_config,
        config_manager.backtest_engine_config)

    # Run.
    estimator.run()
