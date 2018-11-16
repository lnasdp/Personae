# coding=utf-8

import argparse

from personae.app.estimator.estimator import Estimator, ConfigManager

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
        config_manager.loader_config,
        config_manager.handler_config,
        config_manager.model_config,
        config_manager.strategy_config,
        config_manager.backtest_engine_config)

    # Run.
    estimator.run()
