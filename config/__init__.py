# coding=utf-8

import logging
import os

from datetime import datetime

DATETIME_LOAD_MODULE = datetime.now().strftime("%Y%m%d%H%M%S")

# Dir.
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
TEMP_DIR = os.path.join(PROJECT_DIR, 'temp')
DEFAULT_LOG_DIR = os.path.join(TEMP_DIR, 'log')
SAVED_MODEL_DIR = os.path.join(TEMP_DIR, 'model')
DATA_DIR = os.path.join(TEMP_DIR, 'data')
STOCK_DATA_DIR = os.path.join(DATA_DIR, 'stock')
CACHE_DIR = os.path.join(TEMP_DIR, 'cache')

# Backtest.
ROE = 'RoE'
CASH = 'cash'
BETA = 'beta'
CLOSE = 'close'
ALPHA = 'alpha'
AMOUNT = 'amount'
PROFITS = 'profits'
HOLDINGS = 'holdings'
BETA_ROE = 'beta_roe'
PRICE_DIFF = 'price_diff'
RETURN_RATE = 'return_rate'

# Data source.
KEY_NEED_NORMALIZE_DATA = 'need_normalize_data'

# Spider.
DEFAULT_INSTRUMENTS = ['600030', '600999', '000166', '600837', '601066', '002673', '000905']

# A3C
KEY_SESSION_CONFIG = 'session_config'
KEY_ROLE_TASK_MAP = 'role_task_map'
KEY_CLUSTER = 'cluster'

# Base model.
KEY_ROLLING_START_DATES = 'rolling_start_dates'
KEY_USE_SEQUENCE_DATA = 'use_sequence_data'
KEY_ROLLING_END_DATES = 'rolling_end_dates'
KEY_TRAIN_STEPS_LIMIT = 'train_steps_limit'
KEY_BATCH_SIZE_EVALUATE = 'batch_size_eval'
KEY_TRAIN_SAVE_STEPS = 'train_save_steps'
KEY_BATCH_SIZE_TRAIN = 'batch_size_train'
KEY_LOSS_FUNC_NAME = 'loss_func_name'
KEY_LEARNING_RATE = 'learning_rate'
KEY_DROPOUT_PROB = 'dropout_prob'
KEY_SEQ_LENGTH = 'seq_length'
KEY_SESSION = 'session'

# RL model.
KEY_TRAIN_EPISODES_LIMIT = 'train_episodes_limit'
KEY_EVAL_EPISODES = 'eval_episodes'
KEY_SAVE_EPISODES = 'save_episodes'
KEY_BUFFER_SIZE = 'buffer_size'
KET_EPSILON = 'epsilon'
KEY_GAMMA = 'gamma'
KEY_TAU = 'tau'

# Log level.
LEVEL_DEBUG = logging.DEBUG
LEVEL_INFO = logging.INFO
LEVEL_WARNING = logging.WARNING
