# coding=utf-8

import os

from datetime import datetime

DATETIME_LOAD_MODULE = datetime.now().strftime("%Y%m%d%H%M%S")

PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))

TEMP_DIR = os.path.join(PROJECT_DIR, 'temp')

DEFAULT_LOG_DIR = os.path.join(TEMP_DIR, 'log')
SAVED_MODEL_DIR = os.path.join(TEMP_DIR, 'model')

KEY_MODE = 'mode'
KEY_CALLBACK = 'callback'
KEY_CSV_TYPE = 'csv_type'
KEY_FREQUENCY = 'frequency'
KEY_USE_CACHE = 'use_cache'
KEY_MODEL_NAME = 'model_name'
KEY_LABEL_NAME = 'label_name'

KEY_TESTING_END_DATE = 'testing_end_date'
KEY_TRAINING_END_DATE = 'training_end_date'
KEY_FORCE_RELOAD_DATA = 'force_reload_data'
KEY_USE_SEQUENCE_DATA = 'use_sequence_data'
KEY_USE_VALIDATION_SET = 'use_validation_set'
KEY_TESTING_START_DATE = 'testing_start_date'
KEY_VALIDATING_END_DATE = 'validating_end_Date'
KEY_TRAINING_DATA_RATIO = 'training_data_ratio'
KEY_TRAINING_START_DATE = 'training_start_date'
KEY_VALIDATING_START_DATE = 'validating_start_date'
KEY_MIX_TRAINING_SET_PERCENT = 'mix_tra_set_percent'
KEY_MIX_VALIDATING_SET_PERCENT = 'mix_val_set_percent'
KEY_TOLERATE_PERFORMANCE_MULTIPLIER = 'tolerate_performance_multiplier'
KEY_MIX_TRAINING_AND_VALIDATION_SET = 'mix_training_and_validation_set'


KEY_SESSION_CONFIG = 'session_config'


KEY_ROLE_TASK_MAP = 'role_task_map'


# Base model.
KEY_ROLLING_START_DATES = 'rolling_start_dates'
KEY_ROLLING_END_DATES = 'rolling_end_dates'
KEY_TRAIN_STEPS_LIMIT = 'train_steps_limit'
KEY_BATCH_SIZE_EVALUATE = 'batch_size_eval'
KEY_TRAIN_SAVE_STEPS = 'train_save_steps'
KEY_BATCH_SIZE_TRAIN = 'batch_size_train'
KEY_LEARNING_RATE = 'learning_rate'
KEY_DROPOUT_PROB = 'dropout_prob'
KEY_SEQ_LENGTH = 'seq_length'
KEY_SESSION = 'session'
KEY_CLUSTER = 'cluster'

# RL model.
KEY_TRAIN_EPISODES_LIMIT = 'train_episodes_limit'
KEY_EVAL_EPISODES = 'eval_episodes'
KEY_SAVE_EPISODES = 'save_episodes'
KEY_BUFFER_SIZE = 'buffer_size'

KET_EPSILON = 'epsilon'
KEY_GAMMA = 'gamma'
KEY_TAU = 'tau'


CODE_SUCCESS = 'Success'
CODE_FAILURE = 'Failure'
