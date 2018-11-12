# coding=utf-8

import config

from model.SL import MLP
from utility.data_source import TuShareDataSource

ts = TuShareDataSource()

model = MLP.Agent('MLP', ts.x_train.shape[1], ts.y_train.shape[1], **{config.KEY_TRAIN_STEPS_LIMIT: 30000})
model.train(ts.x_train, ts.y_train, ts.x_validate, ts.y_validate)
