# coding=utf-8

import numpy as np

from personae.contrib.data.handler import PredictorDataHandler
# from personae.contrib.model.modeml import LightGBMModel
from personae.contrib.model.model import MLPModel

data_handler = PredictorDataHandler(processed_data_dir=r'/Users/shuyu/Desktop/Affair/Temp/data_tmp/processed',
                                    normalize_data=True,
                                    drop_nan_columns=True)
# data_handler = PredictorDataHandler(raw_data_dir=r'D:\Users\v-shuyw\data\ycz\data_sample\processed')

# model = LightGBMModel(**{
#     'num_threads': 4,
# })
#

x_space, y_space = data_handler.x_train.shape[1], 1

model = MLPModel(x_space=x_space, y_space=y_space, train_steps=10000)

model.fit(data_handler.x_train.values,
          data_handler.y_train.values,
          data_handler.x_validation.values,
          data_handler.y_validation.values)

# model.load()

print(np.corrcoef(data_handler.y_train, model.predict(data_handler.x_train.values)))
print(np.corrcoef(data_handler.y_validation, model.predict(data_handler.x_validation.values)))
print(np.corrcoef(data_handler.y_test, model.predict(data_handler.x_test.values)))
