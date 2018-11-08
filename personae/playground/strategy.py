# coding=utf-8

from personae.contrib.data.handler import PredictorDataHandler
from personae.contrib.model.model import BaseLightGBM


data_handler = PredictorDataHandler(raw_data_dir=r'D:\Users\v-shuyw\data\ycz\data_sample\processed')

model = BaseLightGBM(**{
    'num_threads': 4,
})

model.fit(data_handler.x_train,
          data_handler.y_train,
          data_handler.x_validation,
          data_handler.y_validation)



