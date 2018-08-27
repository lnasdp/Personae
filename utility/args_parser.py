# coding=utf-8

import argparse
import config


spider_args_parser = argparse.ArgumentParser()
spider_args_parser.add_argument("-i", "--instruments", default=config.DEFAULT_INSTRUMENTS, nargs="+")
spider_args_parser.add_argument("-s", "--start_date", default='2017-01-01')
spider_args_parser.add_argument("-e", "--end_date", default='2018-04-01')
