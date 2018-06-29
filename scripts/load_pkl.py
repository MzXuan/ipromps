#!/usr/bin/python
import sys
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.externals import joblib
import glob
import os
import ConfigParser
from sklearn import preprocessing

# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]
# data_index = [range(0,44),range(0,44)]
print data_index


def main():
    # the pkl data
    datasets_pkl_path = os.path.join(datasets_path, 'pkl')
    task_name_path = os.path.join(datasets_pkl_path, 'task_name_list.pkl')
    datasets_norm_preproc_path = os.path.join(datasets_pkl_path, 'datasets_norm_preproc.pkl')
    min_max_scaler_path = os.path.join(datasets_pkl_path, 'min_max_scaler.pkl')
    noise_cov_path = os.path.join(datasets_pkl_path, 'noise_cov.pkl')

if __name__ == '__main__':
    main()