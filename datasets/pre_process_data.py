#!/usr/bin/python
# data structure is: list(task1,2...)-->list(demo1,2...)-->dict(emg,imu,tf...)
import numpy as np
import operator
import pandas as pd
# from scipy.interpolate import griddata
from sklearn.externals import joblib
import glob
import os
import ConfigParser
from sklearn import preprocessing
from scipy.ndimage.filters import gaussian_filter1d

# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, './models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
len_norm = cp_models.getint('datasets', 'len_norm')
num_demo = cp_models.getint('datasets', 'num_demo')
num_joints = cp_models.getint('datasets', 'num_joints')
sigma = cp_models.getint('filter', 'sigma')

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
# read datasets params
data_index = [range(0,30),range(0,30)]
print data_index

# read data
def load_data():
    # datasets-related info
    task_path_list = glob.glob(os.path.join(datasets_path, 'raw/*'))
    task_path_list.sort()
    task_name_list = [task_path.split('/')[-1] for task_path in task_path_list]

    # load raw datasets
    datasets_raw = []
    for task_path in task_path_list:
        task_csv_path = os.path.join(task_path, 'left_hand/csv')
        print('Loading data from: ' + task_csv_path)
        demo_path_list = glob.glob(os.path.join(task_csv_path, '201*'))   # the prefix of dataset file
        demo_path_list.sort()
        demo_temp = []

        for demo_path in demo_path_list:
            data_csv = pd.read_csv(os.path.join(demo_path, 'multiModal_states.csv'))    # the file name of csv
            demo_temp.append({
                              'stamp': (data_csv.values[:, 2].astype(int)-data_csv.values[0, 2])*1e-9,
                              'left_hand': np.hstack([
                                  data_csv.values[:, [207,208,209,197,198,199]].astype(float),   # human left hand position
                                  data_csv.values[:, 7:15].astype(float),  # emg
                                  # data_csv.values[:, 19:23].astype(float),  # IMU
                                  ]),
                              'left_joints': data_csv.values[:, 317:320].astype(float),  # robot ee actually
                              'hand_pos': data_csv.values[:, 207:210].astype(float)
                              })
        datasets_raw.append(demo_temp)

    # filter the datasets: gaussian_filter1d
    datasets_filtered = []
    for task_idx, task_data in enumerate(datasets_raw):
        print('Filtering data of task: ' + task_name_list[task_idx])
        demo_norm_temp = []
        for demo_data in task_data:
            time_stamp = demo_data['stamp']
            # filter the datasets
            left_hand_filtered = gaussian_filter1d(demo_data['left_hand'].T, sigma=sigma).T
            left_joints_filtered = gaussian_filter1d(demo_data['left_joints'].T, sigma=sigma).T
            hand_position_filtered = gaussian_filter1d(demo_data['hand_pos'].T, sigma=sigma).T
            # append them to list
            demo_norm_temp.append({
                'alpha': time_stamp[-1],
                'left_hand': left_hand_filtered,
                'left_joints': left_joints_filtered,
                'hand_pos': hand_position_filtered
            })
        datasets_filtered.append(demo_norm_temp)

    return datasets_filtered, task_name_list

def regulize_channel(datasets,task_name_list):
    # regulize all the channel to 0-1
    datasets4train = []
    for task_idx, demo_list in enumerate(data_index):
        data = [datasets[task_idx][i] for i in demo_list]
        datasets4train.append(data)
    y_full = np.array([]).reshape(0, num_joints)
    for task_idx, task_data in enumerate(datasets4train):
        print('Preprocessing data for task: ' + task_name_list[task_idx])
        for demo_data in task_data:
            h = np.hstack([demo_data['left_hand'], demo_data['left_joints'], demo_data['hand_pos']])
            y_full = np.vstack([y_full, h])
    min_max_scaler = preprocessing.MinMaxScaler()
    datasets_norm_full = min_max_scaler.fit_transform(y_full)

    # regulize data
    len_sum = 0
    datasets_reg = []
    for task_idx in range(len(datasets4train)):
        datasets_temp = []
        for demo_idx in range(len(datasets4train[task_idx])):
            traj_len = len(datasets4train[task_idx][demo_idx]['hand_pos'])
            temp = datasets_norm_full[len_sum:len_sum+traj_len]
            datasets_temp.append({
                                    'left_hand': temp[:, 0:14],
                                    'left_joints': temp[:, 14:17],
                                    'hand_pos': temp[:,17:20],
                                    'alpha': datasets4train[task_idx][demo_idx]['alpha']})
            len_sum = len_sum + traj_len
        datasets_reg.append(datasets_temp)
    return datasets_reg,min_max_scaler

# # normalize length
# def normalize_length():
#     # resample the datasets
#     datasets_norm = []
#     for task_idx, task_data in enumerate(datasets_raw):
#         print('Resampling data of task: ' + task_name_list[task_idx])
#         demo_norm_temp = []
#         for demo_data in task_data:
#             time_stamp = demo_data['stamp']
#             grid = np.linspace(0, time_stamp[-1], len_norm)
#             # filter the datasets
#             left_hand_filtered = gaussian_filter1d(demo_data['left_hand'].T, sigma=sigma).T
#             left_joints_filtered = gaussian_filter1d(demo_data['left_joints'].T, sigma=sigma).T
#             # normalize the datasets
#             left_hand_norm = griddata(time_stamp, left_hand_filtered, grid, method='linear')
#             left_joints_norm = griddata(time_stamp, left_joints_filtered, grid, method='linear')
#             # append them to list
#             demo_norm_temp.append({
#                                     'alpha': time_stamp[-1],
#                                     'left_hand': left_hand_norm,
#                                     'left_joints': left_joints_norm
#                                     })
#         datasets_norm.append(demo_norm_temp)

def main():
    datasets_filtered, task_name_list = load_data()

    datasets_reg,min_max_scaler = regulize_channel(datasets_filtered, task_name_list)

    # save all the datasets
    print('Saving the datasets as pkl ...')
    joblib.dump(task_name_list, os.path.join(datasets_path, 'pkl/task_name_list.pkl'))
    joblib.dump(datasets_filtered, os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))
    joblib.dump(datasets_reg, os.path.join(datasets_path, 'pkl/datasets_reg.pkl'))
    joblib.dump(min_max_scaler, os.path.join(datasets_path, 'pkl/min_max_scaler.pkl'))

    #
    # # the finished reminder
    print('Loaded, filtered, normalized, preprocessed and saved the datasets successfully!!!')


if __name__ == '__main__':
    main()
