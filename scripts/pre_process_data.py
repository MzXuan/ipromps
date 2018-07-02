#!/usr/bin/python
# data structure is: list(task1,2...)-->list(demo1,2...)-->dict(emg,imu,tf...)
import numpy as np
import operator
import pandas as pd
from scipy.interpolate import griddata
from sklearn.externals import joblib
import glob
import os
import ConfigParser
from sklearn import preprocessing
from scipy.ndimage.filters import gaussian_filter1d
import pickle

# the current file path
FILE_PATH = os.path.dirname(__file__)
TASK_NAME_LIST = []

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(FILE_PATH, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(FILE_PATH, cp_models.get('datasets', 'path'))
len_norm = cp_models.getint('datasets', 'len_norm')
# num_demo = cp_models.getint('datasets', 'num_demo')
sigma = cp_models.getint('filter', 'sigma')

TASK_NAME_LIST = cp_models.get('datasets', 'class_name')
TASK_NAME_LIST = TASK_NAME_LIST.split(',')

# # read datasets cfg file
# cp_datasets = ConfigParser.SafeConfigParser()
# cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))


# read data
def load_data():
    global TASK_NAME_LIST

    # datasets-related info
    task_path_list = []
    for task in TASK_NAME_LIST:
        task_path_list.append(os.path.join(datasets_path, 'raw/'+task))


    # load raw datasets
    datasets_raw = []
    for task_path in task_path_list:

        ## get file path
        #read human features
        task_csv_path = os.path.join(task_path, 'left_hand/csv')
        print('Loading data from: ' + task_csv_path)
        hdemo_path_list = glob.glob(os.path.join(task_csv_path, '201*'))   # the prefix of dataset file
        hdemo_path_list.sort()
        demo_temp = []

        #read seperate robot trajectory
        robot_csv_path = os.path.join(task_path, 'left_joints/csv')
        print('Loading data from: ' + robot_csv_path)
        rdemo_path_list = glob.glob(os.path.join(robot_csv_path, '201*'))  # the prefix of dataset file
        rdemo_path_list.sort()

        if len(hdemo_path_list) !=len(rdemo_path_list):
            print ("length of data is not equal, please check the data.")
            return False

        ## read data
        for hdemo_path,rdemo_path in zip(hdemo_path_list,rdemo_path_list):

            #read data from csv file
            hdata_csv = pd.read_csv(os.path.join(hdemo_path, 'multiModal_states.csv'))    # the file name of csv
            rdata_csv = pd.read_csv(os.path.join(rdemo_path, 'multiModal_states.csv'))

            time_stamp = (hdata_csv.values[:, 2].astype(int)-hdata_csv.values[0, 2])*1e-9
            human_traj = np.hstack([
                                  hdata_csv.values[:, [207,208,209,197,198,199]].astype(float),   # human left hand position
                                  hdata_csv.values[:, 7:15].astype(float),  # emg
                                  hdata_csv.values[:, 19:23].astype(float)  # IMU
                                    ])
            rt = (rdata_csv.values[:, 2].astype(int) - rdata_csv.values[0, 2]) * 1e-9  # robot time stamp
            robot_traj = rdata_csv.values[:, 207:210].astype(float)

            #equalize length of human_traj and robot_traj
            if len(robot_traj) != len(human_traj):
                grid = np.linspace(0, rt[-1] , len(time_stamp))
                robot_traj = griddata(rt, robot_traj, grid, method='nearest')  #give linear here if we have all data


            demo_temp.append({
                              'stamp': time_stamp,
                              'alpha': time_stamp[-1],
                              'left_hand': human_traj,
                              'left_joints': robot_traj,  # robot ee actually
                              })
        datasets_raw.append(demo_temp)

    return datasets_raw


def get_feature_index(csv_path='../cfg/models.cfg'):
    global SELECT_CLASS
    # read models params
    cp_models = ConfigParser.SafeConfigParser()
    cp_models.read(os.path.join(FILE_PATH, csv_path))

    emg = cp_models.get("csv_parse", 'emg')
    imu = cp_models.get("csv_parse", 'imu')
    elbow_position = cp_models.get("csv_parse", 'elbow_position')

    h_feature_index = [0, 1, 2]  # wrist position
    r_feature_index = [0, 1, 2]  # end effector position
    if elbow_position == 'enable':
        print('enable elbow position')
        h_feature_index += [3, 4, 5]
    if emg == 'enable':
        print('enable emg data')
        h_feature_index += [6, 7, 8, 9, 10, 11, 12, 13]
    if imu == 'enable':
        print('enable imu data')
        h_feature_index += [14, 15, 16, 17]

    h_dim = len(h_feature_index)
    r_dim = len(r_feature_index)
    print('human feature dim:', h_dim, 'robot feature dim:', r_dim)

    return h_feature_index, r_feature_index, h_dim, r_dim

def select_data(datasets,h_feature_index,r_feature_index):
    ## generate new dataset according to the seletive feature
    data_select = []
    for i in range(0,len(datasets)):
        traj_temp = []
        for traj in datasets[i]:
            traj_temp.append({
                'stamp': traj['stamp'],
                'alpha': traj['alpha'],
                'left_hand': traj['left_hand'][:, h_feature_index],
                'left_joints': traj['left_joints'][:, r_feature_index]
            })
        data_select.append(traj_temp)
    return data_select

def filter_data(datasets):
    global TASK_NAME_LIST
    ## filter the datasets: gaussian_filter1d
    datasets_filtered = []
    for task_idx, task_data in enumerate(datasets):
        print('Filtering data of task: ' + TASK_NAME_LIST[task_idx])
        demo_norm_temp = []

        for demo_data in task_data:
            time_stamp = demo_data['stamp']
            # filter the datasets
            left_hand_filtered = gaussian_filter1d(demo_data['left_hand'].T, sigma=sigma).T
            left_joints_filtered = gaussian_filter1d(demo_data['left_joints'].T, sigma=sigma).T
            # hand_position_filtered = gaussian_filter1d(demo_data['hand_pos'].T, sigma=sigma).T
            # append filtered trajectory to list
            demo_norm_temp.append({
                'stamp': time_stamp,
                'alpha': time_stamp[-1],
                'left_hand': left_hand_filtered,
                'left_joints': left_joints_filtered,
                # 'hand_pos': hand_position_filtered
            })
        datasets_filtered.append(demo_norm_temp)
    return datasets_filtered

def regulize_channel(datasets,h_dim,r_dim):
    global TASK_NAME_LIST
    # regulize all the channel to 0-1
    y_full = np.array([]).reshape(0, h_dim+r_dim)
    for task_idx, task_data in enumerate(datasets):
        print('Preprocessing data for task: ' + TASK_NAME_LIST[task_idx])
        for demo_data in task_data:
            h = np.hstack([demo_data['left_hand'], demo_data['left_joints']])
            y_full = np.vstack([y_full, h])
    min_max_scaler = preprocessing.MinMaxScaler()
    datasets_norm_full = min_max_scaler.fit_transform(y_full)

    # revert back to different classes
    len_sum = 0
    datasets_reg = []
    for task_idx in range(len(datasets)):
        datasets_temp = []
        for demo_idx in range(len(datasets[task_idx])):
            traj_len = len(datasets[task_idx][demo_idx]['left_joints'])
            time_stamp = datasets[task_idx][demo_idx]['stamp']
            temp = datasets_norm_full[len_sum:len_sum+traj_len]
            datasets_temp.append({
                                    'stamp': time_stamp,
                                    'alpha': datasets[task_idx][demo_idx]['alpha'],
                                    'left_hand': temp[:, 0:h_dim],
                                    'left_joints': temp[:, h_dim:h_dim+r_dim]
                                    })
            len_sum = len_sum + traj_len
        datasets_reg.append(datasets_temp)
    return datasets_reg,min_max_scaler

## normalize length
def normalize_length(datasets):
    global TASK_NAME_LIST
    # resample the datasets
    datasets_norm = []
    for task_idx, task_data in enumerate(datasets):
        print('Resampling data of task: ' + TASK_NAME_LIST[task_idx])
        demo_norm_temp = []
        for demo_data in task_data:
            time_stamp = demo_data['stamp']
            grid = np.linspace(0, time_stamp[-1], len_norm)
            # filter the datasets
            left_hand_filtered = gaussian_filter1d(demo_data['left_hand'].T, sigma=sigma).T
            left_joints_filtered = gaussian_filter1d(demo_data['left_joints'].T, sigma=sigma).T
            # normalize the datasets
            left_hand_norm = griddata(time_stamp, left_hand_filtered, grid, method='linear')
            left_joints_norm = griddata(time_stamp, left_joints_filtered, grid, method='linear')
            # append them to list
            demo_norm_temp.append({
                                    'stamp': 0,
                                    'alpha': time_stamp[-1],
                                    'left_hand': left_hand_norm,
                                    'left_joints': left_joints_norm
                                    })
        datasets_norm.append(demo_norm_temp)
    return datasets_norm

def main():
    ## read raw data
    datasets_raw = load_data()

    ## select feature
    h_feature_index, r_feature_index, h_dim, r_dim = get_feature_index()
    datasets_raw_select = select_data(datasets_raw,h_feature_index, r_feature_index)

    ## filtered select data
    datasets_filtered = filter_data(datasets_raw_select)

    ## regulize to 0-1, and normalize length
    datasets_reg,min_max_scaler = regulize_channel(datasets_filtered,h_dim,r_dim)
    datasets_norm = normalize_length(datasets_reg)

    ## save all the datasets
    print('Saving the datasets as pkl ...')
    # joblib.dump(TASK_NAME_LIST, os.path.join(datasets_path, 'pkl/task_name_list.pkl'))
    # joblib.dump(datasets_raw, os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
    # joblib.dump(datasets_raw_select, os.path.join(datasets_path, 'pkl/datasets_raw_select.pkl'))
    # joblib.dump(datasets_filtered, os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'))
    # joblib.dump(datasets_reg, os.path.join(datasets_path, 'pkl/datasets_reg.pkl'))
    # joblib.dump(min_max_scaler, os.path.join(datasets_path, 'pkl/min_max_scaler.pkl'))
    # joblib.dump(datasets_norm, os.path.join(datasets_path, 'pkl/datasets_norm.pkl'))


    pickle.dump(TASK_NAME_LIST, open(os.path.join(datasets_path, 'pkl/task_name_list.pkl'),"wb"))
    pickle.dump(datasets_raw, open(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'),"wb"))
    pickle.dump(datasets_raw_select, open(os.path.join(datasets_path, 'pkl/datasets_raw_select.pkl'),"wb"))
    pickle.dump(datasets_filtered, open(os.path.join(datasets_path, 'pkl/datasets_filtered.pkl'),"wb"))
    pickle.dump(datasets_reg, open(os.path.join(datasets_path, 'pkl/datasets_reg.pkl'),"wb"))
    pickle.dump(min_max_scaler, open(os.path.join(datasets_path, 'pkl/min_max_scaler.pkl'),"wb"))
    pickle.dump(datasets_norm, open(os.path.join(datasets_path, 'pkl/datasets_norm.pkl'),"wb"))
    # # the finished reminder
    print('Loaded, filtered, normalized, preprocessed and saved the datasets successfully!!!')


if __name__ == '__main__':
    main()
