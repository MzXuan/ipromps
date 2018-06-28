#!/usr/bin/python
import sys
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import ConfigParser

# read conf file
file_path = os.path.dirname(__file__)
cp = ConfigParser.SafeConfigParser()
cp.read(os.path.join(file_path, '../cfg/models.cfg'))
# the datasets path
datasets_path = os.path.join(file_path, cp.get('datasets', 'path'))
datasets_pkl_path = os.path.join(datasets_path, 'pkl')
min_max_scaler_path = os.path.join(datasets_pkl_path, 'min_max_scaler.pkl')
min_max_scaler = joblib.load(min_max_scaler_path)

def main(left_hand_index,left_joints_index):
    csv_path = os.path.join(datasets_path, 'info/noise/multiModal_states.csv')
    print file_path
    print datasets_path
    print csv_path

    # read csv file
    data = pd.read_csv(csv_path)

    # extract the all signals data ORIENTATION
    # emg = data.values[:, 7:15].astype(float)
    left_hand = data.values[:, left_hand_index].astype(float)
    left_joints = data.values[:, left_joints_index].astype(float)  # robot ee actually

    # stack them as a big matrix
    full_data = np.hstack([left_hand, left_joints])
    full_data = min_max_scaler.transform(full_data)

    # compute the noise observation covariance matrix
    noise_cov = np.cov(full_data.T) #todo: is this correct?

    #todo: change noise to zero, for debug use
    for i in range(0,len(noise_cov)):
        for j in range(0,len(noise_cov[i])):
            if i==j:
                noise_cov[i][j]=0.01
            else:
                noise_cov[i][j]=0

    # save it in pkl
    joblib.dump(noise_cov, os.path.join(datasets_path, 'pkl/noise_cov.pkl'))
    print('Saved the noise covariance matrix successfully!')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        lh = [207, 208, 209, 197, 198, 199]
        lj = [317, 318, 319]
    elif len(sys.argv) == 2:
        lh = sys.argv[1]
        lh = lh.split(',')
        lh = map(int, lh)
        lj = [317, 318, 319]
    elif len(sys.argv) >= 3:
        lh = sys.argv[1]
        lj = sys.argv[2]

    print ('hand:', lh, 'robot:', lj)

    main(lh, lj)

