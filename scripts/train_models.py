#!/usr/bin/python
import numpy as np
import ipromps_lib
from sklearn.externals import joblib
import os
import ConfigParser
import pickle


# the current file path
file_path = os.path.dirname(__file__)

# read models cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
# read models params
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
len_norm = cp_models.getint('datasets', 'len_norm')
num_basis = cp_models.getint('basisFunc', 'num_basisFunc')
sigma_basis = cp_models.getfloat('basisFunc', 'sigma_basisFunc')
num_alpha_candidate = cp_models.getint('phase', 'num_phaseCandidate')

# the pkl data
datasets_pkl_path = os.path.join(datasets_path, 'pkl')
task_name_path = os.path.join(datasets_pkl_path, 'task_name_list.pkl')
datasets_norm_train_path = os.path.join(datasets_pkl_path, 'train_data_norm.pkl')
min_max_scaler_path = os.path.join(datasets_pkl_path, 'min_max_scaler.pkl')
noise_cov_path = os.path.join(datasets_pkl_path, 'noise_cov.pkl')


def main(h_dim=3,r_dim=3):
    # load the data from pkl
    task_name = pickle.load(open(task_name_path,"rb"))
    datasets_norm_preproc = pickle.load(open(datasets_norm_train_path,"rb"))
    min_max_scaler = pickle.load(open(min_max_scaler_path,"rb"))
    noise_cov = pickle.load(open(noise_cov_path,"rb"))


    # create iProMPs sets
    ipromps_set = [ipromps_lib.IProMP(num_joints=h_dim+r_dim, num_obs_joints=h_dim, num_basis=num_basis,
                                      sigma_basis=sigma_basis, num_samples=len_norm, sigmay=noise_cov,
                                      min_max_scaler=min_max_scaler, num_alpha_candidate=num_alpha_candidate)
                   for x in datasets_norm_preproc]

    # add demo for each IProMPs
    for idx, ipromp in enumerate(ipromps_set):
        print('Training the IProMP for task: %s...' % task_name[idx])
        # for demo_idx in datasets4train[idx]:
        for demo_idx in datasets_norm_preproc[idx]:
            demo_temp = np.hstack([demo_idx['left_hand'], demo_idx['left_joints']])
            ipromp.add_demonstration(demo_temp)   # spatial variance demo
            ipromp.add_alpha(demo_idx['alpha'])   # temporal variance demo

    # save the trained models
    print('Saving the trained models...')
    pickle.dump(ipromps_set, open(os.path.join(datasets_pkl_path, 'ipromps_set.pkl'),"wb"))

    print('Trained the IProMPs successfully!!!')


if __name__ == '__main__':
    main()
