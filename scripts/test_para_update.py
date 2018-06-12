#!/usr/bin/python
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import os
import ConfigParser
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from numpy.linalg import inv


# read the current file path
file_path = os.path.dirname(__file__)
# read model cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))

# load param
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
num_alpha_candidate = cp_models.getint('phase', 'num_phaseCandidate')
task_name_path = os.path.join(datasets_path, 'pkl/task_name_list.pkl')
task_name = joblib.load(task_name_path)
sigma = cp_models.getint('filter', 'sigma')
ipromps_set = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'))
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))
num_joints = cp_models.getint('datasets', 'num_joints')
num_obs_joints = cp_models.getint('datasets', 'num_obs_joints')

# read datasets cfg file
cp_datasets = ConfigParser.SafeConfigParser()
cp_datasets.read(os.path.join(datasets_path, './info/cfg/datasets.cfg'))
# read datasets params
data_index_sec = cp_datasets.items('index_17')
data_index = [map(int, task[1].split(',')) for task in data_index_sec]
info_n_idx = {
            'left_hand': [0],
            'left_joints': [1]
            }
def sigmoid_increase(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_decrease(x):
    return (1 / (1 + np.exp(x)))

def main():
    task_id = 0
    test_index = 20

    obs_ratio_time_1 = np.array([0,  0.5])
    obs_ratio_time_2 = np.array([0.5,  1.0])
    obs_ratio_time = np.column_stack((obs_ratio_time_1,obs_ratio_time_2))


    # read test data
    obs_data_dict = datasets_raw[task_id][test_index]

    left_hand = obs_data_dict['left_hand']
    left_joints = obs_data_dict['left_joints']
    obs_data = np.hstack([left_hand.reshape(len(left_hand),1), left_joints.reshape(len(left_joints),1)])
    timestamp = obs_data_dict['stamp']

    # preprocessing for the data
    obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
    # consider the unobserved info
    obs_data_post_arr[:, num_obs_joints:] = 0.0
    
    for fig_idx,obs_ratio in enumerate(obs_ratio_time):
        obs_data_post_arr =obs_data_post_arr[int(obs_ratio[0]*obs_data.shape[0]):int(obs_ratio[1]*obs_data.shape[0]),:]
        timestamp = timestamp[int(obs_ratio[0]*obs_data.shape[0]):int(obs_ratio[1]*obs_data.shape[0])]
        # phase estimation
        print('Phase estimating...')
        alpha_max_list = []
        for ipromp in ipromps_set:
            alpha_temp = ipromp.alpha_candidate(num_alpha_candidate)
            idx_max = ipromp.estimate_alpha(alpha_temp, obs_data_post_arr, timestamp)
            alpha_max_list.append(alpha_temp[idx_max]['candidate'])
            ipromp.set_alpha(alpha_temp[idx_max]['candidate'])

        # task recognition
        print('Adding via points in each trained model...')
        for task_idx, ipromp in enumerate(ipromps_set):
            for idx in range(len(timestamp)):
                ipromp.add_viapoint(timestamp[idx] / alpha_max_list[task_idx], obs_data_post_arr[idx, :])
            ipromp.param_update(unit_update=True)
            # ipromp.promps[1].plot_prior()
            # ipromp.promps[1].plot_nUpdated()
            # # plt.show()
        print('Computing the likelihood for each model under observations...')

        prob_task = []
        for ipromp in ipromps_set:
            prob_task_temp = ipromp.prob_obs()
            prob_task.append(prob_task_temp)
        idx_max_prob = np.argmax(prob_task)
        # idx_max_prob = 0 # a trick for testing
        print('The max fit model index is task %s' % task_name[idx_max_prob])


    
        x = np.linspace(-10,10,101)
        phase_increase = sigmoid_increase(x)
        phase_decrease = sigmoid_decrease(x)
        for task_idx, ipromps_idx in enumerate(ipromps_set):

            Phi =ipromps_idx.promps[1].Phi
            meanW0 = ipromps_idx.promps[1].meanW
            sigmaW0 = ipromps_idx.promps[1].sigmaW

            mean_traj = np.dot(Phi.T, meanW0)
            sigma_traj_full = np.dot(Phi.T, np.dot(sigmaW0, Phi))
            sigma_traj = np.diag(sigma_traj_full)
            std_traj = 2 * np.sqrt(sigma_traj)

            mean_updated = ipromps_idx.promps[1].meanW_nUpdated
            sigma_updated = ipromps_idx.promps[1].sigmaW_nUpdated

            mean_traj_updated = np.dot(Phi.T, mean_updated)
            sigma_traj_updated = np.diag(np.dot(Phi.T, np.dot(sigma_updated, Phi)))
            std_updated_traj = 2 * np.sqrt(sigma_traj_updated)

            tmp_sigma_1 = [None]*101
            tmp_sigma_2 = [None]*101
            sigma_merge_traj = [None]*101
            mean_merge_traj =  [None]*101
            std_merge_traj = [None]*101

            for idx,phase in enumerate(phase_decrease):
                tmp_sigma_1[idx] = sigma_traj[idx]/phase
            for idx,phase in enumerate(phase_increase):
                tmp_sigma_2[idx] = sigma_traj_updated[idx]/phase
    
            sigma_stack = np.column_stack((tmp_sigma_1,tmp_sigma_2))
            mean_stack = np.column_stack((mean_traj,mean_traj_updated))

            for idx,num in enumerate(sigma_stack):
                tmp = num[0] * num[1]
                divdend =  num[0] + num[1]
                sigma_merge_traj[idx] = tmp / divdend
            
            

            for idx,num in enumerate(mean_stack):
                tmp = num[0] * tmp_sigma_2[idx] + num[1] * tmp_sigma_1[idx]
                divdend =  tmp_sigma_1[idx] + tmp_sigma_2[idx]
                mean_merge_traj[idx] = tmp / divdend
            
            mean_merge_traj = np.array(mean_merge_traj)
            sigma_merge_traj = np.array(sigma_merge_traj)     
            std_merge_traj = 2*np.sqrt(sigma_merge_traj)

            t = np.linspace(0.0, 1.0, 101)
            plt.figure(fig_idx*10+task_idx)
            plt.fill_between(t,mean_traj-std_traj, mean_traj+std_traj, color="b",label="orig_distribution", alpha=0.1)
            plt.plot(t,mean_traj, '--',color="b", linewidth=5,label ="orig_traj",alpha=0.1)

            plt.fill_between(t,mean_traj_updated-std_updated_traj, mean_traj_updated+std_updated_traj, color="y",label="updated_distribution", alpha=0.3)
            plt.plot(t,mean_traj_updated, '--',color="y", linewidth=5,label="updated_traj",alpha=0.3)

            plt.fill_between(t,mean_merge_traj-std_merge_traj, mean_merge_traj+std_merge_traj, color="g",label= "mixed_distribution", alpha=0.5)
            plt.plot(t,mean_merge_traj,'--', color="g", linewidth=5,label="merge_traj",alpha=0.5)
            plt.legend()
            
            mean_W = np.dot(np.linalg.inv(np.dot(Phi, Phi.T)), np.dot(Phi, mean_merge_traj.T)).T
            ipromps_idx.promps[1].meanW = mean_W 

            # for idx,var in enumerate(sigma_full):
            #     sigma_full[idx][idx] = sigma_merge_traj[idx]
            sigma_W = np.dot(np.dot(np.linalg.pinv(Phi.T),sigma_traj_full),np.linalg.pinv(Phi))
            ipromps_idx.promps[1].sigmaW = sigma_W 
    plt.show()








if __name__ == '__main__':
    main()