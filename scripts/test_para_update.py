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
y_list = []
def main():
    task_id = 0
    test_index = 20
    obs_ratio = 0.4

    # read test data
    obs_data_dict = datasets_raw[task_id][test_index]

    left_hand = obs_data_dict['left_hand']
    left_joints = obs_data_dict['left_joints']
    obs_data = np.hstack([left_hand.reshape(len(left_hand),1), left_joints.reshape(len(left_joints),1)])
    timestamp = obs_data_dict['stamp']

    # filter the data
    obs_data = gaussian_filter1d(obs_data.T, sigma=sigma).T
    # preprocessing for the data
    obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
    # consider the unobserved info
    obs_data_post_arr[:, num_obs_joints:] = 0.0

    # choose the data
    num_obs = int(len(timestamp)*obs_ratio)
    num_obs -= num_obs % 15
    obs_data_post_arr = obs_data_post_arr[0:num_obs:15, :]
    timestamp = timestamp[0:num_obs:15]
    obs_data_post_arr = obs_data_post_arr
    timestamp = timestamp

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
    print('Computing the likelihood for each model under observations...')

    prob_task = []
    for ipromp in ipromps_set:
        prob_task_temp = ipromp.prob_obs()
        prob_task.append(prob_task_temp)
    idx_max_prob = np.argmax(prob_task)
    # idx_max_prob = 0 # a trick for testing
    print('The max fit model index is task %s' % task_name[idx_max_prob])

    # # robot motion generation
    # [traj_time, traj] = ipromps_set[idx_max_prob].gen_real_traj(alpha_max_list[idx_max_prob])
    # traj = ipromps_set[idx_max_prob].min_max_scaler.inverse_transform(traj)
    # robot_traj = traj[:, -3:]

    # robot motion generation
    traj_full = []
    for ipromp_id, ipromp in enumerate(ipromps_set):
        [traj_time, traj] = ipromp.gen_real_traj(alpha_max_list[ipromp_id])
        traj = ipromp.min_max_scaler.inverse_transform(traj)
        robot_traj = traj[:, 1]
        human_traj= traj[:, 0]
        traj_full.append([human_traj, robot_traj])
    # plot_human_prior()
    get_parameter()
    # plot_robot_prior()
    
    # plt.plot(y_list[0])
    # # compute_entropy()
    # # plot_human_traj(traj_full)
    # plot_robot_traj(traj_full)
    plt.show()
# plot the prior distribution
    

def compute_entropy():
    import scipy.stats

def get_parameter():
    # from scipy.stats import norm,entropy
    # y_list = []
    
    x = np.linspace(-10,10,101)
    phase_increase = sigmoid_increase(x)
    phase_decrease = sigmoid_decrease(x)
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        Phi =ipromps_idx.promps[1].Phi
        meanW0 = ipromps_idx.promps[1].meanW
        mean_traj = np.dot(Phi.T, meanW0)
        sigmaW0 = ipromps_idx.promps[1].sigmaW
        sigma_traj = np.diag(np.dot(Phi.T, np.dot(sigmaW0, Phi)))
        std_traj = 2 * np.sqrt(sigma_traj)
        mean_updated = ipromps_idx.promps[1].meanW_nUpdated
        mean_traj_updated = np.dot(Phi.T, mean_updated)
        sigma_updated = ipromps_idx.promps[1].sigmaW_nUpdated
        sigma_traj_updated = np.diag(np.dot(Phi.T, np.dot(sigma_updated, Phi)))
        std_updated_traj = 2 * np.sqrt(sigma_traj_updated)

        tmp_sigma_1 = [None]*101
        tmp_sigma_2 = [None]*101
        sigma_merge = [None]*101
        mean_merge =  [None]*101
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
            sigma_merge[idx] = tmp / divdend
        
        std_merge_traj = 2*np.sqrt(sigma_merge)

        for idx,num in enumerate(mean_stack):
            tmp = num[0] * tmp_sigma_2[idx] + num[1] * tmp_sigma_1[idx]
            divdend =  tmp_sigma_1[idx] + tmp_sigma_2[idx]
            mean_merge[idx] = tmp / divdend
        mean_merge = np.array(mean_merge)

        t = np.linspace(0.0, 1.0, 101)
        plt.figure(task_idx)
        plt.fill_between(t,mean_traj-std_traj, mean_traj+std_traj, color="b",label="orig_distribution", alpha=0.1)
        plt.plot(t,mean_traj, '--',color="b", linewidth=5,label ="orig_traj")

        plt.fill_between(t,mean_traj_updated-std_updated_traj, mean_traj_updated+std_updated_traj, color="y",label="updated_distribution", alpha=0.3)
        plt.plot(t,mean_traj_updated, '--',color="y", linewidth=5,label="updated_traj")

        plt.fill_between(t,mean_merge-std_merge_traj, mean_merge+std_merge_traj, color="g",label= "mixed_distribution", alpha=0.5)
        plt.plot(t,mean_merge,'--', color="g", linewidth=5,label="mixed_traj")
        plt.legend()
        
        # print mean_merge
    
            


        
        # for idx,phase enumerate(cov_merge)
        # cov_merge =  inv( inv(np.divide(sigmaW0,phase_decrease)) + inv(np.divide(sigma_updated, phase_increase)))
        # mean_merge = np.dot(inv(np.divide(sigmaW0,phase_decrease)),meanW0 ) + np.dot(inv(np.divide(sigma_updated,phase_increase)),mean_updated)
        # mean_merge = np.dot(inv(cov_merge),mean_merge)
        # y = np.dot(Phi.T,mean_merge)
        # y_list.append(y)


def sigmoid_increase(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_decrease(x):
    return (1 / (1 + np.exp(x)))


def plot_human_prior(num=0):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx+num)
        fig.suptitle('the prior of ' + "human" + ' for ' + task_name[task_idx] + ' model')
        ipromps_idx.promps[0].plot_prior()
        ipromps_idx.promps[0].plot_nUpdated()
        

def plot_robot_prior(num=10):
    for task_idx, ipromps_idx in enumerate(ipromps_set):
        fig = plt.figure(task_idx+num)
        fig.suptitle('the prior of ' + "robot" + ' for ' + task_name[task_idx] + ' model')
        ipromps_idx.promps[1].plot_prior(b_regression=False) 
        ipromps_idx.promps[1].plot_nUpdated()
        

def plot_human_traj(traj_full,num=10):
    fig = plt.figure(num)
    plt.subplot(2, 2, 1)
    plt.plot(traj_full[0][0],label="human_traj")
    plt.subplot(2, 2, 2)
    plt.plot(traj_full[1][0],label="human_traj")
    plt.subplot(2, 2, 3)
    plt.plot(traj_full[2][0],label="human_traj")
    plt.subplot(2, 2, 4)
    plt.plot(traj_full[3][0],label="human_traj")

def plot_robot_traj(traj_full,num=15):
    fig = plt.figure(num)       
    plt.subplot(2, 2, 1)
    plt.plot(traj_full[0][1],label="robot_traj")
    plt.subplot(2, 2, 2)
    plt.plot(traj_full[1][1],label="robot_traj")
    plt.subplot(2, 2, 3)
    plt.plot(traj_full[2][1],label="robot_traj")
    plt.subplot(2, 2, 4)
    plt.plot(traj_full[3][1],label="robot_traj")

if __name__ == '__main__':
    main()