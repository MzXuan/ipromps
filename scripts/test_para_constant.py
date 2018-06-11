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
def main():
    task_id = 0
    test_index = 20

    obs_ratio_time_1 = np.array([0,  0.3, 0.6])
    obs_ratio_time_2 = np.array([0.3,  0.6, 1.0])
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
        print('Computing the likelihood for each model under observations...')

        prob_task = []
        for ipromp in ipromps_set:
            prob_task_temp = ipromp.prob_obs()
            prob_task.append(prob_task_temp)
        idx_max_prob = np.argmax(prob_task)
        # idx_max_prob = 0 # a trick for testing
        print('The max fit model index is task %s' % task_name[idx_max_prob])

        # robot motion generation
        traj_full = []
        for ipromp_id, ipromp in enumerate(ipromps_set):
            [traj_time, traj] = ipromp.gen_real_traj(alpha_max_list[ipromp_id])
            traj = ipromp.min_max_scaler.inverse_transform(traj)
            robot_traj = traj[:, 1]
            human_traj= traj[:, 0]
            traj_full.append([human_traj, robot_traj])
 
    
        x = np.linspace(-10,10,101)
        phase_increase = sigmoid_increase(x)
        phase_decrease = sigmoid_decrease(x)
        
        para_need_list = []
        for task_idx, ipromps_idx in enumerate(ipromps_set):
            Phi =ipromps_idx.promps[1].Phi.astype("float64")
            meanW0 = ipromps_idx.promps[1].meanW.astype("float64")
            mean_traj = np.dot(Phi.T, meanW0).astype("float64")
            sigmaW0 = ipromps_idx.promps[1].sigmaW.astype("float64")
            sigma_traj = np.diag(np.dot(Phi.T, np.dot(sigmaW0, Phi))).astype("float64")
            std_traj = 2 * np.sqrt(sigma_traj).astype("float64")

            mean_updated = ipromps_idx.promps[1].meanW_nUpdated.astype("float64")
            mean_traj_updated = np.dot(Phi.T, mean_updated).astype("float64")
            sigma_updated = ipromps_idx.promps[1].sigmaW_nUpdated.astype("float64")
            sigma_traj_updated = np.diag(np.dot(Phi.T, np.dot(sigma_updated, Phi))).astype("float64")
            std_updated_traj = 2 * np.sqrt(sigma_traj_updated).astype("float64")
            tmp_dict = {'Phi':Phi, 'meanW0':meanW0, 'mean_traj':mean_traj, 'sigmaW0':sigmaW0, 'sigma_traj':sigma_traj, 'std_traj':std_traj, 'mean_updated':mean_updated,
            'mean_traj_updated': mean_traj_updated,'sigma_updated':sigma_updated,'sigma_traj_updated':sigma_traj_updated,'std_updated_traj':std_updated_traj}
            para_need_list.append(tmp_dict) 
        para_need_list = np.array(para_need_list)
        
        tmp_mix = []
        for task_idx, var in enumerate(para_need_list):
            tmp_sigma_1 = [None]*101
            tmp_sigma_2 = [None]*101
            sigma_merge = [None]*101
            mean_merge =  [None]*101
            std_merge_traj = [None]*101

            for idx,phase in enumerate(phase_decrease):
                tmp_sigma_1[idx] = var["sigma_traj"][idx]/phase
            for idx,phase in enumerate(phase_increase):
                tmp_sigma_2[idx] = var['sigma_traj_updated'][idx]/phase
    
            sigma_stack = np.column_stack((tmp_sigma_1,tmp_sigma_2))
            mean_stack = np.column_stack((var["mean_traj"],var["mean_traj_updated"]))

            for idx,num in enumerate(sigma_stack):
                tmp = num[0] * num[1]
                divdend =  num[0] + num[1]
                sigma_merge[idx] = tmp / divdend
            sigma_merge = np.array(sigma_merge)
            std_merge_traj = 2*np.sqrt(sigma_merge)

            for idx,num in enumerate(mean_stack):
                tmp = num[0] * tmp_sigma_2[idx] + num[1] * tmp_sigma_1[idx]
                divdend =  tmp_sigma_1[idx] + tmp_sigma_2[idx]
                mean_merge[idx] = tmp / divdend
            mean_merge = np.array(mean_merge)
            
            tmp_mix.append([mean_merge,sigma_merge])

            t = np.linspace(0.0, 1.0, 101)
            plt.figure(fig_idx*10+task_idx)
            plt.fill_between(t,var["mean_traj"]-var["std_traj"], var["mean_traj"]+var["std_traj"], color="b",label="orig_distribution", alpha=0.1)
            plt.plot(t,var["mean_traj"], '--',color="b", linewidth=5,label ="orig_traj",alpha=0.1)

            plt.fill_between(t,var["mean_traj_updated"]-var["std_updated_traj"], var["mean_traj_updated"]+var["std_updated_traj"], color="r",label="updated_distribution", alpha=0.3)
            plt.plot(t,var["mean_traj_updated"], '--',color="r", linewidth=5,label="updated_traj",alpha=0.3)

            plt.fill_between(t,mean_merge-std_merge_traj, mean_merge+std_merge_traj, color="g",label= "mixed_distribution", alpha=0.5)
            plt.plot(t,mean_merge,'--', color="g", linewidth=5,label="mixed_traj",alpha=0.5)
            plt.legend()
    
        

        for task_idx, ipromps_idx in enumerate(ipromps_set): 
            Phi = ipromps_idx.promps[1].Phi
            mean_W = np.dot(np.linalg.inv(np.dot(Phi, Phi.T)), np.dot(Phi, tmp_mix[task_idx][0].T)).T
            ipromps_idx.promps[1].meanW = mean_W 

            sigmaW0 = ipromps_idx.promps[1].sigmaW.astype("float64")
            sigma_full = np.dot(Phi.T, np.dot(sigmaW0, Phi)).astype("float64")
            for idx,var in enumerate(sigma_full):
                sigma_full[idx][idx] = tmp_mix[task_idx][1][idx]
            sigma_W = np.dot(np.dot(np.linalg.pinv(Phi.T),sigma_full),np.linalg.pinv(Phi))

            ipromps_idx.promps[1].sigmaW = sigma_W 
    # plt.show()
    



def compute_entropy():
    import scipy.stats



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