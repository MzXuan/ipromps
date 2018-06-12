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


def sigmoid_increase():
    y_list = []
    T = np.linspace(-5,5,101)
    for t_ in T:
        y =  (1 / (1 + np.exp(-t_)))
        y_list.append(y)
    return np.array(y_list)

def sigmoid_decrease():
    y_list = []
    T = np.linspace(-5,5,101)
    for t_ in T:
        y =  (1 / (1 + np.exp(t_)))
        y_list.append(y)
    return np.array(y_list)

def main():
    task_id = 0
    test_index = 20

    obs_ratio_time_1 = np.array([0.0,  0.3, 0.6])
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
    
    mean_merge_traj_full = [None,None,None,None]
    sigma_merge_traj_full = [None,None,None,None]

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


        phase_increase = sigmoid_increase()
        phase_decrease = sigmoid_decrease()

        phase_stack = np.column_stack((phase_decrease,phase_increase))

        for task_idx, ipromps_idx in enumerate(ipromps_set):
            Phi = ipromps_idx.Phi
            sigma_merge_traj = []
            mean_merge_traj = []
            mean_traj = []
            sigma_traj = []
            mean_updated_traj = []
            sigma_updated_traj = []


            mean_updated = ipromps_idx.promps[1].meanW_nUpdated
            sigma_updated = ipromps_idx.promps[1].sigmaW_nUpdated

            for idx, phase_ in enumerate (phase_stack):
                
                phase_de = phase_[0]
                phase_in = phase_[1]
                phi = Phi.T[idx]

                if obs_ratio[0] == 0.0:
                    meanW0 = ipromps_idx.promps[1].meanW
                    sigmaW0 = ipromps_idx.promps[1].sigmaW
                    mean_point = np.dot(phi.T, meanW0)
                    sigma_point = np.dot(phi.T, np.dot(sigmaW0, phi))
                    
                else:
                    mean_point = mean_merge_traj_full[task_idx][idx]
                    sigma_point = sigma_merge_traj_full[task_idx][idx]

                mean_point_updated = np.dot(phi.T, mean_updated)
                sigma_point_updated = np.dot(phi.T, np.dot(sigma_updated, phi))

                sigma_point_activated = sigma_point/phase_de
                sigma_updated_activated = sigma_point_updated/phase_in

                sigma_divd_up = sigma_point_activated * sigma_updated_activated
                sigma_divdend = sigma_point_activated + sigma_updated_activated
                sigma_merge_point = sigma_divd_up / sigma_divdend

                mean_divd_up = mean_point*sigma_updated_activated + mean_point_updated*sigma_point_activated
                mean_merge_point =  mean_divd_up / sigma_divdend

                mean_traj.append(mean_point)
                sigma_traj.append(sigma_point)

                mean_updated_traj.append(mean_point_updated)
                sigma_updated_traj.append(sigma_point_updated)

                mean_merge_traj.append(mean_merge_point)
                sigma_merge_traj.append(sigma_merge_point)



            mean_merge_traj = np.array(mean_merge_traj)
            sigma_merge_traj = np.array(sigma_merge_traj)
            
            std_traj = 2*np.sqrt(sigma_traj).astype("float64")
            std_updated_traj = 2*np.sqrt(sigma_updated_traj).astype("float64")
            std_merge_traj = 2*np.sqrt(sigma_merge_traj).astype("float64")


            mean_merge_traj_full[task_idx]= mean_merge_traj
            sigma_merge_traj_full[task_idx]=sigma_merge_traj 


            ###########
            ###########

            t = np.linspace(0.0, 1.0, 101)
            plt.figure(fig_idx*10+task_idx)
            plt.subplot(2, 1, 1)
            plt.title('Gaussian distribution merge')
            plt.fill_between(t,mean_traj-std_traj, mean_traj+std_traj, color="b",label="orig_distribution", alpha=0.1)
            plt.plot(t,mean_traj, '--',color="b", linewidth=5,label ="orig_traj",alpha=0.1)

            plt.fill_between(t,mean_updated_traj-std_updated_traj, mean_updated_traj+std_updated_traj, color="y",label="updated_distribution", alpha=0.3)
            plt.plot(t,mean_updated_traj, '--',color="y", linewidth=5,label="updated_traj",alpha=0.3)

            plt.fill_between(t,mean_merge_traj-std_merge_traj, mean_merge_traj+std_merge_traj, color="g",label= "mixed_distribution", alpha=0.5)
            plt.plot(t,mean_merge_traj,'--', color="g", linewidth=5,label="merge_traj",alpha=0.5)
            plt.subplot(2, 1, 2)
            plt.title('Phase activation')
            plt.plot(t,phase_decrease,color="b")
            plt.plot(t,phase_increase,color="y")
            plt.legend()
            

    plt.show()








if __name__ == '__main__':
    main()