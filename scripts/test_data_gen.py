#!/usr/bin/python
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import griddata
import os
import ConfigParser
from sklearn.externals import joblib
import matplotlib.pyplot as plt


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

task_id = 0
test_index = 20
obs_ratio = 1.0
step = 0.2

def main():
    # error_vs_obs()
    save_data()

def error_vs_obs():
    traj_error_list = []
    obs = []
    end_eff_self_error = []
    # end_eff_human_error = []

    for i in range(1,6):
        obs_ratio = i*step
        obs.append(obs_ratio)

        print ('obs ratio: ', obs_ratio)
        # read test data
        obs_data_dict = datasets_raw[task_id][test_index]

        left_hand = obs_data_dict['left_hand']
        left_joints = obs_data_dict['left_joints']
        obs_data = np.hstack([left_hand, left_joints])
        timestamp = obs_data_dict['stamp']

        # filter the data
        obs_data = gaussian_filter1d(obs_data.T, sigma=sigma).T
        # preprocessing for the data
        obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
        # consider the unobserved info
        obs_data_post_arr[:, num_obs_joints:] = 0.0

        # choose the data
        num_obs = int(len(timestamp) * obs_ratio)
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

        # robot motion generation
        traj_full = []
        for ipromp_id, ipromp in enumerate(ipromps_set):
            [traj_time, traj] = ipromp.gen_real_traj(alpha_max_list[ipromp_id])
            traj = ipromp.min_max_scaler.inverse_transform(traj)
            robot_traj = traj[:, 6:9]
            human_traj = traj[:, 0:6]
            traj_full.append([human_traj, robot_traj])

        dist = 0.
        dist_end = 0.
        k=task_id

        robot_traj_pred = traj_full[k][1]
        human_gd = traj_full[k][0]
        left_joints = obs_data_dict['left_joints']
        grid = np.linspace(0, len(left_joints)-1,num=len(robot_traj_pred))
        left_joints = griddata(range(0,len(left_joints)), left_joints, grid, method='linear')


        for j in range(0,len(traj_full[k][1])):
            dist = dist + math.sqrt(math.pow((robot_traj_pred[j, 0] - left_joints[j, 0]),2)\
                    +math.pow((robot_traj_pred[j, 1] - left_joints[j, 1]),2)\
                        +math.pow((robot_traj_pred[j, 2] - left_joints[j, 2]),2))
            if j == (len(traj_full[k][1])-1):
                dist_end = math.sqrt(math.pow((robot_traj_pred[j, 0] - left_joints[j, 0]),2)\
                    +math.pow((robot_traj_pred[j, 1] - left_joints[j, 1]),2)\
                        +math.pow((robot_traj_pred[j, 2] - left_joints[j, 2]),2))
                rf_final = math.sqrt(math.pow((human_gd[-1, 0] - robot_traj_pred[-1, 0]), 2) \
                                     + math.pow((human_gd[-1, 1] - robot_traj_pred[-1, 1]), 2) \
                                     + math.pow((human_gd[-1, 2] - robot_traj_pred[-1, 2]), 2))

        traj_error_list.append(dist)
        end_eff_self_error.append(dist_end)

        print ("task id is:", task_id, "dist at phase: ",obs_ratio,"error along trajectory is ", dist, "with end-effector final error: ", dist_end,\
               "end_effector between robot and human is: ", rf_final)

        gd_final = math.sqrt(math.pow((human_gd[-1, 0] - left_joints[-1, 0]),2)\
                    +math.pow((human_gd[-1, 1] - left_joints[-1, 1]),2)\
                        +math.pow((human_gd[-1, 2] - left_joints[-1, 2]),2))
        # print (human_gd[-1, 0], human_gd[-1, 1], human_gd[-1, 2],left_joints[-1, 0],left_joints[-1, 1],left_joints[-1, 2])

    print("ground truth end-effector dist between human and robot is: ", gd_final)

    # print result
    plt.grid(True)


    fig = plt.figure(0)
    ax = fig.gca()
    ax.plot(obs,traj_error_list,
            '-', linewidth=5, color='blue', label='error along trajectory', alpha=1)
    ax.set_title('traj error vs observation ratio')
    ax.set_xlabel('ratio observed')
    ax.set_ylabel('error distance in meter')

    plt.grid(True)
    fig2 = plt.figure(1)
    ax2=fig2.gca()
    ax2.plot(obs,end_eff_self_error,
            '-', linewidth=5, color='green', label='final end-effector error vs robot ground truth', alpha=1.)
    ax2.set_title('end_effector error vs observation ratio')
    ax2.set_xlabel('ratio observed')
    ax2.set_ylabel('error distance in meter')

    plt.show()

    # save the conditional result
    print('Saving the post IProMPs...')
    joblib.dump(ipromps_set, os.path.join(datasets_path, 'pkl/ipromps_set_post_offline.pkl'))
    # save the robot traj
    print('Saving the robot traj...')
    joblib.dump([traj_full, obs_data_dict, num_obs], os.path.join(datasets_path, 'pkl/robot_traj_offline.pkl'))


def save_data():
    print ('obs ratio: ', obs_ratio)
    # read test data
    obs_data_dict = datasets_raw[task_id][test_index]

    left_hand = obs_data_dict['left_hand']
    left_joints = obs_data_dict['left_joints']
    obs_data = np.hstack([left_hand, left_joints])
    timestamp = obs_data_dict['stamp']

    # filter the data
    obs_data = gaussian_filter1d(obs_data.T, sigma=sigma).T
    # preprocessing for the data
    obs_data_post_arr = ipromps_set[0].min_max_scaler.transform(obs_data)
    # consider the unobserved info
    obs_data_post_arr[:, num_obs_joints:] = 0.0

    # choose the data
    num_obs = int(len(timestamp) * obs_ratio)
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
        robot_traj = traj[:, 6:9]
        human_traj = traj[:, 0:6]
        traj_full.append([human_traj, robot_traj])

        # # test: robot motion generation for task2
    # [traj_time2, traj2] = ipromps_set[2].gen_real_traj(alpha_max_list[2])
    # traj2 = ipromps_set[2].min_max_scaler.inverse_transform(traj2)
    # robot_traj2 = traj2[:, -3:]

    # save the conditional result
    print('Saving the post IProMPs...')
    joblib.dump(ipromps_set, os.path.join(datasets_path, 'pkl/ipromps_set_post_offline.pkl'))
    # save the robot traj
    print('Saving the robot traj...')
    joblib.dump([traj_full, obs_data_dict, num_obs], os.path.join(datasets_path, 'pkl/robot_traj_offline.pkl'))






if __name__ == '__main__':
    main()