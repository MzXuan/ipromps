#!/usr/bin/python
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import griddata
import os
import ConfigParser
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seperate_data


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
IPROMPS_SET = joblib.load(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'))



def main():
    datasets_test = joblib.load(os.path.join(datasets_path, 'pkl/test_data_raw.pkl'))
    _, _, h_dim, r_dim = seperate_data.get_feature_index()

    # initialize obs ratio
    step = 0.05
    obs_list = []
    for i in range(1,int(1/step)):
        obs_list.append(i*step)
    obs_list.append(1.0)
    print obs_list

    # test error in for loop
    for i,trajs in enumerate(datasets_test):
        for test_traj_raw in trajs:
            for obs_ratio in obs_list:
                obs_data = gen_offline_obs(test_traj_raw,obs_ratio)

    # print ("The ground truth class is: ", task_name[task_id])
    # error_vs_obs()
    # save_data()


def gen_offline_obs(test_traj_raw,obs_ratio):
    global IPROMPS_SET
    test_traj = np.hstack([test_traj_raw['left_hand'], test_traj_raw['left_joints']])
    num_obs_joints = test_traj_raw['left_hand'].shape[1]
    timestamp = test_traj_raw['stamp']
    # filter the data
    test_traj = gaussian_filter1d(test_traj.T, sigma=sigma).T
    # preprocessing for the data
    obs_data_post_arr = IPROMPS_SET[0].min_max_scaler.transform(test_traj)
    # consider the unobserved info

    obs_data_post_arr[:, num_obs_joints:] = 0.0

    # choose the data
    num_obs = int(len(timestamp) * obs_ratio)
    num_obs -= num_obs % 15
    obs_data_post_arr = obs_data_post_arr[0:num_obs:15, :]
    timestamp = timestamp[0:num_obs:15]
    obs_data_post_arr = obs_data_post_arr
    timestamp = timestamp

    return obs_data


def predict_result(obs_data):
    # phase estimation
    print('Phase estimating...')
    alpha_max_list = []
    for ipromp in IPROMPS_SET:
        alpha_temp = ipromp.alpha_candidate(num_alpha_candidate)
        idx_max = ipromp.estimate_alpha(alpha_temp, obs_data_post_arr, timestamp)
        alpha_max_list.append(alpha_temp[idx_max]['candidate'])
        ipromp.set_alpha(alpha_temp[idx_max]['candidate'])

    # task recognition
    print('Adding via points in each trained model...')
    for task_idx, ipromp in enumerate(IPROMPS_SET):
        for idx in range(len(timestamp)):
            ipromp.add_viapoint(timestamp[idx] / alpha_max_list[task_idx], obs_data_post_arr[idx, :])
        ipromp.param_update(unit_update=True)
    print('Computing the likelihood for each model under observations...')

    prob_task = []
    for ipromp in IPROMPS_SET:
        prob_task_temp = ipromp.prob_obs()
        prob_task.append(prob_task_temp)
    idx_max_prob = np.argmax(prob_task)
    # idx_max_prob = 0 # a trick for testing
    print('The max fit model index is task %s' % task_name[idx_max_prob])

    # robot motion generation
    traj_full = []
    for ipromp_id, ipromp in enumerate(IPROMPS_SET):
        [traj_time, traj] = ipromp.gen_real_traj(alpha_max_list[ipromp_id])
        traj = ipromp.min_max_scaler.inverse_transform(traj)
        human_traj = traj[:, 0:len_h]
        robot_traj = traj[:, len_h:len_h + len_r]

        traj_full.append([human_traj, robot_traj])

def error_vs_obs():
    traj_error_list = []
    obs = []
    end_eff_self_error = []
    # end_eff_human_error = []

    for i in range(1,6):
        obs_ratio = i*step
        obs.append(obs_ratio)

        print ('obs ratio: ', obs_ratio)

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

    # save the conditional result
    print('Saving the post IProMPs...')
    joblib.dump(IPROMPS_SET, os.path.join(datasets_path, 'pkl/ipromps_set_post_offline.pkl'))
    # save the robot traj
    print('Saving the robot traj...')
    joblib.dump([traj_full, obs_data_dict, num_obs], os.path.join(datasets_path, 'pkl/robot_traj_offline.pkl'))


def visulize_result():
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


# def save_data():
#
#     # save the conditional result
#     print('Saving the post IProMPs...')
#     joblib.dump(ipromps_set, os.path.join(datasets_path, 'pkl/ipromps_set_post_offline.pkl'))
#     # save the robot traj
#     print('Saving the robot traj...')
#     joblib.dump([traj_full, obs_data_dict, num_obs], os.path.join(datasets_path, 'pkl/robot_traj_offline.pkl'))

if __name__ == '__main__':
    main()