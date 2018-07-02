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
import pickle


# read the current file path
file_path = os.path.dirname(__file__)
# read model cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))

# load param
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))
num_alpha_candidate = cp_models.getint('phase', 'num_phaseCandidate')
task_name_path = os.path.join(datasets_path, 'pkl/task_name_list.pkl')
task_name = pickle.load(open(task_name_path,"rb"))
sigma = cp_models.getint('filter', 'sigma')
IPROMPS_SET = pickle.load(open(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'),"rb"))

def main():
    global IPROMPS_SET

    datasets_test = pickle.load(open(os.path.join(datasets_path, 'pkl/test_data_raw.pkl'),"rb"))
    _, _, h_dim, r_dim = seperate_data.get_feature_index()

    # initialize obs ratio
    step = 0.1
    obs_list = []
    for i in range(1,int(1/step)):
        obs_list.append(i*step)
    obs_list.append(1.0)
    # print obs_list
    handover_pos_error = np.zeros(len(obs_list))
    human_predict_error = np.zeros(len(obs_list))
    predict_datasets = []
    count = 0
    # test error in for loop
    for i,trajs in enumerate(datasets_test):
        predict_temp = []
        for test_traj_raw in trajs:
            test_traj, timestamp = prepross_traj(test_traj_raw)
            print('The ground truth task %s' % task_name[i])
            IPROMPS_SET = pickle.load(open(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'),"rb"))  # refresh
            obs_ratio_s=0.0
            for j,obs_ratio_e in enumerate(obs_list):

                obs_data,obs_timestamp = gen_offline_obs(test_traj,timestamp,obs_ratio_s,obs_ratio_e)

                # update batch by batch
                obs_ratio_s=obs_ratio_e
                # obs_ratio_s = 0

                # # update directly from start
                # IPROMPS_SET = pickle.load(open(os.path.join(datasets_path, 'pkl/ipromps_set.pkl'),"rb"))

                pred_class,predict_data = predict_result(obs_data,obs_timestamp,h_dim,r_dim)
                #compute robot handover error
                pos_gd = test_traj_raw['left_joints'][-1]
                pos_pred = predict_data[pred_class][1][-1]
                handover_pos_error[j] = handover_pos_error[j]+Cartesian_dist(pos_gd,pos_pred)
                #compute human prediction error
                h_pos_gd = test_traj_raw['left_hand'][-1,0:3]
                h_pred = predict_data[pred_class][0][-1,0:3]
                human_predict_error[j] = human_predict_error[j]+Cartesian_dist(h_pos_gd,h_pred)
                #todo: compute classification rate
            count+=1
            print("tested trajectories:", count)

    handover_pos_error = np.divide(handover_pos_error,count)
    human_predict_error = np.divide(human_predict_error,count)

    # print and visualize result
    print ('ratio:', obs_list)
    print ('robot error:', handover_pos_error)
    print ('human error:', human_predict_error)

    visualize_result(obs_list,handover_pos_error,'blue')
    visualize_result(obs_list,human_predict_error,'green')
    plt.legend(["robot error","human error"])
    plt.show()

def prepross_traj(test_traj_raw):
    global IPROMPS_SET
    test_traj = np.hstack([test_traj_raw['left_hand'], test_traj_raw['left_joints']])
    num_obs_joints = test_traj_raw['left_hand'].shape[1]
    timestamp = test_traj_raw['stamp']

    # filter the data
    test_traj = gaussian_filter1d(test_traj.T, sigma=sigma).T
    # preprocessing for the data
    test_traj = IPROMPS_SET[0].min_max_scaler.transform(test_traj)
    # consider the unobserved info

    test_traj[:, num_obs_joints:] = 0.0

    return test_traj,timestamp

def gen_offline_obs(test_traj,timestamp,obs_ratio1, obs_ratio2):
    # choose the data
    step = 2
    num_obs_s = int(len(timestamp) * obs_ratio1)
    num_obs = int(len(timestamp) * obs_ratio2)
    num_obs -= num_obs % step
    obs_data_post_arr = test_traj[num_obs_s:num_obs:step, :]
    obs_timestamp = timestamp[num_obs_s:num_obs:step]

    return obs_data_post_arr,obs_timestamp


def predict_result(obs_data,timestamp,h_dim,r_dim):
    global IPROMPS_SET
    # phase estimation
    # print('Phase estimating...')
    alpha_max_list = []
    for ipromp in IPROMPS_SET:
        alpha_temp = ipromp.alpha_candidate(num_alpha_candidate)
        idx_max = ipromp.estimate_alpha(alpha_temp, obs_data, timestamp)
        alpha_max_list.append(alpha_temp[idx_max]['candidate'])
        ipromp.set_alpha(alpha_temp[idx_max]['candidate'])

    # task recognition
    # print('Adding via points in each trained model...')
    for task_idx, ipromp in enumerate(IPROMPS_SET):
        for idx in range(len(timestamp)):
            ipromp.add_viapoint(timestamp[idx] / alpha_max_list[task_idx], obs_data[idx, :])
        ipromp.param_update(unit_update=True)
    # print('Computing the likelihood for each model under observations...')

    prob_task = []
    for ipromp in IPROMPS_SET:
        prob_task_temp = ipromp.prob_obs()
        prob_task.append(prob_task_temp)
    idx_max_prob = np.argmax(prob_task)
    # idx_max_prob = 0 # a trick for testing
    print('The max fit model index is task %s' % task_name[idx_max_prob])

    pred_class = idx_max_prob

    # robot motion generation
    traj_full = []
    for ipromp_id, ipromp in enumerate(IPROMPS_SET):
        [traj_time, traj] = ipromp.gen_real_traj(alpha_max_list[ipromp_id])
        traj = ipromp.min_max_scaler.inverse_transform(traj)
        human_traj = traj[:, 0:h_dim]
        robot_traj = traj[:, h_dim:h_dim + r_dim]

        traj_full.append([human_traj, robot_traj])

    return pred_class, traj_full

def Cartesian_dist(pos1,pos2):
    dist = math.sqrt(math.pow((pos1[0] - pos2[0]), 2) \
                         + math.pow((pos1[1] - pos2[1]), 2) \
                         + math.pow((pos1[2] - pos2[2]), 2))
    return dist

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


def visualize_result(obs,error,color):
    # print result
    plt.grid(True)
    fig = plt.figure(0)
    ax = fig.gca()
    ax.plot(obs,error,
            '-', linewidth=5, color=color, label='error along trajectory', alpha=1)
    ax.set_title('traj error vs observation ratio')
    ax.set_xlabel('ratio observed')
    ax.set_ylabel('error distance in meter')

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