#!/usr/bin/python
# Filename: imu_emg_pose_test.py

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import iprompslib_imu_emg_pose
import scipy.linalg
# from scipy.stats import entropy
# import rospy
import math
from sklearn.externals import joblib

plt.close('all')    # close all windows
len_normal = 101    # the len of normalized traj, don't change it
nrDemo = 20         # number of trajectoreis for training
obs_ratio = 20

# plot options
b_plot_raw_dateset = False
b_plot_prior_distribution = False
b_plot_update_distribution = False


#################################
# load raw date sets
#################################
dataset_aluminum_hold = joblib.load('./pkl/dataset_aluminum_hold.pkl')
dataset_spanner_handover = joblib.load('./pkl/dataset_spanner_handover.pkl')
dataset_tape_hold = joblib.load('./pkl/dataset_tape_hold.pkl')

#################################
# load norm date sets
#################################
dataset_aluminum_hold_norm = joblib.load('./pkl/dataset_aluminum_hold_norm.pkl')
dataset_spanner_handover_norm = joblib.load('./pkl/dataset_spanner_handover_norm.pkl')
dataset_tape_hold_norm = joblib.load('./pkl/dataset_tape_hold_norm.pkl')


#################################
# Interaction ProMPs train
#################################
# create a 3 tasks iProMP
ipromp_aluminum_hold = iprompslib_imu_emg_pose.IProMP(num_joints=19, nrBasis=11, sigma=0.05, num_samples=101)
ipromp_spanner_handover = iprompslib_imu_emg_pose.IProMP(num_joints=19, nrBasis=11, sigma=0.05, num_samples=101)
ipromp_tape_hold = iprompslib_imu_emg_pose.IProMP(num_joints=19, nrBasis=11, sigma=0.05, num_samples=101)

# add demostration
for idx in range(nrDemo):
    # train aluminum_hold
    demo_temp = np.hstack([dataset_aluminum_hold_norm[idx]["imu"], dataset_aluminum_hold_norm[idx]["emg"]])
    demo_temp = np.hstack([demo_temp, dataset_aluminum_hold_norm[idx]["pose"]])
    ipromp_aluminum_hold.add_demonstration(demo_temp)
    # train spanner_handover
    demo_temp = np.hstack([dataset_spanner_handover_norm[idx]["imu"], dataset_spanner_handover_norm[idx]["emg"]])
    demo_temp = np.hstack([demo_temp, dataset_spanner_handover_norm[idx]["pose"]])
    ipromp_spanner_handover.add_demonstration(demo_temp)
    # tain tape_hold
    demo_temp = np.hstack([dataset_tape_hold_norm[idx]["imu"], dataset_tape_hold_norm[idx]["emg"]])
    demo_temp = np.hstack([demo_temp, dataset_tape_hold_norm[idx]["pose"]])
    ipromp_tape_hold.add_demonstration(demo_temp)

# model the phase distribution
for i in range(nrDemo):
    alpha = (len(dataset_aluminum_hold[i]["imu"]) - 1) / 50.0 / 1.0
    ipromp_aluminum_hold.add_alpha(alpha)


################################
# Interaction ProMPs test
################################
# constract the testset
# aluminum hold
test_set_temp = np.hstack((dataset_aluminum_hold_norm[20]["imu"], dataset_aluminum_hold_norm[20]["emg"]))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
robot_response = dataset_aluminum_hold_norm[20]["pose"];
# spanner handover
test_set_temp = np.hstack((dataset_spanner_handover_norm[20]["imu"], dataset_spanner_handover_norm[20]["emg"]))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
robot_response = dataset_spanner_handover_norm[20]["pose"]
# tape hold
test_set_temp = np.hstack((dataset_tape_hold_norm[20]["imu"], dataset_tape_hold_norm[20]["emg"]))
test_set = np.hstack((test_set_temp, np.zeros([len_normal, 7])))
robot_response = dataset_tape_hold_norm[20]["pose"]

# the measurement noise
imu_meansurement_noise_cov = np.eye((4))*10000
emg_meansurement_noise_cov = np.eye((8))*250
pose_meansurement_noise_cov = np.eye((7))*0.01
meansurement_noise_cov_full = scipy.linalg.block_diag(imu_meansurement_noise_cov, emg_meansurement_noise_cov, pose_meansurement_noise_cov)

# add via/obsys points to update the distribution
for idx in range(obs_ratio):
    ipromp_aluminum_hold.add_viapoint(0.01*idx, test_set[idx, :], meansurement_noise_cov_full)
    ipromp_spanner_handover.add_viapoint(0.01*idx, test_set[idx, :], meansurement_noise_cov_full)
    ipromp_tape_hold.add_viapoint(0.01*idx, test_set[idx, :], meansurement_noise_cov_full)

# the model info
print('the number of demonstration is ',nrDemo)
print('the number of observation is ', obs_ratio/100.0)

# likelihood of observation
prob_aluminum_hold = ipromp_aluminum_hold.prob_obs()
print('from obs, the log pro of aluminum_hold is', prob_aluminum_hold)
##
prob_spanner_handover = ipromp_spanner_handover.prob_obs()
print('from obs, the log pro of spanner_handover is', prob_spanner_handover)
##
prob_tape_hold = ipromp_tape_hold.prob_obs()
print('from obs, the log pro of tape_hold is', prob_tape_hold)

idx_max_pro = np.argmax([prob_aluminum_hold, prob_spanner_handover, prob_tape_hold])
if idx_max_pro == 0:
    print('the obs comes from aluminum_hold')
elif idx_max_pro == 1:
    print('the obs comes from spanner_handover')
elif idx_max_pro == 2:
    print('the obs comes from tape_hold')


#############################
# compute the position error
#############################
position_error = None
# if idx_max_pro == 0:
predict_robot_response = ipromp_aluminum_hold.generate_trajectory()
position_error = np.linalg.norm(predict_robot_response[-1,12:15]-robot_response[-1,0:3])
print('if aluminum_hold, the obs position error is', position_error)
# elif idx_max_pro == 1:
predict_robot_response = ipromp_spanner_handover.generate_trajectory()
position_error = np.linalg.norm(predict_robot_response[-1, 12:15] - robot_response[-1,0:3])
print('if spanner_handover, the obs position error is', position_error)
# elif idx_max_pro == 2:
predict_robot_response = ipromp_tape_hold.generate_trajectory()
position_error = np.linalg.norm(predict_robot_response[-1, 12:15] - robot_response[-1,0:3])
print('if tape_hold, the obs position error is', position_error)


# #############################
# # the KL divergence of IMU
# #############################
# mean_a_imu = ipromp_aluminum_hold.mean_W_full[0:44]
# cov_a_imu = ipromp_aluminum_hold.cov_W_full[0:44,0:44]
# mean_s_imu = ipromp_spanner_handover.mean_W_full[0:44]
# cov_s_imu = ipromp_spanner_handover.cov_W_full[0:44,0:44]
# kl_divergence_imu_a_s = math.log(np.linalg.det(cov_s_imu)/np.linalg.det(cov_a_imu)) - 44 \
#                         + np.trace(np.dot(np.linalg.inv(cov_s_imu), cov_a_imu)) + \
#                         np.dot((mean_s_imu-mean_a_imu).T, np.dot(np.linalg.inv(cov_s_imu), (mean_s_imu-mean_a_imu)))
#
# mean_a_imu_emg = ipromp_aluminum_hold.mean_W_full[0:132]
# cov_a_imu_emg = ipromp_aluminum_hold.cov_W_full[0:132,0:132]
# mean_s_imu_emg = ipromp_spanner_handover.mean_W_full[0:132]
# cov_s_imu_emg = ipromp_spanner_handover.cov_W_full[0:132,0:132]
# kl_divergence_imu_emg_a_s = math.log(np.linalg.det(cov_s_imu_emg)/np.linalg.det(cov_a_imu_emg)) - 132\
#                         + np.trace(np.dot(np.linalg.inv(cov_s_imu_emg), cov_a_imu_emg)) + \
#                         np.dot((mean_s_imu_emg-mean_a_imu_emg).T, np.dot(np.linalg.inv(cov_s_imu_emg), (mean_s_imu_emg - mean_a_imu_emg)))


#############################
# plot raw data
#############################
if b_plot_raw_dateset == True:
    ## plot the aluminum hold task raw data
    plt.figure(0)
    for ch_ex in range(4):
       plt.subplot(411+ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_aluminum_hold[idx]["imu"][:, ch_ex])), dataset_aluminum_hold[idx]["imu"][:, ch_ex])
    plt.figure(1)
    for ch_ex in range(8):
       plt.subplot(421+ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_aluminum_hold[idx]["emg"][:, ch_ex])), dataset_aluminum_hold[idx]["emg"][:, ch_ex])
    plt.figure(2)
    for ch_ex in range(7):
       plt.subplot(711+ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_aluminum_hold[idx]["pose"][:, ch_ex])), dataset_aluminum_hold[idx]["pose"][:, ch_ex])
    ## plot the spanner handover task raw data
    plt.figure(10)
    for ch_ex in range(4):
       plt.subplot(411 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_spanner_handover[idx]["imu"][:, ch_ex])), dataset_spanner_handover[idx]["imu"][:, ch_ex])
    plt.figure(11)
    for ch_ex in range(8):
       plt.subplot(421 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_spanner_handover[idx]["emg"][:, ch_ex])), dataset_spanner_handover[idx]["emg"][:, ch_ex])
    plt.figure(12)
    for ch_ex in range(7):
       plt.subplot(711 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_spanner_handover[idx]["pose"][:, ch_ex])), dataset_spanner_handover[idx]["pose"][:, ch_ex])
    ## plot the tape hold task raw data
    plt.figure(20)
    for ch_ex in range(4):
       plt.subplot(411 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_tape_hold[idx]["imu"][:, ch_ex])), dataset_tape_hold[idx]["imu"][:, ch_ex])
    plt.figure(21)
    for ch_ex in range(8):
       plt.subplot(421 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_tape_hold[idx]["emg"][:, ch_ex])), dataset_tape_hold[idx]["emg"][:, ch_ex])
    plt.figure(22)
    for ch_ex in range(7):
       plt.subplot(711 + ch_ex)
       for idx in range(nrDemo):
           plt.plot(range(len(dataset_tape_hold[idx]["pose"][:, ch_ex])), dataset_tape_hold[idx]["pose"][:, ch_ex])


#############################
# plot the prior distributioin
#############################
if b_plot_prior_distribution == True:
    # plot ipromp_aluminum_hold
    plt.figure(50)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_aluminum_hold.promps[i].plot(ipromp_aluminum_hold.x, color='g', legend='alumnium hold model, imu');plt.legend()
    plt.figure(51)
    for i in range(8):
        plt.subplot(421+i)
        ipromp_aluminum_hold.promps[4+i].plot(ipromp_aluminum_hold.x, color='g', legend='alumnium hold model, emg');plt.legend()
    plt.figure(52)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_aluminum_hold.promps[4+8+i].plot(ipromp_aluminum_hold.x, color='g', legend='alumnium hold model, pose');plt.legend()
    # plot ipromp_spanner_handover
    plt.figure(60)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_spanner_handover.promps[i].plot(ipromp_spanner_handover.x, color='g', legend='spanner handover model, imu');plt.legend()
    plt.figure(61)
    for i in range(8):
        plt.subplot(421+i)
        ipromp_spanner_handover.promps[4+i].plot(ipromp_spanner_handover.x, color='g', legend='spanner handover model, emg');plt.legend()
    plt.figure(62)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_spanner_handover.promps[4+8+i].plot(ipromp_spanner_handover.x, color='g', legend='spanner handover model, pose');plt.legend()
    # plot ipromp_tape_hold
    plt.figure(70)
    for i in range(4):
        plt.subplot(411+i)
        ipromp_tape_hold.promps[i].plot(ipromp_tape_hold.x, color='g', legend='tape hold model, imu');plt.legend()
    plt.figure(71)
    for i in range(8):
        plt.subplot(421+i)
        ipromp_tape_hold.promps[4+i].plot(ipromp_tape_hold.x, color='g', legend='tape hold model, emg');plt.legend()
    plt.figure(72)
    for i in range(7):
        plt.subplot(711+i)
        ipromp_tape_hold.promps[4+8+i].plot(ipromp_tape_hold.x, color='g', legend='tape hold model, pose');plt.legend()


#############################
# plot the updated distributioin
#############################
if b_plot_update_distribution == True:
    # plot ipromp_aluminum_hold
    plt.figure(50)
    for i in range(4):
        plt.subplot(411+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[i].plot_updated(np.arange(0,1.01,0.01), color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(51)
    for i in range(8):
        plt.subplot(421+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[4+i].plot_updated(np.arange(0,1.01,0.01), color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(52)
    for i in range(7):
        plt.subplot(711+i)
        plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend();
        ipromp_aluminum_hold.promps[4+8+i].plot_updated(np.arange(0,1.01,0.01), color='b', legend='updated distribution', via_show=False); plt.legend();
    # plot ipromp_spanner_handover
    plt.figure(60)
    for i in range(4):
        plt.subplot(411+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[i].plot_updated(np.arange(0,1.01,0.01), color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(61)
    for i in range(8):
        plt.subplot(421+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[4+i].plot_updated(np.arange(0,1.01,0.01), color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(62)
    for i in range(7):
        plt.subplot(711+i)
        plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_spanner_handover.promps[4+8+i].plot_updated(np.arange(0,1.01,0.01), color='b', legend='updated distribution', via_show=False); plt.legend();
    # plot ipromp_tape_hold
    plt.figure(70)
    for i in range(4):
        plt.subplot(411+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[i].plot_updated(np.arange(0,1.01,0.01), color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(71)
    for i in range(8):
        plt.subplot(421+i)
        plt.plot(ipromp_aluminum_hold.x, test_set[:, 4+i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[4+i].plot_updated(np.arange(0,1.01,0.01), color='b', legend='updated distribution', via_show=True); plt.legend();
    plt.figure(72)
    for i in range(7):
        plt.subplot(711+i)
        plt.plot(ipromp_aluminum_hold.x, robot_response[:, i], color='r', linewidth=3, label='ground truth'); plt.legend()
        ipromp_tape_hold.promps[4+8+i].plot_updated(np.arange(0,1.01,0.01), color='b', legend='updated distribution', via_show=False); plt.legend();

plt.show()