#!/usr/bin/python
import numpy as np
from hmmlearn import hmm
import os
import ConfigParser
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

# read the current file path
file_path = os.path.dirname(__file__)
# read model cfg file
cp_models = ConfigParser.SafeConfigParser()
cp_models.read(os.path.join(file_path, '../cfg/models.cfg'))
datasets_path = os.path.join(file_path, cp_models.get('datasets', 'path'))

# load dataset
datasets_raw = joblib.load(os.path.join(datasets_path, 'pkl/datasets_raw.pkl'))

# task name
task_name = joblib.load(os.path.join(datasets_path, 'pkl/task_name_list.pkl'))

# joint number
num_joints = cp_models.getint('datasets', 'num_joints')


def main():
    states = task_name
    n_task = len(states)

    # don't know the task prior here
    start_probability = np.array(([1.0/n_task]*n_task))

    # don't know the task transition probability here
    transition_probability = np.array([1.0/n_task]*n_task*n_task).reshape([n_task, n_task])

    gmm_model_full = [GaussianMixture(n_components = 3) for i in range(n_task)]

    task_data_full = []
    for task_idx, task_data in enumerate(datasets_raw):
        demo_data_full = np.array([]).reshape(0, num_joints)
        for demo_data in task_data:
            h = np.hstack([demo_data['left_hand'], demo_data['left_joints']])
            demo_data_full = np.vstack([demo_data_full, h])
        task_data_full.append(demo_data_full)

    for task_data, gmm_model in zip(task_data_full, gmm_model_full):
        gmm_model.fit(task_data)

    # hmm_model = hmm.GMMHMM()
    # hmm_model.n_components = n_task
    # hmm_model.startprob_ = start_probability
    # hmm_model.transmat_ = transition_probability
    # hmm_model.means_ = gmm_model_full[0].means_

    # testing
    hmm_model1 = hmm.GMMHMM()
    hmm_model1.n_components = 10
    hmm_model1.fit(task_data_full[0])
    print 1

if __name__ == '__main__':
    main()
