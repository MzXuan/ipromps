#!/usr/bin/python
import os
import ConfigParser
from sklearn.externals import joblib

# the current file path
FILE_PATH = os.path.dirname(__file__)
SELECT_CLASS = []
DATASETS_PATH = ''


def get_feature_index(csv_path='../cfg/models.cfg'):
    global SELECT_CLASS
    # read models params
    cp_models = ConfigParser.SafeConfigParser()
    cp_models.read(os.path.join(FILE_PATH, csv_path))
    SELECT_CLASS = cp_models.get('datasets', 'class_index')
    SELECT_CLASS = map(int, SELECT_CLASS.split(','))
    emg = cp_models.get("csv_parse",'emg')
    imu = cp_models.get("csv_parse",'imu')
    elbow_position = cp_models.get("csv_parse",'elbow_position')
    
    h_feature_index = [0,1,2] #wrist position
    r_feature_index = [0,1,2] #end effector position
    if elbow_position == 'enable':
        print('enable elbow position')
        h_feature_index +=[3,4,5]
    if emg == 'enable':
        print('enable emg data')
        h_feature_index +=[6,7,8,9,10,11,12,13]
    if imu == 'enable':
        print('enable imu data')
        h_feature_index +=[14,15,16,17]
    
    h_dim = len(h_feature_index)
    r_dim = len(r_feature_index)
    print('human feature dim:',h_dim,'robot feature dim:',r_dim)
    
    return h_feature_index,r_feature_index,h_dim,r_dim
    

##seperate to train, val, test1
def split(data, train_part=0.7, val_part=0.8, test_part=1.0):
    train_data = []
    val_data = []
    test_data = []

    for trajs in data:
        num = len(trajs)
        train_pos = int(num * train_part)
        val_pos = int(num * val_part)

        train_data.append(trajs[:train_pos])
        val_data.append(trajs[train_pos:val_pos])
        test_data.append(trajs[val_pos:])

    return train_data,val_data,test_data

def main():
    # read models cfg file
    cp_models = ConfigParser.SafeConfigParser()
    cp_models.read(os.path.join(FILE_PATH, '../cfg/models.cfg'))
    DATASETS_PATH = os.path.join(FILE_PATH, cp_models.get('datasets', 'path'))
    # read norm data
    datasets_norm_path = os.path.join(DATASETS_PATH, 'pkl/datasets_norm.pkl')
    datasets_norm = joblib.load(datasets_norm_path)

    h_feature_index, r_feature_index, h_dim, r_dim = get_feature_index()

    ## generate new dataset according to the seletive feature
    data4train = []
    for i in SELECT_CLASS:
        traj_temp = []
        for traj in datasets_norm[i]:
            traj_temp.append({
                'alpha': traj['alpha'],
                'left_hand': traj['left_hand'][:, h_feature_index],
                'left_joints': traj['left_joints'][:, r_feature_index]
            })
        data4train.append(traj_temp)
    
    #split data to different dataset
    train_data, val_data, test_data = split(data4train)

    ##save for future use
    joblib.dump(train_data, os.path.join(DATASETS_PATH, 'pkl/train_data.pkl'))
    joblib.dump(val_data, os.path.join(DATASETS_PATH, 'pkl/val_data.pkl'))
    joblib.dump(test_data, os.path.join(DATASETS_PATH, 'pkl/test_data.pkl'))


if __name__ == '__main__':
    main()