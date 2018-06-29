#!/usr/bin/python
import load_data
import train_models
import noise_cov_cal
import sys


def main():

    print('hand:', left_hand_index, 'robot:', left_joints_index)
    print ('generate noise')
    noise_cov_cal.main(left_hand_index,left_joints_index)
    print('## Running the %s' % train_models.__name__)
    train_models.main()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        lh = [207,208,209,197,198,199]
        lj = [317,318,319]
    elif len(sys.argv)==2:
        lh=sys.argv[1]
        lh = lh.split(',')
        lh = map(int,lh)
        lj = [317, 318, 319]
    elif len(sys.argv)>=3:
        lh=sys.argv[1]
        lj=sys.argv[2]

    print ('hand:',lh,'robot:',lj)
    global left_hand_index
    global left_joints_index

    left_hand_index = lh
    left_joints_index = lj

    main()

