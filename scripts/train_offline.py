#!/usr/bin/python
import train_models
import noise_cov_cal
import seperate_data

def main():

    print ('seperate data to different sets...')
    seperate_data.main()
    _,_,h_dim,r_dim = seperate_data.get_feature_index()

    print('hand:', h_dim, 'robot:', r_dim)

    print ('generating noise...')
    noise_cov_cal.main(h_dim,r_dim)

    print('## Running the %s' % train_models.__name__)
    train_models.main(h_dim,r_dim)


if __name__ == '__main__':
    main()