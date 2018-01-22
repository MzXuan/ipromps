# Interaction ProMP
This package is the implement of Interaction ProMP described [here](http://www.ausy.tu-darmstadt.de/uploads/Team/PubGJMaeda/phase_estim_IJRR.pdf).  
We use EMG signals to enhance the task recognition. We not make sure if the EMG signals are correlated with robot motion and we will confirm it latter. 

# Dependences
- Python >=2.6
- NumPy
- sklearn
- SciPy >= 0.19.1
- pandas

## upgrade scipy 
1. Install the gfortran. Maybe need to install gfortran-5 as dependency.  
2. Upgrade the scipy with "sudo easy_install --upgrade scipy"  
(not sure need to upgrade the numpy with "sudo easy_install --upgrade numpy")  
ref: https://askubuntu.com/questions/682825/how-to-update-to-the-latest-numpy-and-scipy-on-ubuntu-14-04lts  

# The package architecture  
├── cfg  
│   └── params.cfg  
├── datasets  
│   └── dataset_name  
│       ├── info  
│       ├── pkl  
│       └── raw  
├── scripts  
│   ├── bag_to_csv.sh  
│   ├── batch_bag2csv.py  
│   ├── data_visualization.py  
│   ├── ipromps_lib.py  
│   ├── load_data.py  
│   ├── noise_cov_cal.py  
│   ├── test_online.py  
│   ├── train_models.py  
│   └── train_offline.py  
├── README.md  
├── CMakeLists.txt  
└─ package.xml  


## cfg
`params.cfg`: the configuration including all params.

## datasets
The datasets path involves multiple demo dadta and to save the middle data. 


## scripts
The scripts to load data, train models and test it online. 
`bag_to_csv.sh`: a script to convert single rosbag to csv, called in `batch_bag2csv.py`  
`batch_bag2csv.py`: a batch python script to convert rosbag to csv, run it in terminal for some ros shell script  
`ipromps_lib.py`: the lib for IProMP including unit ProMP, ProMPs and Interaction ProMPs
`load_data.py` : load the data from csv file, filter the data and resample the data as same duration  
`train_models.py`: train the models from the loaded data
`train_offline.py`: train the Interaction ProMPs (load data and train model), call for `load_data.py` and `train_models.py`  
`data_visualization.py`: visualization for data  
`noise_cov_cal.py`: theorically measure the observation noise covariance matrix  
`test_online`: test the trained models  


# The command steps to this package
All commands run in **baxter.sh** space.  
1 `roslaunch openni_launch openni.launch`: open the xtion  
2 `roslaunch aruco_hand_eye baxter_xtion_cal_pub.launch`: load the baxter-xtion calibration result  
3 `rosrun rviz rviz`: open the ros visualization windows  
4 `rosrun openni_tracker openni_tracker`: open the human skeleton tracking node  
5 `roslaunch myo_driver myo_raw_pub.launch`: start the Myo armband node  
6 `rosrun states_manager states_pub.py`: open the state manager node  

# The notes for collecting datasets
## notes for datasets collection
Everytime when collecting the datasets, please read these notes. 
1. **Check** the csv file derived from rosbag 
2. **In the same time**, start to record all interest data and demonstrate the collaborative task motion (human and robot)  
3. Demonstrate with **spatio-temporal variance**  
4. No human hand **occlusion and joint** with robot  

## record the training process
Take some significant notes to be used in testing.  
1. Armband wear position on hand  
2. Object grasp position  
3. Collaborative task process video  

## possible options to improve this model
Some possible option to improvement the performance of this model are list as below. 
1. Use robot twist/joint/cartesian space  
2. EMG signals' effect (dose it correlate with robot motion?)  
3. Design the tasks with different motion of robot arm, human hand, elbow, shoulder...  
4. Use double robot arm  
5. No robot orientation change in simple test at the beginning  
