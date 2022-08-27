# Auxiliary Learning with Joint Task and Data Scheduling
by Hong Chen, Xin Wang, Chaoyu Guan, Yue Liu and Wenwu Zhu.

## Introduction
This work can be used when you want to learn a model for the target task with the help of some auxiliary tasks. This work helps you to find the most valuable data samples under each task so that your target task can be learned better. Some of the auxilearn codes are modified from the [AuxLearn Repo](https://github.com/AvivNavon/AuxiLearn), we thank them for offering the code.
![_](./JTDS_figure.jpg)
## Dataset
The provided code is for the CUB dataset in the paper. You can first download the dataset from the [link](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). Then unzip the dataset in the current folder and new a folder named preprocess_data, and split the dataset as follows: 
```
python data_preprocess.py
```
The dataset splitted information will be restored in the preprocess_data folder.

## Run the experiment
When you finish the data preprocessing process, you can run the script to reproduce the CUB experiments.
```
sh run_bilevel.sh
```
Additionally, if you want to run the noisy setting, change the corrupted to 1 and set the corresponding corrupted ratio in the run_bilevel.sh(like the other examples in the command line).
### Explanation for key configs
    + corupted: 0 for clean setting, 1 for corrupted setting
    + corupted ratio: the ratio for corupted labels
    + method: common for learning with all tasks and data together, joint for our selection method

## Scheduler implementation and bilevel optimization
The joint scheduler is implemented with the auxilearn.hypernet.MonoJoint. The upper level optimization process is shown in line 323- line376 in the train_bilevel.py.

## Citation
```
@inproceedings{chen2022auxiliary,
title = {Auxiliary Learning with Joint Task and Data Scheduling},
author = {Chen, Hong and Wang Xin, and Guan, Chaoyu and Liu, Yue and Zhu Wenwu},
booktitle = {International Conference on Machine Learning},
pages = {3634--3647},
year = {2022},
organization = {PMLR}
}
```

