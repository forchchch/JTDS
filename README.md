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

## N-JTDS baseline
If you want to run with the N-JTDS baseline, it needs the index of each sample, so the dataset will be a little different，which is given in Bird_dataset_naive. To run this baseline, please use the following command line:
```
sh run_naive.sh
```
The configs are the same as before.
## The semi-supervised setting
The semi-supervised setting depends on how you choose the data for the primary task. In our paper, we randomly choose 5 samples for each class and then generate a loss mask(1 for the sampled data, 0 for others) for the main task. You can generate the mask for the samples in the way you need. We do not incorporate this part in our released code, but it is not hard to implement this by little modification to the data_preprocess.py to choose the samples, and then according to the choosed samples to generate mask in the Bird_dataset.py, finally use the mask to weight the loss in the training procedure.

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

