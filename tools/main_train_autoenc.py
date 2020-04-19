#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 07:45:40 2020

@author: arpan

@Description: main file for training Auto-Encoder on strokes
"""

import _init_paths

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from utils.resnet_feature_extracter import Img2Vec
import numpy as np
import os
import time
import copy
import matplotlib.pyplot as plt
from PIL import Image
import json
import cv2
from utils import autoenc_utils
import datasets.videotransforms as videotransforms
from models import autoenc
from datasets.dataset import CricketStrokesDataset


# Set the path of the input videos and the openpose extracted features directory
# Server Paths
CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"
DATASET = "/opt/datasets/cricket/ICC_WT20"
POSE_FEATS = "/home/arpan/cricket/output_json"
BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
#TRAIN_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/train"
#VAL_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/val"
#TEST_FRAMES = "/home/arpan/VisionWorkspace/Cricket/batsman_detection/ICC_WT20_frames/test"
#ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"

# Local Paths
if not os.path.exists(DATASET):
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    POSE_FEATS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/data/output_json"
    BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"


if __name__ == '__main__':
    
    ###########################################################################
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
    # get list of label filenames containing temporal stroke and batsman labels
    train_lab = [f.rsplit('.',1)[0] +".json" for f in train_lst]
    val_lab = [f.rsplit('.',1)[0] +".json" for f in val_lst]
    test_lab = [f.rsplit('.',1)[0] +".json" for f in test_lst]
    train_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in train_lst]
    val_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in val_lst]
    test_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in test_lst]
    
    #####################################################################
    
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    tr_gt_batsman = [os.path.join(BAT_LABELS, f) for f in train_gt_batsman]
    sizes = [autoenc_utils.getTotalFramesVid(os.path.join(DATASET, f)) for f in train_lst]
    print("Size : {}".format(sizes))
    
    bboxes = True
    epsilon = 50
    visualize = False
    n_clusters = 5
    clust_no = 1
    first_frames = 2
    
    ###########################################################################
    # Create a Dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(112),
                                           #videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(112)])
    
    # Prepare datasets and dataset loaders (training and validation/testing)
    train_dataset = CricketStrokesDataset(train_lst, DATASET, LABELS, CLASS_IDS, frames_per_clip=16, \
                                   train=True, transform=train_transforms)
    
    x, y = train_dataset.__getitem__(0)
    print(x.shape)
    print(y)
    
#    for i, (x, y) in enumerate(data_loader):
##                x = vid_seq[0]
##                y = vid_seq[2]
##                # x is batch x depth x H x W x ch, y is class_idx
##                # print("x type", type(x.data))
#                # transpose to batch x ch x depth x H x W
#        x = x.permute(0, 4, 1, 2, 3).float()
    ###########################################################################
    # Create a model
    
    
    
    ###########################################################################
    # train
    
    
    
    ###########################################################################
    # Validate / Evaluate
    
    
    
