#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 07:51:38 2020

@author: arpan

@Description: Utils functions for training AutoEncoder
"""

import os
import cv2

# function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split_dataset_files(datasetPath):
    '''Split the dataset files into training, validation and test sets.
    Only for the highlights dataset. It is assumed that all the video files 
    are at the same path. 
    Parameters:
    ----------
    datasetPath : str
        complete path to the video dataset. The folder contains 26 video files
        
    Returns:
    -------
    filenames : list of str
        3 lists with filenames. 16 for training, 5 for validation and 5 for testing
    '''
    filenames = sorted(os.listdir(datasetPath))         # read the filename
#    filenames = [t.split('.')[0] for t in filenames]   # remove the extension
    return filenames[:16], filenames[16:21], filenames[21:]


def getTotalFramesVid(srcVideoPath):
    """
    Return the total number of frames in the video
    
    Parameters:
    ------
    srcVideoPath: str
        complete path of the source input video file
        
    Returns:
    ------
    total frames present in the given video file
    """
    cap = cv2.VideoCapture(srcVideoPath)
    # if the videoCapture object is not opened then exit without traceback
    if not cap.isOpened():
        print("Error reading the video file !!")
        return 0

    tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return int(tot_frames)