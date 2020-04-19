#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 03:04:34 2020

@author: arpan
"""

import os
import numpy as np
import sys
import json
import cv2


def extract_all_feats(vidsPath, labelsPath, nbins, mag_thresh=5):
    """
    Function to iterate on all the training videos and extract the relevant features.
    vidsPath: str
        path to the dataset containing the videos
    labelsPath: str
        path to the JSON files for the labels.
    mag_thresh: float
        pixels with >mag_thresh will be considered significant and used for clustering
    nbins: int
        No. of bins in which the angles have to be divided.
    """
    video_files = sorted(os.listdir(vidsPath))
    json_files = sorted(os.listdir(labelsPath))
    all_feats = None
    bins = np.linspace(0, 2*np.pi, (nbins+1))
    for i, v_file in enumerate(video_files):
        print('-'*60)
        print(str(i+1)+". v_file :: ", v_file)
        video_name = v_file.rsplit('.', 1)[0]
        #print("video_name :: ", video_name)
        json_file = video_name + '.json'
        #print("json file :: ", json_file)
        
        # read labels from JSON file
        if json_file not in json_files:
            print("json file not found!")
            sys.exit(0)
        with open(os.path.join(labelsPath, json_file), 'r') as fr:
            frame_dict = json.load(fr)
        frame_indx = list(frame_dict.values())[0]
        #print('frame_indx :: ', frame_indx)

        #f_loc = os.path.join(flow_numpy_path, video_name)
        stroke_features = extract_flow_angles(os.path.join(vidsPath, v_file), frame_indx, bins, mag_thresh)
        if all_feats is None:
            all_feats = stroke_features
        else:
            all_feats = np.vstack((all_feats, stroke_features))
        #break
    return all_feats

def extract_flow_angles(vidFile, frame_indx, hist_bins, mag_thresh):
    '''
    Extract optical flow maps from video vidFile for all the frames and put the angles with >mag_threshold in different 
    bins. The bins vector is the feature representation for the stroke. 
    Use only the strokes given by list of tuples frame_indx.
    Parameters:
    ------
    vidFile: str
        complete path to a video
    frame_indx: list of tuples (start_frameNo, end_frameNo)
        each tuple in the list denotes the starting frame and ending frame of a stroke.
    hist_bins: 1d np array 
        bin divisions (boundary values). Used np.linspace(0, 2*PI, 11) for 10 bins
    mag_thresh: int
        minimum size of the magnitude vectors that are considered (no. of pixels shifted in consecutive frames of OF)
    
    '''
    
    cap = cv2.VideoCapture(vidFile)
    if not cap.isOpened():
        print("Capture object not opened. Aborting !!")
        sys.exit(0)
    ret = True
    stroke_features = []
    prvs, next_ = None, None
    for m, n in frame_indx:   #localization tuples
        
        #print("stroke {} ".format((m, n)))
        sum_norm_mag_ang = np.zeros((len(hist_bins)-1))  # for optical flow maxFrames - 1 size
        frameNo = m
        while ret and frameNo <= n:
            if (frameNo-m) == 0:    # first frame condition
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameNo)
                ret, frame1 = cap.read()
                if not ret:
                    print("Frame not read. Aborting !!")
                    break
                # resize and then convert to grayscale
                #cv2.imwrite(os.path.join(flow_numpy_path, str(frameNo)+".png"), frame1)
                prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                #prvs = scale_and_crop(prvs, scale)
                frameNo +=1
                continue
                
            ret, frame2 = cap.read()
            #cv2.imwrite(os.path.join(flow_numpy_path, str(frameNo)+".png"), frame2)
            # resize and then convert to grayscale
            next_ = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            
            #print("Mag > 5 = {}".format(np.sum(mag>THRESH)))
            pixAboveThresh = np.sum(mag>mag_thresh)
            #use weights=mag[mag>THRESH] to be weighted with magnitudes
            #returns a tuple of (histogram, bin_boundaries)
            ang_hist = np.histogram(ang[mag>mag_thresh], bins=hist_bins)
            sum_norm_mag_ang +=ang_hist[0]
#            if not pixAboveThresh==0:
#                sum_norm_mag_ang[frameNo-m-1] = np.sum(mag[mag > THRESH])/pixAboveThresh
#                sum_norm_mag_ang[(maxFrames-1)+frameNo-m-1] = np.sum(ang[mag > THRESH])/pixAboveThresh
            frameNo+=1
            prvs = next_
        stroke_features.append(sum_norm_mag_ang/(n-m+1))
    cap.release()
    #cv2.destroyAllWindows()
    stroke_features = np.array(stroke_features)
    return stroke_features

    