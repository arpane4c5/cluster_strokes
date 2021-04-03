#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:22:22 2020

@author: Ashish (modified by Arpan)

Use this file to view strokes using strokes_labels 
and assign labels to it
1 -> Leg[0 - 90)
2 -> Right Drive[90 - 180)
3 -> Left Drive[180 - 270)
4 -> Cut[270 - 360)
5 -> Ambigous(infield - within 30 yards)
"""

import cv2
import os
import time
import json
import numpy as np

LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
#videos_path = "/home/ashish/Desktop/BTP/flow_cluster/Cricket_Highlights_dataset-sample"
#shots_label_path = "/home/ashish/Desktop/BTP/UntrimmedStrokeLocalization/sample_labels_shots/ICC WT20"
shots_classes_file_path = "batsman_classes.txt"

## take shot name and class
## as list and save to shots_classes_file_path 
def save2_file(name, clss):
    with open(shots_classes_file_path, 'a') as fr:
        fr.write(name+","+clss)
        fr.write("\n")
        
## dict of all videos shots
def get_shots(shots_label_path):
    label_files = os.listdir(shots_label_path)
    video_shots_dict = {}
    for lab_file in label_files:
        video = lab_file.split('.')[0]
        with open(os.path.join(shots_label_path, lab_file), 'r') as fr:
            shots_dict = json.load(fr)
        key = os.path.join("ICC WT20", video+".avi")
        shots_list = shots_dict[key]
        
#        new_shots_list = []
#        for shot in shots_list:
#            total_shot_count+=1
#            f1, f2 = shot
#            new_shots_list.append((f1, f2))
#            #new_shots_list.extend(list(range(f1, f2+1)))
#        # 1d list of shots frame
        video_shots_dict[video+'.avi'] = shots_list
    print("total shot count :", len(shots_list))
    return video_shots_dict


## get shots label dict with 
## key: video name value: shots frame list
video_shots_dict = get_shots(LABELS)
## get list of videos
videos = os.listdir(DATASET)

## get the last video idx and its shot idx(count) to start assigning classes
def get_video_frameidx():
    with open(shots_classes_file_path, 'r+') as fr:
        lines = fr.readlines()
        if(len(lines)==0):
            vid_count = 0
            shot_count = 0
        else:
            count=1
            vid_name = lines[-1].split(',')[0].strip().split('_')[0]
            shot_count = [vid_name for line in lines if line.
                          split(',')[0].strip().split('_')[0]==vid_name ]
            shot_count = len(shot_count)
            vid_count = videos.index(vid_name)
    return vid_count, shot_count

vid_count, shot_count = get_video_frameidx()
total_vid_count = len(videos)


while(vid_count<total_vid_count):
    video_name = videos[vid_count]
    shots_list = video_shots_dict[video_name]
    print("video_name: ", video_name, vid_count)
    if len(shots_list) == shot_count:
        vid_count+=1
        shot_count = 0
        continue
    ## read video
    video_path = os.path.join(DATASET, video_name)
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    frame_count = 0
    
    while ret:
        ret, frame2 = cap.read()
        if shot_count == len(shots_list):
            frame_count+=1
            continue
        shot = shots_list[shot_count]
        if frame_count >= shot[0] and frame_count <= shot[-1]:
            cv2.namedWindow("frame-{}".format(shot_count))
            cv2.imshow("frame-{}".format(shot_count), frame2)
            k = cv2.waitKey(40)
            if frame_count == shot[-1]:
                cv2.destroyAllWindows()
                shot_class = input("enter class: ")
                shot_name = video_name+"_{}_{}".format(shot[0],shot[1])
                save2_file(shot_name , shot_class)
                shot_count +=1
            frame_count+=1
            continue
        frame_count+=1
