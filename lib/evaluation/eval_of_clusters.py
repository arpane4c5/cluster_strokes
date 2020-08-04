#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 05:55:45 2020

@author: arpan

@Description: Evaluate the optical flow clusters.
"""

import os
import numpy as np
import sys
import json
import cv2
import csv
from itertools import permutations

# save video according to their label
def evaluate(labels, stroke_names, results_dir, data_path):
    '''Receive a list of cluster assignments and write stroke videos in their 
    respective cluster directories.
    
    Parameters:
    -----------
    labels : np.ndarray
        1D array of cluster labels (0, 1, 2, ..)
    stroke_names : list of str
        list of stroke names (each element of type vidname_stfrm_endfrm)
    results_dir : str
        destination directory prefix name
    data_path : str
        path containing the video dataset
    
    '''
    
    n_clusters = max(labels) + 1
    print("clusters, ", n_clusters)
    for i in range(n_clusters):
        ######
        try:
            os.makedirs(os.path.join(results_dir, str(i)))
        except Exception as e:
            print("Exception : ", e)
        #######
        for count,j in enumerate(np.where(labels==i)[0]):
            vid_data = stroke_names[j].rsplit('_', 2)
            m, n = map(int, vid_data[-2:])
            vid_name = vid_data[0]            
            #f = ''.join(vid_name.split(' ')[2:-1])+"_"+str(m)+"_"+str(n)
            if ".avi" in vid_name or ".mp4" in vid_name:
                if '/' in vid_name:     # full path given
                    src_vid = vid_name
                    vid_name = vid_name.rsplit('/', 1)[-1]
                else:
                    src_vid = os.path.join(data_path, vid_name)
                dest_vid = vid_name.rsplit('.', 1)[0]+"_"+str(m)+"_"+str(n)
            else:
                src_vid = os.path.join(data_path, vid_name+".avi")
                dest_vid = vid_name+"_"+str(m)+"_"+str(n)
            save_video(src_vid, dest_vid, m, n, i, results_dir)
            if count==9:
                break

            
def get_frame(cap, frame_no):
    '''Read a frame of given position from the VideoCapture Object provided.
    Parameters:
    -----------
    cap : cv2.VideoCapture
        VideoCapture object reference
    frame_no : int
        position from where the frame is to be read.
        
    Returns:
    --------
    np.ndarray of shape (H, W, 3)
    
    '''
    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # check for valid frame number
    if frame_no >= 0 & frame_no <= totalFrames:
        # set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
        _, img = cap.read()
        return img
    print("invalid frame, ", frame_no)
    sys.exit()

def save_video(src_vid, dest_vid, m, n, label, results_dir):
    eval_path = os.path.join(results_dir, str(label))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_out_path = os.path.join(eval_path, dest_vid+'.avi')
    out = cv2.VideoWriter(vid_out_path, fourcc, 25.0, (320, 180), True)
    cap = cv2.VideoCapture(src_vid)
    if cap.isOpened():
        pass
    else:
        print("closed")
        sys.exit()
    for i in range(m, n+1):
        img = cv2.resize(get_frame(cap, i), (320, 180), interpolation=cv2.INTER_AREA)
        out.write(img)
    cap.release()
    out.release()


def get_ordered_strokes_list(data_path, labels_path):
    flows_path = []
    
    video_files = sorted(os.listdir(data_path))
    json_files = sorted(os.listdir(labels_path))
    for i, v_file in enumerate(video_files):
        #print('-'*60)
        #print(str(i+1)+". v_file :: ", v_file)
        video_name = v_file.rsplit('.', 1)[0]
        json_file = video_name + '.json'
        # read labels from JSON file
        if json_file not in json_files:
            print("json file not found!")
            sys.exit(0)
        with open(os.path.join(labels_path, json_file), 'r') as fr:
            frame_dict = json.load(fr)
        frame_indx = list(frame_dict.values())[0]
        for m, n in frame_indx:
            flows_path.append('{}_{}_{}'.format(video_name, m, n))
            
    return flows_path


def get_accuracy(all_feats, pred_values, cluster_labels_path, data_path, labels_path):
    # get the ground truth from file
    gt_keys, gt_values = get_cluster_labels(cluster_labels_path)
    
    if '.avi' in gt_keys[0] or '.mp4' in gt_keys[0]:
        gt_keys = [s.replace('.avi', '') for s in gt_keys]
        gt_keys = [s.replace('.mp4', '') for s in gt_keys]
    
    if min(gt_values) == 1:
        gt_values = [v-1 for v in gt_values]
    
    # Read filenames in the same order
    pred_keys = get_ordered_strokes_list(data_path, labels_path)
    print("Length of pred_keys {} :: Length of gt_keys {} :: Length of pred_values {}"\
          .format(len(pred_keys), len(gt_keys), len(pred_values)))
    
    # check all stroke names of predictions are present at corresponding gt list locations
    for i, vidname in enumerate(pred_keys):
        assert vidname == gt_keys[i], "Key {} not matching {}".format(i, vidname)
    
    return calculate_accuracy(gt_values, pred_values)
    
    
def calculate_accuracy(gt_list, pred_list):
    """
    Get two dictionaries with labels. 
    """
    gt_clus_nos = sorted(list(set(gt_list)))
    pred_clus_nos = sorted(list(set(pred_list)))
    n_clusters_gt = len(gt_clus_nos)
    n_clusters_pred = len(pred_clus_nos)
    assert n_clusters_gt==n_clusters_pred, "Unequal no. of clusters {} and {}".format(n_clusters_gt, n_clusters_pred)
    
    for i,x in enumerate(pred_clus_nos):
        assert x==gt_clus_nos[i], "Cluster no. mismatch {} / {}".format(gt_clus_nos[i], x)
        
    acc_values = []
    perm_list = list(permutations(pred_clus_nos))
    for perm_tuple in perm_list:
        
        pred_list_permuted = assign_clusters(perm_tuple, pred_list)
        #print(pred_list[0:11], pred_list_permuted[0:11],perm_tuple)
        acc_tuple = len(perm_tuple)*[0]
        # For each cluster in tuple, find the no. of correct predictions and save at specific tuple indx position
        for t in perm_tuple:
            acc_tuple[t] = sum([(gt_list[j]==pred and gt_list[j]==t) for j,pred in enumerate(pred_list_permuted)])
        
        acc_values.append(acc_tuple)
        
    return acc_values, perm_list, gt_list, pred_list
    
    
def assign_clusters(perm_tuple, labs):
    """
    Take input as permutation tuple and interchange labels in list labs
    Eg. if perm_tuple = (1,2,0) and labs = [0,0,0,1,1,1,2,2,2], then 
    return [1, 1, 1, 2, 2, 2, 0, 0, 0]
    """
    temp = len(labs)*[-1]
    for i,clust in enumerate(labs):
        for j,t in enumerate(perm_tuple):
            if clust==j:
                temp[i]=t
    return temp
    
    
    
def get_cluster_labels(cluster_labels_path):
    labs_keys = []
    labs_values = []
    with open(cluster_labels_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:    
            #print("{} :: Class : {}".format(row[0], row[1]))
            labs_keys.append(row[0])
            labs_values.append(int(row[1]))
            line_count += 1
        print("Read {} ground truth stroke labels from file.".format(line_count))
    return labs_keys, labs_values
