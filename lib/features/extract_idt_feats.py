#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:07:04 2020

@author: arpan

@Description: Form the dictionary of IDT features using the extracted *.out files 
of strokes (extraction done using idt_strokes.sh script)
"""

import os
import numpy as np
import pandas as pd
import json
import pickle

def read_pooled_partition_feats(vidsPath, labelsPath, idtFeatsPath, partition_lst, 
                                traj_len=15):
    """
    Function to iterate on all the strokes and corresponding IDT features of strokes
    and form a dictionary for the same.
    vidsPath: str
        path to the dataset containing the videos
    labelsPath: str
        path to the JSON files for the labels.
    idtFeatsPath: str
        path to the *.out files extracted using iDT binary
    partition_lst: list of video_ids
        video_ids are the filenames (without extension)
        
    Returns:
    --------
    dict of pooled stroke features (O(NFrames) x FeatSize) and list of stroke keys 
    """
    strokes_name_id = []
    all_feats = {}
    for i, v_file in enumerate(partition_lst):
        print('-'*60)
        print(str(i+1)+". v_file :: ", v_file)
        if '.avi' in v_file or '.mp4' in v_file:
            v_file = v_file.rsplit('.', 1)[0]
        json_file = v_file + '.json'
        #print("json file :: ", json_file)
        # read labels from JSON file
        assert os.path.exists(os.path.join(labelsPath, json_file)), "{} doesn't exist!".format(json_file)
            
        with open(os.path.join(labelsPath, json_file), 'r') as fr:
            frame_dict = json.load(fr)
        frame_indx = list(frame_dict.values())[0]
        for m,n in frame_indx:
            k = v_file+"_"+str(m)+"_"+str(n)
            if n-m < traj_len:
                print("Skipping : {}".format(k+".out"))
                break
#            print("Stroke {} - {} : {}".format(m,n, n-m))
            strokes_name_id.append(k)
            # Extract the stroke features
            feat_name = k+".out"
            all_feats[k] = read_stroke_feats(os.path.join(idtFeatsPath, feat_name),  traj_len)
            # pool the features for common frame somehow and represent as a global feature
#        break
    return all_feats, strokes_name_id

def read_partition_feats(vidsPath, labelsPath, idtFeatsPath, partition_lst, traj_len=15):
    """
    Function to iterate on all the strokes and corresponding IDT features of strokes
    and form a dictionary for the same.
    vidsPath: str
        path to the dataset containing the videos
    labelsPath: str
        path to the JSON files for the labels.
    idtFeatsPath: str
        path to the *.out files extracted using iDT binary
    partition_lst: list of video_ids
        video_ids are the filenames (without extension)
        
    Returns:
    --------
    numpy array of size (nStrokes x nbins)
    """
    strokes_name_id = []
    all_feats = {}
    for i, v_file in enumerate(partition_lst):
        print('-'*60)
        print(str(i+1)+". v_file :: ", v_file)
        if '.avi' in v_file or '.mp4' in v_file:
            v_file = v_file.rsplit('.', 1)[0]
        json_file = v_file + '.json'
        #print("json file :: ", json_file)
        # read labels from JSON file
        assert os.path.exists(os.path.join(labelsPath, json_file)), "{} doesn't exist!".format(json_file)
            
        with open(os.path.join(labelsPath, json_file), 'r') as fr:
            frame_dict = json.load(fr)
        frame_indx = list(frame_dict.values())[0]
        for m,n in frame_indx:
            k = v_file+"_"+str(m)+"_"+str(n)
            if n-m < traj_len:
                print("Skipping : {}".format(k+".out"))
                break
#            print("Stroke {} - {} : {}".format(m,n, n-m))
            strokes_name_id.append(k)
            # Extract the stroke features
            feat_name = k+".out"
            all_feats[k] = read_stroke_feats(os.path.join(idtFeatsPath, feat_name), m, n, traj_len)
#        break
    return all_feats, strokes_name_id


def read_stroke_feats(featPath, start_fno, end_fno, traj_len):
    """Read the IDT features from the disk, save it in pandas dataframe and return
    the labeled dataframe with last column removed and associated column names.
    Parameters:
    -----------
    featPath : str
        full path to a strokes' feature file (*.out file) extracted using the binary.
    traj_len : int
        the trajectory length used at the time of extraction.
        
    Returns:
    --------
    pd.DataFrame with stroke features and column names.
    """
    assert os.path.isfile(featPath), "File not found {}".format(featPath)
    traj_info = ['frameNum', 'mean_x', 'mean_y', 'var_x', 'var_y', 'length', 'scale', 'x_pos',
                 'y_pos', 't_pos']
    Trajectory = ['Traj_'+str(i) for i in range(2 * traj_len)]
    HOG = ['HOG_'+str(i) for i in range(96)]
    HOF = ['HOF_'+str(i) for i in range(108)]
    MBHx = ['MBHx_'+str(i) for i in range(96)]
    MBHy = ['MBHy_'+str(i) for i in range(96)]
    
    traj_vals = Trajectory + HOG + HOF + MBHx + MBHy
    col_names = traj_info + traj_vals
    
    df = pd.read_csv(featPath, sep='\t', header=None)
    
    # Drop last column having NaN values
    df = df.iloc[:, :-1]
    assert len(col_names) == df.shape[1], "Columns mismatch."
    
    df.columns = col_names
#    # drop first column which has frameNums
#    df = df.iloc[:, 1:]
    df['frameNum'] -= (start_fno + traj_len)
    return df

if __name__ == '__main__':
    traj_len = 25
    
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    IDT_FEATS = "/home/arpan/VisionWorkspace/Cricket/StrokeAttention/logs/idt_strokes"
    base_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs"
    
    train_lst = ['ICC WT20 - Afghanistan vs South Africa - Match Highlights.avi', 
                 'ICC WT20 - England v Sri Lanka Match Highlights.avi', 
                 'ICC WT20 - India vs Australia Highlights.avi', 
                 'ICC WT20 - India vs Pakistan Match Highlights.avi', 
                 'ICC WT20 - South Africa vs England - Match Highlights.avi', 
                 'ICC WT20 Afghanistan vs Zimbabwe Match Highlights.avi', 
                 'ICC WT20 Australia vs Bangladesh - Match Highlights.avi', 
                 'ICC WT20 Bangladesh v Ireland Match Highlights.avi', 
                 'ICC WT20 Bangladesh vs Netherlands Highlights.avi', 
                 'ICC WT20 Bangladesh vs Oman Match Highlights.avi', 
                 'ICC WT20 England v New Zealand - Semi-Final Highlights.avi', 
                 'ICC WT20 Final - England vs West Indies - Match Highlights.avi', 
                 'ICC WT20 Hong Kong v Afghanistan Match Highlights.avi', 
                 'ICC WT20 Hong Kong v Scotland Highlights.avi', 
                 'ICC WT20 Hong Kong vs Zimbabwe Highlights.avi', 
                 'ICC WT20 India vs Bangladesh - Match Highlights.avi']
    
    val_lst = ['ICC WT20 Ireland v Oman Match Highlights.avi', 
               'ICC WT20 Ireland vs Oman Highlights.avi', 
               'ICC WT20 Netherlands vs Ireland Match Highlights.avi', 
               'ICC WT20 New Zealand vs Pakistan - Match Highlights.avi', 
               'ICC WT20 Scotland vs Afghanistan Highlights.avi']

    test_lst = ['ICC WT20 Scotland vs Zimbabwe Highlights.avi', 
                'ICC WT20 South Africa v Sri Lanka - Match Highlights.avi', 
                'ICC WT20 South Africa vs West Indies - Match Highlights.avi', 
                'ICC WT20 Sri Lanka v Afghanistan Cricket Match Highlights.avi', 
                'ICC WT20 West Indies v India - Semi-Final Highlights.avi']
    
    base_name = os.path.join(base_path, "bow_HL_idt_pooled"+str(traj_len))

    if not os.path.isdir(base_name):
        os.makedirs(base_name)
        
    features, strokes_name_id = read_partition_feats(DATASET, LABELS, 
                                                     IDT_FEATS+"_TrajLen"+str(traj_len),
                                                     train_lst, traj_len) 
    
    with open(os.path.join(base_name, "idt_feats_traj"+str(traj_len)+".pkl"), "wb") as fp:
        pickle.dump(features, fp)
    with open(os.path.join(base_name, "idt_snames_traj"+str(traj_len)+".pkl"), "wb") as fp:
        pickle.dump(strokes_name_id, fp)
        
        
