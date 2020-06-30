#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 07:51:38 2020

@author: arpan

@Description: Utils functions for training AutoEncoder
"""

import _init_paths

import os
import cv2
import torch
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from utils import trajectory_utils as traj_utils


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

def separate_stroke_tensors(inputs, vid_path, stroke):
    '''
    Separate the features of different strokes in a lists.
    
    Parameters:
    -----------
    inputs : Torch.tensor (size = batch x feature_width)
        tensor for a batch of sequence inputs, extracted from feature extractor
    vid_path : tuple of str (len = batch)
        tuples of stroke paths, got from data_loader
    stroke : list of list of int
        Len = 2. Two lists of int representing starting and ending frame positions
        
    Returns:
    --------
    list of Torch.tensors for different strokes
    
    '''
    from collections import Counter
    
    st_frms_dict, end_frms_dict = Counter(stroke[0]), Counter(stroke[1])
    st_frms_keys = [x for i, x in enumerate(stroke[0]) if i == stroke[0].index(x)]
    end_frms_keys = [x for i, x in enumerate(stroke[1]) if i == stroke[1].index(x)]
    
    assert len(st_frms_dict) == len(end_frms_dict), "Invalid stroke pts."
    stroke_vectors, stroke_names = [], []
    
    in_idx = 0
    for i, st_frm in enumerate(st_frms_keys):
        stroke_vectors.append(inputs[in_idx:(in_idx + st_frms_dict[st_frm]), ...])
        curr_stroke = vid_path[in_idx]+'_'+str(st_frm)+'_'+str(end_frms_keys[i])
        stroke_names.append(curr_stroke)
        in_idx += st_frms_dict[st_frm]
    
    return stroke_vectors, stroke_names
    

def plot_sequences(stroke_vecs, stroke_names):
    
    stroke_lens = [stroke.size()[0] for stroke in stroke_vecs]
    all_feats = torch.cat(stroke_vecs, axis=0).data.numpy()
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure(2)
    
    all_feats = normalize(all_feats, norm='l2')
    pca_applied = traj_utils.apply_PCA(all_feats)
    
    indx = 0
    for i, slen in enumerate(stroke_lens):
#        plt.plot(pca_applied[indx:(indx+slen), 0], pca_applied[indx:(indx+slen), 1], \
#                             color=colors[i % len(colors)], marker='.', lw=1 )
        x = pca_applied[indx:(indx+slen), 0]
        y = pca_applied[indx:(indx+slen), 1]
        plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], \
                   color=colors[i % len(colors)],  scale_units='xy', angles='xy', \
                   scale=1, width=0.003, headwidth=6)      
        indx += slen
    
    plt.show()

def group_strokewise(trajectories, stroke_names):
    '''Receive list of trajectory arrays in sublists corresponding to strokes defined
    in stroke_names. Convert to sub-sublist by grouping video wise and stroke wise.
    
    '''
    traj, names, vid_traj, vid_names = [], [], [], []
    assert len(trajectories) == len(stroke_names), "Trajectories and Names length mismatch"
    if len(stroke_names) > 0:
        prev_vname = stroke_names[0].rsplit('_', 2)[0]
    for idx, sname in enumerate(stroke_names):
        # different video when stroke changes
        if prev_vname not in sname:
            traj.append(vid_traj)
            names.append(vid_names)
            vid_traj, vid_names = [], []
            
        # append stroke trajectories of same video
        vid_traj.append(trajectories[idx])
        vid_names.append(sname)
        prev_vname = sname.rsplit('_', 2)[0]
        
    # last video
    if len(vid_names) > 0:
        traj.append(vid_traj)
        names.append(vid_names)
    
    return traj, names
        
def enc_forward(model, enc_input, SEQ_SIZE, INPUT_SIZE, device):
    assert enc_input.shape[0] % SEQ_SIZE == 0, "NRows not a multiple of SEQ_SIZE"
    # reshape to (-1, SEQ_SIZE, INPUT_SIZE)
    enc_input = enc_input.reshape(-1, SEQ_SIZE, INPUT_SIZE).to(device)
    enc_output = model.encoder(enc_input) # (-1, 1, 32) lstm last layer
    enc_output = enc_output.squeeze(axis = 1).cpu().data.numpy() # (-1, 32)
    return enc_output

