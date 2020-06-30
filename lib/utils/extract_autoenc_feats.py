#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 07:45:40 2020

@author: arpan

@Description: main file for training Auto-Encoder on strokes
"""

import _init_paths

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.resnet_feature_extracter import Img2Vec
#import numpy as np
#import os
#import time

#from mpl_toolkits.mplot3d import Axes3D
#from PIL import Image
from utils import autoenc_utils
import datasets.videotransforms as videotransforms
from models import autoenc
from datasets.dataset import CricketStrokesDataset
import pickle

#
## Set the path of the input videos and the openpose extracted features directory
CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#log_dir = "checkpoints"
#N_EPOCHS = 15
#SEQ_SIZE = 6
#INPUT_SIZE = 2048
#HIDDEN_SIZE = 32#64#1024
#NUM_LAYERS = 2
#BATCH_SIZE = 2*SEQ_SIZE # set to the number of images of a seqence # 36


def extract_sequence_feats(model_path, DATASET, LABELS, BATCH_SIZE, SEQ_SIZE, \
                          INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, nfrms_per_clip=1, \
                          partition='all', nstrokes=-1):
    
    ###########################################################################
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
    # get list of label filenames containing temporal stroke and batsman labels
#    train_lab = [f.rsplit('.',1)[0] +".json" for f in train_lst]
#    val_lab = [f.rsplit('.',1)[0] +".json" for f in val_lst]
#    test_lab = [f.rsplit('.',1)[0] +".json" for f in test_lst]
#    train_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in train_lst]
#    val_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in val_lst]
#    test_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in test_lst]
    
    #####################################################################
    
    if partition == 'all':
        partition_lst = train_lst
        partition_lst.extend(val_lst)
        partition_lst.extend(test_lst)
    elif partition == 'train':
        partition_lst = train_lst
    elif partition == 'val':
        partition_lst = val_lst
    elif partition == 'test':
        partition_lst = test_lst
        
#    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
#    val_labs = [os.path.join(LABELS, f) for f in val_lab]
#    tr_gt_batsman = [os.path.join(BAT_LABELS, f) for f in train_gt_batsman]
#    sizes = [autoenc_utils.getTotalFramesVid(os.path.join(DATASET, f)) for f in train_lst]
#    print("Size : {}".format(sizes))
    
    ###########################################################################
    # Create a Dataset
    
    clip_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225]),]) \
                     if nfrms_per_clip == 1 else transforms.Compose([videotransforms.RandomCrop(112), \
                                           #videotransforms.RandomHorizontalFlip(),\
                                           ])
#    test_transforms = transforms.Compose([videotransforms.CenterCrop(112)])
    
    # Prepare datasets and dataset loaders (training and validation/testing)
#    train_dataset = CricketStrokesDataset(train_lst, DATASET, LABELS, CLASS_IDS, 
#                                          frames_per_clip=SEQ_SIZE, train=True, 
#                                          transform=clip_transform)
    part_dataset = CricketStrokesDataset(partition_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=nfrms_per_clip, train=True, 
                                         transform=clip_transform)
    

    data_loader = DataLoader(dataset=part_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#    data_loader = {'train': train_loader, 'test': val_loader}

    ###########################################################################
    # Create a model and load the weights 
    model = autoenc.AutoEncoderRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    print("Loading model ...")
    model.load_state_dict(torch.load(model_path))
    
    model = model.to(device)

    ###########################################################################
    # Validate / Evaluate
    model.eval()
    stroke_names = []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    extractor = Img2Vec()
    prev_stroke = None
    
    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loader.__len__(), BATCH_SIZE))
    assert BATCH_SIZE % SEQ_SIZE == 0, "BATCH_SIZE should be a multiple of SEQ_SIZE"
    for bno, (inputs, vid_path, stroke, _) in enumerate(data_loader):
        print("Batch No : {}".format(bno))
        inputs = extractor.get_vec(inputs)
        # convert to start frames and end frames from tensors to lists
        stroke = [s.tolist() for s in stroke]
        inputs_lst, batch_stroke_names = autoenc_utils.separate_stroke_tensors(inputs, \
                                                                    vid_path, stroke)
        
        if bno == 0:
            prev_stroke = batch_stroke_names[0]
        
        for enc_idx, enc_input in enumerate(inputs_lst):
            # get no of sequences that can be extracted from enc_input tensor
            nSeqs = int(enc_input.size(0) / SEQ_SIZE)
            if prev_stroke == batch_stroke_names[enc_idx]:
                if nSeqs == 0:
                    if len(stroke_traj) > 0:
                        num_strokes += 1
                        # append current stroke to trajectories
                        trajectories.append(stroke_traj)
                        stroke_names.append(batch_stroke_names[enc_idx])
                        stroke_traj = []
                    continue
                # send first nSeqs to encoder
                enc_output = autoenc_utils.enc_forward(model, enc_input[:(nSeqs*SEQ_SIZE), ...], \
                                         SEQ_SIZE, INPUT_SIZE, device)
                # convert to [[[stroke1(size 32 each) ... ], [], ...], [ [], ... ]]
                stroke_traj.extend([enc_output[i, :] for i in range(enc_output.shape[0])])
#                sequence_outputs.append(enc_output)
                #seq_stroke_names.append(batch_stroke_names[enc_idx])
                #prev_stroke = seq_stroke_names[-1]
            else:
                # append old stroke to trajectories
                if len(stroke_traj) > 0:
                    num_strokes += 1
                    trajectories.append(stroke_traj)
                    stroke_names.append(prev_stroke)
                    stroke_traj = []
                if nSeqs > 0:
                    # send last nSeqs to encoder
                    enc_output = autoenc_utils.enc_forward(model, enc_input[-(nSeqs*SEQ_SIZE):, ...], \
                                             SEQ_SIZE, INPUT_SIZE, device)
                    stroke_traj.extend([enc_output[i, :] for i in range(enc_output.shape[0])])
#                    sequence_outputs.append(enc_output)
                prev_stroke = batch_stroke_names[enc_idx]
                
        if num_strokes == nstrokes:
            break
       
    # for last batch only if extracted for full dataset
    if len(stroke_traj) > 0 and nstrokes < 0:
        trajectories.append(stroke_traj)
        stroke_names.append(batch_stroke_names[-1])
        
    trajectories, stroke_names = autoenc_utils.group_strokewise(trajectories, stroke_names)
    #stroke_vecs, stroke_names =  aggregate_outputs(sequence_outputs, seq_stroke_names)
    #stroke_vecs = [stroke.cpu().data.numpy() for stroke in stroke_vecs]
    
    # save to disk
#    np.save("trajectories.npy", trajectories)
#    with open('stroke_names_val.pkl', 'wb') as fp:
#        pickle.dump(stroke_names, fp)

    # read the files from disk
#    trajectories = np.load("trajectories.npy")
#    with open('stroke_names_val.pkl', 'rb') as fp:
#        stroke_names = pickle.load(fp)

    #plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], bins, thresh, 'cluster_pca_ordered.png')    
#    print(len(stroke_vecs))
#    print("#Parameters : {}".format(autoenc_utils.count_parameters(model)))
    
    return trajectories, stroke_names