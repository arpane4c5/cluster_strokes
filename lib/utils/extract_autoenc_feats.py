#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 07:45:40 2020

@author: arpan

@Description: main file for training Auto-Encoder on strokes
"""

import _init_paths

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.resnet_feature_extracter import Img2Vec, Clip2Vec
from utils import autoenc_utils
import datasets.videotransforms as videotransforms
from models import autoenc
from datasets.dataset import CricketStrokesDataset
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_2DCNN_feats(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, \
                        partition='all', nstrokes=-1):
    '''
    Extract sequence features from AutoEncoder.
    
    Parameters:
    -----------
    DATASET : str
        path to the video dataset
    LABELS : str
        path containing stroke labels
    CLASS_IDS : str
        path to txt file defining classes, similar to THUMOS
    BATCH_SIZE : int
        size for batch of clips
        Autoencoder.
    partition : str
        'all' / 'train' / 'test' / 'val' : Videos to be considered
    nstrokes : int
        partial extraction of features (if don't want to execute for entire dataset)
    
    Returns:
    --------
    trajectories, stroke_names
    
    '''
    
    ###########################################################################
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
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
        
    ###########################################################################
    # Create a Dataset    
    # Frame-wise transform
    clip_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                              std=[0.229, 0.224, 0.225]),]) 
    # For using Frame level transform, the framewiseTransform flag turned on
    part_dataset = CricketStrokesDataset(partition_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=1, train=True, 
                                         framewiseTransform=True,
                                         transform=clip_transform)
    
    data_loader = DataLoader(dataset=part_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ###########################################################################
    # Extract using the data_loader
    stroke_names = []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    extractor = Img2Vec()
    #INPUT_SIZE = extractor.layer_output_size
    prev_stroke = None
    
    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loader.__len__(), BATCH_SIZE))
    for bno, (inputs, vid_path, stroke, _) in enumerate(data_loader):
        # get video clips (B, SL, C, H, W)
        print("Batch No : {}".format(bno))
        # Extract spatial features using 2D ResNet
        inputs = torch.stack([extractor.get_vec(x) for x in inputs])
            
        # convert to start frames and end frames from tensors to lists
        stroke = [s.tolist() for s in stroke]
        inputs_lst, batch_stroke_names = autoenc_utils.separate_stroke_tensors(inputs, \
                                                                    vid_path, stroke)
        
        if bno == 0:
            prev_stroke = batch_stroke_names[0]
        
        for enc_idx, enc_input in enumerate(inputs_lst):
            # get no of sequences that can be extracted from enc_input tensor
            nSeqs = enc_input.size(0)
            if prev_stroke != batch_stroke_names[enc_idx]:
                # append old stroke to trajectories
                if len(stroke_traj) > 0:
                    num_strokes += 1
                    trajectories.append(stroke_traj)
                    stroke_names.append(prev_stroke)
                    stroke_traj = []
            
            # enc_input is same as enc_output while extraction of features.
            enc_output = enc_input
            enc_output = enc_output.squeeze(axis=1).cpu().data.numpy()
            # convert to [[[stroke1(size 32 each) ... ], [], ...], [ [], ... ]]
            stroke_traj.extend([enc_output[i, :] for i in range(enc_output.shape[0])])
            prev_stroke = batch_stroke_names[enc_idx]
            
        if nstrokes >= -1 and num_strokes == nstrokes:
            break
       
    # for last batch only if extracted for full dataset
    if len(stroke_traj) > 0 and nstrokes < 0:
        trajectories.append(stroke_traj)
        stroke_names.append(batch_stroke_names[-1])
        
    trajectories, stroke_names = autoenc_utils.group_strokewise(trajectories, stroke_names)
    #stroke_vecs, stroke_names =  aggregate_outputs(sequence_outputs, seq_stroke_names)    
    return trajectories, stroke_names

def extract_3DCNN_feats(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, SEQ_SIZE=16, STEP=16, \
                        model_path=None, nclasses=5, partition='all', nstrokes=-1):
    '''
    Extract sequence features from AutoEncoder.
    
    Parameters:
    -----------
    DATASET : str
        path to the video dataset
    LABELS : str
        path containing stroke labels
    CLASS_IDS : str
        path to txt file defining classes, similar to THUMOS
    BATCH_SIZE : int
        size for batch of clips
    SEQ_SIZE : int
        no. of frames in a clip (min. 16 for 3D CNN extraction)
    STEP : int
        stride for next example. If SEQ_SIZE=16, STEP=8, use frames (0, 15), (8, 23) ...
    partition : str
        'all' / 'train' / 'test' / 'val' : Videos to be considered
    nstrokes : int
        partial extraction of features (do not execute for entire dataset)
    
    Returns:
    --------
    trajectories, stroke_names
    
    '''
    
    ###########################################################################
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
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
        
    ###########################################################################
    # Create a Dataset    
    # Clip level transform. Use this with framewiseTransform flag turned off
    clip_transform = transforms.Compose([videotransforms.CenterCrop(224),
                                         videotransforms.ToPILClip(), 
                                         videotransforms.Resize((112, 112)),
#                                         videotransforms.RandomCrop(112), 
                                         videotransforms.ToTensor(), 
                                         videotransforms.Normalize(),
                                        #videotransforms.RandomHorizontalFlip(),\
                                        ])

    part_dataset = CricketStrokesDataset(partition_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, 
                                         step_between_clips=STEP, train=True, 
                                         framewiseTransform=False,
                                         transform=clip_transform)
    
    data_loader = DataLoader(dataset=part_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ###########################################################################
    # Validate / Evaluate
    stroke_names = []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    extractor = Clip2Vec(model_path, nclasses)
    #INPUT_SIZE = extractor.layer_output_size
    prev_stroke = None
    
    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loader.__len__(), BATCH_SIZE))
    assert SEQ_SIZE>=16, "SEQ_SIZE should be >= 16"
    for bno, (inputs, vid_path, stroke, _) in enumerate(data_loader):
        # get video clips (B, SL, C, H, W)
        print("Batch No : {}".format(bno))
        # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
        inputs = inputs.permute(0, 2, 1, 3, 4).float()
        inputs = extractor.get_vec(inputs)
        
        # convert to start frames and end frames from tensors to lists
        stroke = [s.tolist() for s in stroke]
        inputs_lst, batch_stroke_names = autoenc_utils.separate_stroke_tensors(inputs, \
                                                                    vid_path, stroke)
        
        if bno == 0:
            prev_stroke = batch_stroke_names[0]
        
        for enc_idx, enc_input in enumerate(inputs_lst):
            # get no of sequences that can be extracted from enc_input tensor
            nSeqs = enc_input.size(0)
            if prev_stroke != batch_stroke_names[enc_idx]:
                # append old stroke to trajectories
                if len(stroke_traj) > 0:
                    num_strokes += 1
                    trajectories.append(stroke_traj)
                    stroke_names.append(prev_stroke)
                    stroke_traj = []
            
            enc_output = enc_input
            enc_output = enc_output.squeeze(axis=1).cpu().data.numpy()
            # convert to [[[stroke1(size 32 each) ... ], [], ...], [ [], ... ]]
            stroke_traj.extend([enc_output[i, :] for i in range(enc_output.shape[0])])
            prev_stroke = batch_stroke_names[enc_idx]
            
                
        if nstrokes >=-1 and num_strokes == nstrokes:
            break
       
    # for last batch only if extracted for full dataset
    if len(stroke_traj) > 0 and nstrokes < 0:
        trajectories.append(stroke_traj)
        stroke_names.append(batch_stroke_names[-1])
        
    trajectories, stroke_names = autoenc_utils.group_strokewise(trajectories, stroke_names)
    #stroke_vecs, stroke_names =  aggregate_outputs(sequence_outputs, seq_stroke_names)
    return trajectories, stroke_names


def extract_sequence_feats(model_path, DATASET, LABELS, CLASS_IDS, BATCH_SIZE, 
                           INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, SEQ_SIZE=16, STEP=16, 
                           partition='all', nstrokes=-1):
    '''
    Extract sequence features from AutoEncoder.
    
    Parameters:
    -----------
    model_path : str
        relative path to the checkpoint file for Autoencoder
    DATASET : str
        path to the video dataset
    LABELS : str
        path containing stroke labels
    BATCH_SIZE : int
        size for batch of clips
    INPUT_SIZE : int
        size of the extracted feature vector (output of ResNet). Input size of
        Autoencoder.
    HIDDEN_SIZE : int
        hidden size of autoencoder. #Parameters of autoencoder depend on it.
    NUM_LAYERS : int
        No. of GRU layers in the Autoencoder
    SEQ_SIZE : int
        no. of frames in a clip
    STEP : int
        stride for next example. If SEQ_SIZE=16, STEP=8, use frames (0, 15), (8, 23) ...
    partition : str
        'all' / 'train' / 'test' / 'val' : Videos to be considered
    nstrokes : int
        partial extraction of features (do not execute for entire dataset)
    
    Returns:
    --------
    trajectories, stroke_names
    
    '''
    
    ###########################################################################
    # Read the strokes 
    # Divide the highlight dataset files into training, validation and test sets
    train_lst, val_lst, test_lst = autoenc_utils.split_dataset_files(DATASET)
    print("No. of training videos : {}".format(len(train_lst)))
    
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
        
    ###########################################################################
    # Create a Dataset    
    # Frame-wise transform
    clip_transform = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                              std=[0.229, 0.224, 0.225]),]) 
    
    # Clip level transform. Use this with framewiseTransform flag turned off
#    clip_transform = transforms.Compose([videotransforms.ToPILClip(), 
#                                         videotransforms.Resize((112, 112)),
##                                         videotransforms.RandomCrop(112), 
#                                         videotransforms.ToTensor(), 
#                                         videotransforms.Normalize(),
#                                        #videotransforms.RandomHorizontalFlip(),\
#                                        ])

    part_dataset = CricketStrokesDataset(partition_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, step_between_clips=STEP,
                                         train=True, framewiseTransform=True,
                                         transform=clip_transform)
    
    data_loader = DataLoader(dataset=part_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ###########################################################################
    # Create a model and load the weights of AutoEncoder
    model = autoenc.AutoEncoderRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    if model_path is not None:
        print("Loading model ...")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)

    ###########################################################################
    # Validate / Evaluate
    model.eval()
    stroke_names = []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    extractor = Img2Vec()
#    extractor = Clip2Vec()
    #INPUT_SIZE = extractor.layer_output_size
    prev_stroke = None
    
    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loader.__len__(), BATCH_SIZE))
#    assert BATCH_SIZE % SEQ_SIZE == 0, "BATCH_SIZE should be a multiple of SEQ_SIZE"
    for bno, (inputs, vid_path, stroke, _) in enumerate(data_loader):
        # get video clips (B, SL, C, H, W)
        print("Batch No : {}".format(bno))
        # Extract spatial features using 2D ResNet
        if isinstance(extractor, Img2Vec):
            inputs = torch.stack([extractor.get_vec(x) for x in inputs])
        # Extract spatio-temporal features from clip using 3D ResNet
        else:
            # for SEQ_LEN >= 16
            inputs = inputs.permute(0, 2, 1, 3, 4).float()
            inputs = extractor.get_vec(inputs)
            
        # convert to start frames and end frames from tensors to lists
        stroke = [s.tolist() for s in stroke]
        inputs_lst, batch_stroke_names = autoenc_utils.separate_stroke_tensors(inputs, \
                                                                    vid_path, stroke)
        
        if bno == 0:
            prev_stroke = batch_stroke_names[0]
        
        for enc_idx, enc_input in enumerate(inputs_lst):
            # get no of sequences that can be extracted from enc_input tensor
            nSeqs = enc_input.size(0)
            if prev_stroke != batch_stroke_names[enc_idx]:
                # append old stroke to trajectories
                if len(stroke_traj) > 0:
                    num_strokes += 1
                    trajectories.append(stroke_traj)
                    stroke_names.append(prev_stroke)
                    stroke_traj = []
            
            enc_output = model.encoder(enc_input.to(device))
            enc_output = enc_output.squeeze(axis=1).cpu().data.numpy()
            # convert to [[[stroke1(size 32 each) ... ], [], ...], [ [], ... ]]
            stroke_traj.extend([enc_output[i, :] for i in range(enc_output.shape[0])])
            prev_stroke = batch_stroke_names[enc_idx]
            
                
        if nstrokes >= -1 and num_strokes >= nstrokes:
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

#    print("#Parameters : {}".format(autoenc_utils.count_parameters(model)))
    
    return trajectories, stroke_names

if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"
    # Server Paths
    if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
        LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
        DATASET = "/opt/datasets/cricket/ICC_WT20"
    
    model_path = "checkpoints/autoenc_gru_resnet50_ep10_w6_Adam.pt"
    SEQ_SIZE = 16
    STEP = 16
    INPUT_SIZE = 2048
    HIDDEN_SIZE = 32#64#1024
    NUM_LAYERS = 2
    BATCH_SIZE = 8 
    # Extract autoencoder features 
#    trajectories, stroke_names = extract_sequence_feats(model_path, DATASET, LABELS, 
#                                    CLASS_IDS, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, 
#                                    NUM_LAYERS, SEQ_SIZE, STEP, partition='all', 
#                                    nstrokes=3)
    
#    trajectories, stroke_names = extract_2DCNN_feats(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, partition='all', nstrokes=3)
    
    trajectories, stroke_names = extract_3DCNN_feats(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, partition='all', nstrokes=3)