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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from features.resnet_feature_extracter import Img2Vec, Clip2Vec
import numpy as np
import pandas as pd
import os
import time
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from PIL import Image
from utils import autoenc_utils
import datasets.videotransforms as videotransforms
from models import autoenc
from datasets.dataset import CricketStrokesDataset
from math import fabs
from collections import defaultdict
import pickle
from utils import trajectory_utils as traj_utils
from utils import spectral_utils
from utils import plot_utils
from sklearn.preprocessing import normalize
from models.select_backbone import select_resnet


CLASS_IDS = "configs/Class Index_Strokes.txt"

# Local Paths
DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
POSE_FEATS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/data/output_json"
BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_dir = "checkpoints"
N_EPOCHS = 20
SEQ_SIZE = 19
CLIP_SIZE = 16
INPUT_SIZE = 2048
HIDDEN_SIZE = 32#64#1024
NUM_LAYERS = 2
BATCH_SIZE = 16 # set to the number of images of a seqence # 36
#model_path = os.path.join(log_dir, "28_3_batsman_lstm_autoencoder_model_wholeDataset.pt")
model_checkpoint = 'autoenc_gru_resnet3D_ep'

# takes a model to train along with dataset, optimizer and criterion
def train(model, datasets_loader, optimizer, scheduler, criterion, extractor, nEpochs,
          start_ep=0, use_gpu=True):
    global training_stats
    training_stats = defaultdict()
#    best_model_wts = copy.deepcopy(model.state_dict())
#    best_acc = 0.0
    
    for epoch in range(nEpochs):
        if epoch < start_ep:
            continue
        print("-"*60)
        print("Epoch -> {} ".format((epoch+1)))
        training_stats[epoch] = {}
        # for each epoch train the model and then evaluate it
        for phase in ['train', 'test']:
            #print("phase->", phase)
            dataset = datasets_loader[phase]
            training_stats[epoch][phase] = {}
            accuracy = 0
            net_loss = 0
            if phase == 'train':
                if torch.__version__.split('.')[0]=='0':
                    scheduler.step()
                model.train(True)
            elif phase == 'test':
                #print("validation")
                model.train(False)
            
            for i, (inputs, vid_path, stroke, labels) in enumerate(dataset):
                
                (B, SL, C, H, W) = inputs.shape
                if torch.__version__.split('.')[0]=='0':
                    inputs = torch.autograd.Variable(inputs)

                # get inputs of shape (B, S, H, W, C). Convert to (B*S, C, H, W)
                #inputs = inputs.permute(0, 4, 1, 2, 3).float()     # For 3D backbone
#                inputs = inputs.permute(0, 1, 4, 2, 3).view(-1, C, H, W)
                # Extract spatial features using 2D ResNet, one batch all sequences at a time
                if isinstance(extractor, Img2Vec):
                    inputs = torch.stack([extractor.get_vec(x) for x in inputs])
                # Extract spatio-temp features using 3D ResNet, one clip of all batches at a time. 
                else:
                    # for SEQ_LEN >= 16, iterate on sequences of 16 frame window clips
                    seq_input = []
#                    inputs = inputs.permute(0, 2, 1, 3, 4)
#                    seq_input = [extractor.get_vec(inputs[:,:,sl:(sl+CLIP_SIZE), ...]) \
#                                                   for sl in range(SL-CLIP_SIZE+1)]
                    for sl in range(SL-CLIP_SIZE+1):
                        input_blob = inputs[:, sl:(sl+CLIP_SIZE), ...].clone()
                        input_blob = input_blob.permute(0, 2, 1, 3, 4)
                        seq_input.append(extractor.get_vec(input_blob))
                    seq_input = torch.cat(seq_input, dim=1)
        
                seq_input = seq_input.reshape(-1, SEQ_SIZE-CLIP_SIZE+1, INPUT_SIZE).to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(seq_input)

                    inv_idx = torch.arange(SEQ_SIZE - CLIP_SIZE, -1, -1).long()
                    loss = criterion(outputs, seq_input[:, inv_idx, :])

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                net_loss += loss.item() * inputs.size(0)
                
#                print("Phase : {} :: Batch : {} :: Loss : {} :: Accuracy : {}"\
#                          .format(phase, (i+1), net_loss, accuracy))
#                if phase == 'train':
#                    optimizer.zero_grad()
#                    loss.backward()
#                    optimizer.step()
                if (i+1) == 100:
                    break
#            accuracy = fabs(accuracy)/len(datasets_loader[phase].dataset)
            accuracy = fabs(accuracy)/(BATCH_SIZE*SEQ_SIZE*(i+1))
            training_stats[epoch][phase]['loss'] = net_loss
            training_stats[epoch][phase]['acc'] = accuracy
            training_stats[epoch][phase]['lr'] = optimizer.param_groups[0]['lr']
            
            if phase == 'train' and torch.__version__.split('.')[0]=='1':
                scheduler.step()
        # Display at end of epoch
        print("Phase {} : Train :: Epoch : {} :: Loss : {} :: Accuracy : {} : LR : {}"\
              .format(i, (epoch+1), training_stats[epoch]['train']['loss'],\
                      training_stats[epoch]['train']['acc'], \
                                    optimizer.param_groups[0]['lr']))
        print("Phase : Test :: Epoch : {} :: Loss : {} :: Accuracy : {}"\
              .format((epoch+1), training_stats[epoch]['test']['loss'],\
                      training_stats[epoch]['test']['acc']))
        
        if ((epoch+1)%10) == 0:
            save_model_checkpoint(model, optimizer, epoch+1, "Adam", win=SEQ_SIZE, use_gpu=True)

    # Save dictionary after all the epochs
    save_stats_dict(training_stats)
    # Training finished
    return model

def save_model_checkpoint(model, optimizer, epoch, opt, win=16, use_gpu=True):
    # Save only the model params
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    name = os.path.join(log_dir, model_checkpoint+str(epoch)+"_w"+str(win)+"_"+opt+".pt")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, name)
    print("Model saved to disk... {}".format(name))
    
def load_saved_checkpoint(model, optimizer, epoch, opt, win):
    assert os.path.exists(log_dir), "Checkpoint path does not exist."
    name = os.path.join(log_dir, model_checkpoint+str(epoch)+"_w"+str(win)+"_"+opt+".pt")
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    # now individually transfer the optimizer parts...
    # https://github.com/pytorch/pytorch/issues/2830
#    for state in optimizer.state.values():
#        for k, v in state.items():
#            if isinstance(v, torch.Tensor):
#                state[k] = v.to(device)
                
    return model, optimizer, epoch

# save training stats
def save_stats_dict(stats):
    with open(os.path.join(log_dir, model_checkpoint+'_stats.pickle'), 'wb') as fr:
        pickle.dump(stats, fr, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def copy_pretrained_weights(model_src, model_tar):
    params_src = model_src.named_parameters()
    params_tar = model_tar.named_parameters()
    dict_params_tar = dict(params_tar)
    for name_src, param_src in params_src:
        if name_src in dict_params_tar:
            dict_params_tar[name_src].data.copy_(param_src.data)
            dict_params_tar[name_src].requires_grad = False     # Freeze layer wts


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
    
    ###########################################################################
    # Create a Dataset
#    data_transforms = transforms.Compose([transforms.ToPILImage(),
#                                          transforms.Resize((224, 224)),
#                                          transforms.ToTensor(),
#                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                                               std=[0.229, 0.224, 0.225]),])
    # Clip level transform. Use this with framewiseTransform flag turned off
    clip_transform = transforms.Compose([videotransforms.CenterCrop(224),
                                         videotransforms.ToPILClip(), 
                                         videotransforms.Resize((112, 112)),
#                                         videotransforms.RandomCrop(112), 
                                         videotransforms.ToTensor(), 
                                         videotransforms.Normalize(),
                                        #videotransforms.RandomHorizontalFlip(),\
                                        ])
    
    # Prepare datasets and dataset loaders (training and validation/testing)
    train_dataset = CricketStrokesDataset(train_lst, DATASET, LABELS, CLASS_IDS, 
                                          frames_per_clip=SEQ_SIZE, train=True,
                                          framewiseTransform=False,
                                          transform=clip_transform)
#    train_dataset = CricketStrokesDataset(train_lst, DATASET, LABELS, CLASS_IDS, 
#                                          frames_per_clip=1, train=True, 
#                                          transform=data_transforms)
    val_dataset = CricketStrokesDataset(val_lst, DATASET, LABELS, CLASS_IDS, 
                                          frames_per_clip=SEQ_SIZE, 
                                          step_between_clips=SEQ_SIZE//2, train=False, 
                                          framewiseTransform=False,
                                          transform=clip_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    data_loader = {'train': train_loader, 'test': val_loader}

    ###########################################################################
    # Get the ResNet3D feature extractor and retrieve the output dimension from it
    extractor = Clip2Vec()
    INPUT_SIZE = extractor.layer_output_size
#    extractor = Img2Vec()
    
    ###########################################################################
    # Create a model
    model = autoenc.AutoEncoderRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, SEQ_SIZE-CLIP_SIZE+1)
#    print("Loading model ...")
#    model.load_state_dict(torch.load(model_path))
    
    model = model.to(device)
    criterion = nn.MSELoss()
#    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                 lr = 0.01)
#    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), \
#                                 lr = 0.001, momentum=0.9)
    
    # set the scheduler, optimizer and retrain (eg. SGD)
    # Decay LR by a factor of 0.1 every 7 epochs
    step_lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    
    ###########################################################################
    #####################################################################
    # Load checkpoint model and optimizer state
    s_epoch = 0
    for i in range(N_EPOCHS, 0, -1):    
        if os.path.isfile(os.path.join(log_dir, model_checkpoint+str(i)+"_w"+str(SEQ_SIZE)+"_Adam.pt")):
            model, optimizer, s_epoch = load_saved_checkpoint(model, optimizer, i, \
                                                              "Adam", SEQ_SIZE)
            break

    #####################################################################
    
    start = time.time()
    s = start
    
#    # Training (finetuning) and validating
#    model = train(model, data_loader, optimizer, step_lr_scheduler, criterion, 
#                  extractor, nEpochs=N_EPOCHS, start_ep=s_epoch, use_gpu=True)
        
    end = time.time()
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))
    
    
    ###########################################################################
    # Validate / Evaluate
    model.eval()
    sequence_outputs, seq_stroke_names, stroke_names = [], [], []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    prev_stroke = None
    
    print("Total Batches : {} :: BATCH_SIZE : {}".format(val_loader.__len__(), BATCH_SIZE))
#    assert BATCH_SIZE % SEQ_SIZE == 0, "BATCH_SIZE should be a multiple of SEQ_SIZE"
    for bno, (inputs, vid_path, stroke, _) in enumerate(val_loader):
        
        print("Batch No : {}".format(bno))
        (SL, C, H, W) = inputs.shape[-4:]
        # inputs of size (B, SL, C, H, W) for SL>=1 or (B, C, H, W) fpr SL=1
        # stack after feature extraction to shape (B, SL-CLIP_SIZE, FeatWidth)
        if isinstance(extractor, Img2Vec):
            inputs = torch.stack([extractor.get_vec(x) for x in inputs])
        # Extract spatio-temporal features from clip using 3D ResNet
        else:
            # for SEQ_LEN >= 16, iterate on sequences of 16 frame window clips
            seq_input = []
            for sl in range(SL-CLIP_SIZE+1):
                input_blob = inputs[:, sl:(sl+CLIP_SIZE), ...].clone()
                input_blob = input_blob.permute(0, 2, 1, 3, 4)
                seq_input.append(extractor.get_vec(input_blob))
            inputs = torch.cat(seq_input, dim=1)
            
        inputs = inputs.reshape(-1, SEQ_SIZE-CLIP_SIZE+1, INPUT_SIZE).to(device)
            
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
            enc_output = enc_output.squeeze(axis=1)
            enc_output = enc_output.view(enc_output.shape[0], -1).cpu().data.numpy()
            # convert to [[[stroke1(size 32 each) ... ], [], ...], [ [], ... ]]
            stroke_traj.extend([enc_output[i, :] for i in range(enc_output.shape[0])])
            prev_stroke = batch_stroke_names[enc_idx]
        if num_strokes >= 20:
            break
            
    # for last batch
    if len(stroke_traj) > 0 :
        trajectories.append(stroke_traj)
        stroke_names.append(batch_stroke_names[-1])
                
    trajectories, stroke_names = autoenc_utils.group_strokewise(trajectories, stroke_names)
    #stroke_vecs, stroke_names =  aggregate_outputs(sequence_outputs, seq_stroke_names)
    #stroke_vecs = [stroke.cpu().data.numpy() for stroke in stroke_vecs]
    
    # save to disk
#    np.save("trajectories_resnet50_pre.npy", trajectories)
#    with open('stroke_names_resnet50_pre.pkl', 'wb') as fp:
#        pickle.dump(stroke_names, fp)
#
#    # read the files from disk
#    trajectories = np.load("trajectories_resnet50_pre.npy")
#    with open('stroke_names_resnet50_pre.pkl', 'rb') as fp:
#        stroke_names = pickle.load(fp)
    
    # plot the strokes as sequence of points
    plot_utils.plot_trajectories3D(trajectories, stroke_names, log_dir)
    
    #plot_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], bins, thresh, 'cluster_pca_ordered.png')
    
#    print(len(stroke_vecs))
    
#    print("#Parameters : {}".format(autoenc_utils.count_parameters(model)))
    