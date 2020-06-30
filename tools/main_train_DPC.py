#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 23:38:08 2020

@author: ashish
"""


import _init_paths

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from utils.resnet_feature_extracter import Img2Vec
import numpy as np
import pandas as pd
import os
import time
import copy
import matplotlib.pyplot as plt
#from PIL import Image
from utils import autoenc_utils
import datasets.videotransforms as videotransforms
from models import autoenc
from datasets.dataset import CricketStrokesDataset
from math import fabs
from collections import defaultdict
import pickle
from utils import cluster_of_utils as clus_utils
from sklearn.preprocessing import normalize

from models.model_full_3D_resnet18 import DPC_RNN

# Set the path of the input videos and the openpose extracted features directory
# Server Paths
CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"
DATASET = "/home/ashish/Desktop/BTP/flow_cluster/Cricket_Highlights_dataset/"
#POSE_FEATS = "/home/arpan/cricket/output_json"
#BAT_LABELS = "/home/arpan/VisionWorkspace/Cricket/batsman_pose_track/batsman_pose_gt"
LABELS = "/home/ashish/Desktop/BTP/UntrimmedStrokeLocalization/sample_labels_shots/ICC WT20/"
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

model_path = "checkpoints/28_3_batsman_lstm_autoencoder_model_wholeDataset.pt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_dir = "checkpoints"
N_EPOCHS = 2
SEQ_SIZE = 16
'''
INPUT_SIZE = 2048
HIDDEN_SIZE = 32#64#1024
NUM_LAYERS = 2
'''
BATCH_SIZE = 1 # set to the number of images of a seqence # 36

# takes a model to train along with dataset, optimizer and criterion
def train(model, datasets_loader, optimizer, scheduler, criterion, nEpochs, start_ep=0, \
          use_gpu=True):
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
        for phase in ['train']:
            print("phase->", phase)
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
            
            for i, (video, vid_path, stroke, labels) in enumerate(dataset):
                
                # return a 16 x ch x depth x H x W 
                x, y = video, labels
                '''
                if torch.__version__.split('.')[0]=='0':
                    x, y = torch.autograd.Variable(x) , torch.autograd.Variable(y)
                '''
                x = x.view(1, 2, 3, 16, 112, 112)
                if torch.__version__.split('.')[0]=='0':
                    x = torch.autograd.Variable(x)
                # print("x type", type(x.data))
                
                preds = model(x)
                curr_batch = int(preds.size(0)/SEQ_SIZE)
                
                loss = criterion(preds, GT)
                accuracy += torch.mean(torch.stack([loss])).item()
                # y is [batch, 1] , preds should be [batch, 2]
                # for gru sequences preds can be [batch, seq, 2]
                '''
                if FRAME_LABELS:
                    loss = criterion(preds, y.view(curr_batch * SEQ_SIZE))
                    accuracy += get_accuracy(preds, y.view(curr_batch * SEQ_SIZE))
                else:
                    loss = criterion(preds, y[:, 0])
                    accuracy += get_accuracy(preds, y[:, 0])
                '''
                #print(preds, y)
                '''
                net_loss += loss.data.cpu().numpy()
                '''
                net_loss = accuracy
                
#                print("Phase : {} :: Batch : {} :: Loss : {} :: Accuracy : {}"\
#                          .format(phase, (i+1), net_loss, accuracy))
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if (i+1) == 10000:
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
#        print("Phase : Test :: Epoch : {} :: Loss : {} :: Accuracy : {}"\
#              .format((epoch+1), training_stats[epoch]['test']['loss'],\
#                      training_stats[epoch]['test']['acc']))
        
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
    name = os.path.join(log_dir, 'c3d_gru_ep'+str(epoch)+"_w"+str(win)+"_"+opt+".pt")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, name)
    print("Model saved to disk... {}".format(name))
    
def load_saved_checkpoint(model, optimizer, epoch, opt, win):
    assert os.path.exists(log_dir), "Checkpoint path does not exist."
    name = os.path.join(log_dir, 'c3d_gru_ep'+str(epoch)+"_w"+str(win)+"_"+opt+".pt")
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

def get_1D_preds(preds):
    preds_new = []
    for pred in preds.data.cpu().numpy():
        # print("i preds", pred)
        idx = np.argmax(pred)
        preds_new.append(idx)
    return np.asarray(preds_new)

def get_accuracy(preds, targets):
    preds_new = get_1D_preds(preds)
    tar_new = targets.data.cpu().numpy()
    # print("preds", preds_new[:5])
    # print("targets", tar_new[:5])
    acc = sum(preds_new == tar_new)*1.0
    return acc

# save training stats
def save_stats_dict(stats):
    with open(os.path.join(log_dir, 'stats.pickle'), 'wb') as fr:
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
            

def aggregate_outputs(seq_outputs, seq_stroke_names):
    '''
    Use some aggregation function (mean etc.) to obtain stroke representation.
    Parameters:
    -----------
    seq_outputs: list
    
    '''
    all_strokes, stroke_vecs = [], []
    prev_stroke_name = seq_stroke_names[0]
    stroke_names = [seq_stroke_names[0]]
    for i, curr_stroke_name in enumerate(seq_stroke_names):
        if prev_stroke_name != curr_stroke_name:
            all_strokes.append(stroke_vecs)
            stroke_names.append(seq_stroke_names[i])
            stroke_vecs = []
            
        stroke_vecs.append(seq_outputs[i])
        prev_stroke_name = curr_stroke_name
        
    if len(stroke_vecs) != 0:
        all_strokes.append(stroke_vecs)
        
    # call aggregation function
    all_strokes = [torch.cat(stroke_lst, axis=0) for stroke_lst in all_strokes]
    
    return all_strokes, stroke_names

#def get_output_means(stroke)


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
    st_frms_keys = sorted(list(st_frms_dict.keys()))
    end_frms_keys = sorted(list(end_frms_dict.keys()))
    assert len(st_frms_dict) == len(end_frms_dict), "Invalid stroke pts."
    stroke_vectors, stroke_names = [], []
    
    in_idx = 0
    for i, st_frm in enumerate(st_frms_keys):
        stroke_vectors.append(inputs[in_idx:(in_idx + st_frms_dict[st_frm]), :])
        curr_stroke = vid_path[in_idx]+'_'+str(st_frm)+'_'+str(end_frms_keys[i])
        stroke_names.append(curr_stroke)
        in_idx += st_frms_dict[st_frm]
    
    return stroke_vectors, stroke_names
    

def plot_sequences(stroke_vecs, stroke_names):
    
    stroke_lens = [stroke.size()[0] for stroke in stroke_vecs]
    all_feats = torch.cat(stroke_vecs, axis=0).data.numpy()
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure(2)
    
    pca_applied = clus_utils.apply_PCA(all_feats)
    
    indx = 0
    for i, slen in enumerate(stroke_lens):
        plt.plot(pca_applied[indx:(indx+slen), 0], pca_applied[indx:(indx+slen), 1], \
                             color=colors[i % len(colors)], marker='.', lw=1 )
        indx += slen
    
#    markers = list("12348spphH+xXdD")
#    for color,n_cluster in enumerate(best_tuple):
#            
#        cls_data = data[labels==n_cluster]
#        plt.scatter(cls_data[:,0], cls_data[:,1], c=colors[color%8], marker = markers[color], label="C"+str(color+1))
#    plt.legend(loc='upper right')
#    if plotname=='cluster_pca_ordered.png':
#        plt.xlim((-1,1))
#        plt.ylim((-1,1))
    #plt.savefig(os.path.join(RESULTS_DIR, "bins_"+str(bins)+"_th_"+str(thresh), str(n_clusters)+'_'+plotname))
    plt.show()
    
    

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
    '''
    train_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in train_lst]
    val_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in val_lst]
    test_gt_batsman = [f.rsplit('.',1)[0] +"_gt.txt" for f in test_lst]
    '''
    
    #####################################################################
    
    tr_labs = [os.path.join(LABELS, f) for f in train_lab]
    val_labs = [os.path.join(LABELS, f) for f in val_lab]
    '''
    tr_gt_batsman = [os.path.join(BAT_LABELS, f) for f in train_gt_batsman]
    '''
    sizes = [autoenc_utils.getTotalFramesVid(os.path.join(DATASET, f)) for f in train_lst]
    print("Size : {}".format(sizes))
    
    '''
    bboxes = True
    epsilon = 50
    visualize = False
    n_clusters = 5
    clust_no = 1
    first_frames = 2
    '''
    
    ###########################################################################
    # Create a Dataset
    '''
    data_transforms = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((112,112)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                               std=[0.229, 0.224, 0.225]),])
    '''

    train_transforms = transforms.Compose([videotransforms.RandomCrop(112),
                                           #videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(112)])
    
    # Prepare datasets and dataset loaders (training and validation/testing)
    val_dataset = CricketStrokesDataset(test_lst, DATASET, LABELS, CLASS_IDS, 
                                          frames_per_clip=SEQ_SIZE, train=False, 
                                          transform=test_transforms)
    
    train_dataset = CricketStrokesDataset(train_lst, DATASET, LABELS, CLASS_IDS, 
                                          frames_per_clip=SEQ_SIZE, train=True, 
                                          transform=train_transforms)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loader = {'train': train_loader, 'test': val_loader}

    ###########################################################################
    # Create a model
    #model = autoenc.AutoEncoderRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS)
    model = DPC_RNN(sample_size=128, num_seq=2,
                    seq_len=SEQ_SIZE, pred_step=1 ,network='full3Dresnet18')
    print("Loading model ...")
    #model.load_state_dict(torch.load(model_path))
    
    model = model.to(device)
    '''
    criterion = nn.CrossEntropyLoss()
    '''
    criterion = nn.MSELoss()
    #criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                 lr = 0.001)
#    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), \
#                                 lr = 0.001, momentum=0.9)
    
    # set the scheduler, optimizer and retrain (eg. SGD)
    # Decay LR by a factor of 0.1 every 7 epochs
    step_lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    
    ###########################################################################
    #####################################################################
    # Load checkpoint model and optimizer state
    '''
    s_epoch = 0
    for i in range(N_EPOCHS, 0, -1):    
        if os.path.exists(model_path):
#            model, optimizer, s_epoch = load_saved_checkpoint(model, optimizer, i, \
#                                                              "Adam", SEQ_SIZE)
            break
    '''
        
#    if torch.cuda.device_count() > 1:
#        print("Let's use", torch.cuda.device_count(), "GPUs!")
#        # Parallely run on multiple GPUs using DataParallel
#        model = nn.DataParallel(model)

    #####################################################################
    
    start = time.time()
    s = start
    
#    # Training (finetuning) and validating
    model = train(model, data_loader, optimizer, step_lr_scheduler, \
                  criterion, nEpochs=N_EPOCHS, use_gpu=False)
        
    end = time.time()
    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))
    
    
    ###########################################################################
    # Validate / Evaluate
    model.eval()
    sequence_outputs, seq_stroke_names = [], []
    num_strokes = 0
    extractor = Img2Vec()
    prev_stroke_idx = 0
    
    print("Total Batches : {} :: BATCH_SIZE : {}".format(val_loader.__len__(), BATCH_SIZE))
    for bno, (inputs, vid_path, stroke, _) in enumerate(val_loader):
        print("Batch No : {}".format(bno))
        inputs = extractor.get_vec(inputs)
        # convert to start frames and end frames from tensors to lists
        stroke = [s.tolist() for s in stroke]
        inputs_lst, stroke_names = separate_stroke_tensors(inputs, vid_path, stroke)
        
        # If all the batch samples are from the same stroke, ignore last non-full batch
        if len(inputs_lst) == 1 and inputs_lst[0].size()[0] % SEQ_SIZE == 0:
            input_seq = inputs_lst[0].reshape(-1, SEQ_SIZE, INPUT_SIZE).to(device)
            outputs = model.encoder(input_seq)
            sequence_outputs.append(outputs.squeeze(axis=1).cpu())
            seq_stroke_names.append(stroke_names[0])
        elif len(inputs_lst) > 1: #Ignore batch with multiple strokes
            num_strokes += (len(inputs_lst) - 1)
        
#        for i, input_seq in enumerate(inputs_lst):
#            # Ignoring the last partial batch if does not match, or first partial batch
#            if input_seq.size()[0] % SEQ_SIZE == 0:
#                input_seq = input_seq.reshape(-1, SEQ_SIZE, INPUT_SIZE).to(device)
#                #with torch.set_grad_enabled(phase != 'val'):
#                outputs = model.encoder(input_seq)
#                sequence_outputs.append(outputs.cpu())
#                seq_stroke_names.append(stroke_names[i])
#                #np_outputs= outputs.data.cpu().numpy().flatten()
#            else:
#                num_strokes +=1
            
        if num_strokes == 12:
            break
        
        
    stroke_vecs, stroke_names =  aggregate_outputs(sequence_outputs, seq_stroke_names)
    
    # plot the strokes as sequence of points
    plot_sequences(stroke_vecs, stroke_names)
    
    #clus_utils.plot_clusters(pca_flows, labels, perm_tuples[best_indx], bins, thresh, 'cluster_pca_ordered.png')
    
    
    print(len(stroke_vecs))
    
    print("#Parameters : {}".format(autoenc_utils.count_parameters(model)))
    
    