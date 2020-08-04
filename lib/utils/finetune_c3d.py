#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 07:45:40 2020

@author: arpan

@Description: File for finetuning ResNet 3D on training set features of strokes
"""

import _init_paths

import os
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from utils.resnet_feature_extracter import Clip2Vec
from utils import autoenc_utils
import datasets.videotransforms as videotransforms
from datasets.dataset import CricketStrokesDataset
import copy
import time
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
    if min(labs_values) == 1:
        labs_values = [l-1 for l in labs_values]
        labs_keys = [k.replace('.avi', '') for k in labs_keys]
    return labs_keys, labs_values

def get_batch_labels(vid_path, stroke, labs_keys, labs_values):
    
    labels = []
    for i, vid_name in enumerate(vid_path):
        
        vid_name = vid_name.rsplit('/', 1)[1]
        if '.avi' in vid_name:
            vid_name = vid_name.replace('.avi', '')
        elif '.mp4' in vid_name:
            vid_name = vid_name.replace('.mp4', '')
        stroke_name = vid_name+"_"+str(stroke[0][i].item())+"_"+str(stroke[1][i].item())
        assert stroke_name in labs_keys, "Key does not match : {}".format(stroke_name)
        labels.append(labs_values[labs_keys.index(stroke_name)])
        
    return torch.tensor(labels)
        
def save_model_checkpoint(base_name, model, ep, opt):
    """
    TODO: save the optimizer state with epoch no. along with the weights
    """
    # Save only the model params
    name = os.path.join(base_name, "3dresnet18_ep"+str(ep)+"_"+opt+".pt")
#    if use_gpu and torch.cuda.device_count() > 1:
#        model = model.module    # good idea to unwrap from DataParallel and save

    torch.save(model.state_dict(), name)
    print("Model saved to disk... {}".format(name))
    

def train_model(model, dataloaders, criterion, optimizer, scheduler, 
                labs_keys, labs_values, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for bno, (inputs, vid_path, stroke, labels) in enumerate(dataloaders[phase]):
                
                labels = get_batch_labels(vid_path, stroke, labs_keys, labs_values)
                # Extract spatio-temporal features from clip using 3D ResNet (For SL >= 16)
                inputs = inputs.permute(0, 2, 1, 3, 4).float()
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
#                print("Batch No : {} / {}".format(bno, len(dataloaders[phase])))
#                if (bno+1) % 10 == 0:
#                    break
                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, \
          time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def finetune_3DCNN(DATASET, LABELS, CLASS_IDS, BATCH_SIZE, ANNOTATION_FILE, 
                   SEQ_SIZE=16, STEP=16, nstrokes=-1, N_EPOCHS=25, base_name=""):
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

    train_dataset = CricketStrokesDataset(train_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, 
                                         step_between_clips=STEP, train=True, 
                                         framewiseTransform=False,
                                         transform=clip_transform)
    val_dataset = CricketStrokesDataset(val_lst, DATASET, LABELS, CLASS_IDS, 
                                         frames_per_clip=SEQ_SIZE, 
                                         step_between_clips=STEP, train=False, 
                                         framewiseTransform=False,
                                         transform=clip_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    data_loaders = {"train": train_loader, "test": val_loader}

    ###########################################################################
    
    labs_keys, labs_values = get_cluster_labels(ANNOTATION_FILE)
    
    num_classes = len(list(set(labs_values)))
    
    ###########################################################################    
    # load model and set loss function
    model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
    
    for ft in model.parameters():
        ft.requires_grad = False
    
    inp_feat_size = model.fc.in_features
    model.fc = nn.Linear(inp_feat_size, num_classes)
    model = model.to(device)
    
    # load checkpoint:
    if os.path.isfile(os.path.join(base_name, "3dresnet18_ep"+str(N_EPOCHS)+"_Adam.pt")):
        model.load_state_dict(torch.load(os.path.join(base_name, "3dresnet18_ep"+str(N_EPOCHS)+"_Adam.pt")))
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()
    
    # Layers to finetune. Last layer should be displayed
    params_to_update = model.parameters()
    print("Params to learn:")
    
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    
    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(params_to_update, lr=0.001)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = StepLR(optimizer_ft, step_size=8, gamma=0.1)
    
#    # Observe that all parameters are being optimized
#    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
#    ###########################################################################
#    # Training the model    
#    
#    start = time.time()
#    
#    model_ft = train_model(model, data_loaders, criterion, optimizer_ft, 
#                           exp_lr_scheduler, labs_keys, labs_values,
#                           num_epochs=N_EPOCHS)
#        
#    end = time.time()
#    
#    # save the best performing model
#    save_model_checkpoint(base_name, model_ft, N_EPOCHS, "Adam")
#    
#    print("Total Execution time for {} epoch : {}".format(N_EPOCHS, (end-start)))
    
    ###########################################################################
    # Validate / Evaluate
    stroke_names = []
    trajectories, stroke_traj = [], []
    num_strokes = 0
    model = model.eval()
#    extractor = Clip2Vec()
    #INPUT_SIZE = extractor.layer_output_size
    prev_stroke = None
    
    print("Total Batches : {} :: BATCH_SIZE : {}".format(data_loaders['test'].__len__(), BATCH_SIZE))
    assert SEQ_SIZE>=16, "SEQ_SIZE should be >= 16"
    for bno, (inputs, vid_path, stroke, _) in enumerate(data_loaders['test']):
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


if __name__ == '__main__':
    # Local Paths
    LABELS = "/home/arpan/VisionWorkspace/Cricket/scripts/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
    DATASET = "/home/arpan/VisionWorkspace/VideoData/sample_cricket/ICC WT20"
    CLASS_IDS = "/home/arpan/VisionWorkspace/Cricket/cluster_strokes/configs/Class Index_Strokes.txt"
    # Server Paths
    if os.path.exists("/opt/datasets/cricket/ICC_WT20"):
        LABELS = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
        DATASET = "/opt/datasets/cricket/ICC_WT20"
    
    ANNOTATION_FILE = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/shots_classes.txt"

    SEQ_SIZE = 16
    STEP = 8
    INPUT_SIZE = 2048
    HIDDEN_SIZE = 32#64#1024
    NUM_LAYERS = 2
    BATCH_SIZE = 32
    N_EPOCHS = 25
    base_path = "/home/arpan/VisionWorkspace/Cricket/CricketStrokeLocalizationBOVW/logs"
    
    trajectories, stroke_names = finetune_3DCNN(DATASET, LABELS, CLASS_IDS, 
                                                BATCH_SIZE, ANNOTATION_FILE,
                                                SEQ_SIZE, STEP, nstrokes=-1, 
                                                N_EPOCHS=N_EPOCHS, base_name=base_path)