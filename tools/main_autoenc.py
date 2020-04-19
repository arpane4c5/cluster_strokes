#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 06:05:10 2020

@author: arpan

@Description: main file for training Auto-Encoder on strokes
"""

import _init_paths

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from utils.resnet_feature_extracter import Img2Vec
import numpy as np
import os
import time
import copy
import matplotlib.pyplot as plt
from PIL import Image
import json
import cv2

from models import autoenc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sequence_length = 6
input_size = 2048
hidden_size = 32#64#1024
num_layers = 2
batch_size = 6 # set to the number of images of a seqence # 36
############
val_frame_dump_path = 'batsman-frames/val/batsman/'
anotations_path = "val_anotations/"  ##batsman frames in shots
shots_anotations_path = "/home/arpan/VisionWorkspace/shot_detection/supporting_files/sample_set_labels/sample_labels_shots/ICC WT20"
model_path = "checkpoints/28_3_batsman_lstm_autoencoder_model_wholeDataset.pt"
val = "val/"
cluster_result_path = "cluster-result/"

extractor = Img2Vec()


data_dir = './val/'

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])







## define model here
model = autoenc.AutoEncoderRNN(input_size, hidden_size, num_layers)
print("loading model")
model.load_state_dict(torch.load('checkpoints/28_3_batsman_lstm_autoencoder_model_wholeDataset.pt'))
print("loaded")
model = model.to(device)

### get shots list for each video from the anotations
### it requires annotation file and searches videos corresponding folder
### 
def get_shots_list_list():
  # shots_label_files = os.listdir(shots_anotations_path)
  # videos_files = [item.split(".json")[0]+".avi" for item in shots_label_files]
  # batsman_label_files = [item.split(".json")[0]+"_gt.txt" for item in shots_label_files]

  videos_files = os.listdir(val)
  shots_label_files = [video.split(".avi")[0]+".json" for video in videos_files]
  batsman_label_files = [item.split(".json")[0]+"_gt.txt" for item in shots_label_files]

  ## list of videos shots
  shots_list_list = []
  for shots_label_file, video_file, batsman_label_file in zip(shots_label_files, videos_files, batsman_label_files):
    print("*********************************************************************")
    print("video file: ", video_file)

    shots_label_file_path = os.path.join(shots_anotations_path, shots_label_file)
    batsman_label_file_path = os.path.join(anotations_path, batsman_label_file)

    shots_list = []
    shot_list = []
    shot_count = 0

    with open(shots_label_file_path, "r") as fr:
      shot_dict = json.load(fr)
      key = list(shot_dict.keys())[0]
      print("key: ",key)
      ft = 0
      for item in shot_dict[key]:
        f1, f2 = item
        shot_list.extend([-1]*(f1-ft))
        shot_list.extend([shot_count]*(f2-f1+1))
        ft = f2+1
        shot_count +=1
    
    with open(batsman_label_file_path, "r") as fr:
      batsman_frames = [int(line.split(',')[0]) for line in fr.readlines()]
      batsman_frames = np.asarray(batsman_frames)
    print("batsman idx: ", type(batsman_frames),int(shot_list[-1]))
    print("suim: ", sum(shot_list))
    shot_list_mask = np.asarray([0]*len(shot_list))
    shot_list = np.asarray(shot_list)
    shot_list_mask[batsman_frames] = 1
    shot_list[np.where(shot_list_mask==0)] = -1
    print("suim: ", sum(shot_list))
    print("shot_list: ", shot_list)

    for i in range(shot_count):
      single_shot = list(np.where(shot_list==i)[0].tolist())
      print("single_shot:", single_shot)
      shots_list.append(single_shot)
    shots_list_list.append(shots_list)
  return videos_files, shots_list_list


videos_list, shots_list_list = get_shots_list_list()


################################################################################################


################################################################################################
### sampling applied on frames of each shot
### using minimum shot-frames count in whole videos
################################################################################################
def shotFrames_count(shot_list_list):
  ## each shot-frame count after sampling = min_frame_count
  ## finding min_frame_count
  min_frame_count = 999999999999999999
  for video in shot_list_list:
    for shot in video:
      shot_len = len(shot)
      if shot_len < min_frame_count:
        min_frame_count = shot_len
  return min_frame_count

# % ls $val_frame_dump_path
sum_ =0
for L in shots_list_list:
  for l in L:
    print (len(l))
    sum_+=len(l)
print(sum_)


## get predictions for each video in validation
## get to each video and use sliding window on each shot for commulative prediction of each shot
## then predict for each shot in a single video and print them separately
def get_val_inference():
    flag=False
    since = time.time()
    phase = 'val'
    if phase == 'val':
        # scheduler.step()
        model.eval()
    else:
        model.train()
    ##################################
    counter = 1
    ensembled_shot_vector_video_list = []
    #########
    ##min frame count for sampling
    min_frm_cnt = shotFrames_count(shots_list_list)
    print("min_frm_cnt: ", min_frm_cnt)
    ########

    for video, shots_list in zip(videos_list , shots_list_list):
      ensembled_shot_vector_video = []
      print("Video: ", video, "encoder-results-val/{}/".format(video.split('.avi')[0]))
      # os.mkdir("encoder-results-val/{}/".format(video.split('.avi')[0]))
      for s, shot in enumerate(shots_list):
        
        ##########
        ## frame sampling
        shot_len = len(shot)
        print("before sampling frame count: ", shot_len, end=" ")
        while(shot_len != min_frm_cnt ):
          smplg_fctr = int(shot_len/min_frm_cnt)
          ### removing all frames from shot ending
          if(smplg_fctr < 2):
            residue_len = shot_len - min_frm_cnt
            shot = shot[:-residue_len]
          shot = shot[::smplg_fctr]
          shot_len = len(shot)
          print("intermediate sampling frame count: ", shot_len,end=" ")
        print("after sampling frame count: ", shot_len)
        ##########

        ensembled_shot_vector = []
        print("counter: ", counter)
        print("shot idx: ", s, " shots size: ", len(shot))
        print("shots\n")
        for i in range(0, len(shot)-batch_size):
          imgs = []
          # print("counter: ", counter)
          for ii in range(0, batch_size):
            path = os.path.join('batsman-frames/{}/batsman/{}_{:012}.png'.format(phase,video.split(".")[0],shot[i+ii]))
            try:
              img = data_transforms(Image.open(path))
            except Exception as e:
              print("&&&&&&&&&&&&&&&&&&&&error: ", e)
            imgs.append(img)

          #   print(shot[i+ii-1]," ",end='')
          # print("\n")
          counter+=1
          inputs = torch.stack(imgs)
          inputs = extractor.get_vec(inputs)
          # print("inputs shape: ", inputs.size())
          inputs = inputs.reshape(-1, sequence_length, input_size).to(device)
          with torch.set_grad_enabled(phase != 'val'):
            outputs = model.encoder(inputs)
            np_outputs= outputs.data.cpu().numpy().flatten()
            ensembled_shot_vector.extend(list(np_outputs))
            # print("list(np_outputs) ", np.shape(list(np_outputs)))
            # np.save("encoder-results-val/{}/{}_{}".format(video.split('.avi')[0], s, i), np_outputs)
            # print("outputs", outputs.size())
            # sys.exit()
            # inv_idx = torch.arange(sequence_length - 1, -1, -1).long()
        print("final shot vector: ", np.shape(ensembled_shot_vector))
        ensembled_shot_vector_video.append(ensembled_shot_vector)
        # if(flag==True): break
        # if(flag==False): flag=True
        counter+=batch_size
      print("ensembled_shot_vector_video: ", np.shape(ensembled_shot_vector_video))
      ensembled_shot_vector_video_list.append(ensembled_shot_vector_video)
      print("ensembled_shot_vector_video_list: ", np.shape(ensembled_shot_vector_video_list))
      print("total shots inferred: ", counter)
    ##################################
    print()

    time_elapsed = time.time() - since
    print('validation complete in {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return ensembled_shot_vector_video_list


feature_list = get_val_inference()


### reshape all shots vectors to same width

max_ = -10000000
for video in feature_list:
  print("video: ", np.shape(video))
  for shots in video:
    print("shots: ", np.shape(shots))
    if len(shots) > max_ : max_ = len(shots)
print(max_)

new_feature_list = []
for video in feature_list:
  new_video = []
  for shot in video:
    new_shot = []
    shot_len = len(shot)
    print("initial len: ",shot_len)
    residue = max_ - shot_len
    # if(residue==0):continue
    residue_arr = shot[shot_len-sequence_length:]
    # print(residue_arr)
    shot.extend(residue_arr*int(residue/sequence_length))
    shot_len = len(shot)
    print("2nd len: ",shot_len)
    residue = max_ - shot_len
    residue_arr = shot[shot_len-residue:]
    shot.extend(residue_arr)
    # print(shot)
    new_shot = shot
    print(new_shot)
    print("new_shot len: ",len(new_shot))
    new_video.append(new_shot)
  new_feature_list.append(new_video)

print("new feature shape: ", np.shape(new_feature_list))



from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#######
# now only for one video 
#######
print(np.shape(feature_list))
feature_array1 = np.asarray(feature_list[0])#.reshape(np.shape(feature_array)[0], -1)
feature_array2 = np.asarray(feature_list[1])#.reshape(np.shape(feature_array)[0], -1)
# print("feature_array: ", np.shape(feature_array))

feature_array = np.concatenate([feature_array1,feature_array2])
print("feature_array: ", np.shape(feature_array))

X_embedded = TSNE(n_components=2).fit_transform(feature_array)
print(X_embedded.shape)

x = np.array(X_embedded[:,0])
y = np.array(X_embedded[:,1])

plt.scatter(x, y)
plt.show()