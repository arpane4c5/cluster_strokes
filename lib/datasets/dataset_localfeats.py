#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 19:18:29 2021

@author: arpan
"""

from __future__ import with_statement
from __future__ import absolute_import
from __future__ import division
import json

import numpy as np
import torch

import glob
import os
import pickle
#import cv2
#import utils
import datasets.folder as folder
import pandas as pd

from torchvision.datasets.vision import VisionDataset

class LocalFeatureSequencesDataset(VisionDataset):
    """
    `Cricket Feature Sequence Dataset for strokes dataset.

    Args:
        root (string): Root directory of the Cricket Dataset.
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int, optional): number of frames between each clip.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        framewiseTransform (bool, optional): If the transform has to be applied
            to each frame separately and resulting frames are to be stacked.
        transform (callable, optional): A function/transform that takes in a HxWxC video
            and returns a transformed version (CxHxW) for a frame. Additional dimension
            for a video level transform

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames (without transform)
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, feat_path, videos_list, lab_keys, lab_values, dataset_path, 
                 strokes_dir, frames_per_clip, step_between_clips=1, 
                 train=True, frame_rate=None, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(LocalFeatureSequencesDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.lab_keys = lab_keys
        self.lab_values = lab_values
        assert os.path.isfile(feat_path), "Path does not exist. {}".format(feat_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        # read the labels file        
        with open(feat_path, "rb") as fp:
            self.features = pickle.load(fp)
        
#        extensions = ('avi', 'mp4', )
        self.train = train

        videos_path = [os.path.join(dataset_path, x) for x in videos_list]
        
        vid_strokes = self.read_stroke_labels(videos_path)
        self.stroke_ids = self.generate_stroke_ids(vid_strokes, videos_list)
        stroke_present_bool = [k in self.features.keys() for k in self.stroke_ids]
        assert False not in stroke_present_bool, "Stroke not in feature keys"
        self.ft_lens = [self.features[k].shape[0] for k in self.stroke_ids]
        self.stroke_tuples = self.generate_sequences(self.features, self.stroke_ids, 
                                                      frames_per_clip, step_between_clips)
        self.stroke_nsamples = [len(tups) for tups in self.stroke_tuples]
        self.sample_tuples = []
        self.tuple_keys = []
        for i, tups in enumerate(self.stroke_tuples):
            self.sample_tuples.extend(tups)
            self.tuple_keys.extend([self.stroke_ids[i]] * len(tups))
    
    def read_stroke_labels(self, video_list):
        '''Read the stroke tuples from the labels json files for each video. 
        '''
        vid_strokes = []
        for video in video_list:
            vidname = video.rsplit('/', 1)[-1]
            stroke_file = os.path.join(self.strokes_dir, vidname.rsplit('.', 1)[0]+'.json')
            assert os.path.isfile(stroke_file), "File not found {}".format(stroke_file)
            with open(stroke_file, 'r') as fp:
                strokes = json.load(fp)
            vid_strokes.append(strokes[list(strokes.keys())[0]])
        return vid_strokes
    
    def generate_stroke_ids(self, vid_strokes, videos_list):
        '''Generate the stroke_ids from videos_list and corresponding stroke tuples.
        These are the set of keys for the partition features dictionary.
        '''
        stroke_ids = []
        for i, vidfile in enumerate(videos_list):
            for st, end in vid_strokes[i]:
                v = vidfile.rsplit('.', 1)[0]
                stroke_ids.append(v+'_'+str(st)+'_'+str(end))
        return stroke_ids
        
    def generate_sequences(self, feats, ids, seqlen, step):
        
        seqs = []
        for i, key in enumerate(ids):
            ft_len = self.ft_lens[i]
            assert ft_len >= seqlen, "Stroke length {} < seqlen {}".format(ft_len, seqlen)
            st_seqs = list(range(0, ft_len - seqlen + 1, step))
            end_seqs = list(range(0 + seqlen, ft_len + 1, step))
            seqs.append(list(zip(st_seqs, end_seqs)))
            
        return seqs
    
    def __len__(self):
        return len(self.sample_tuples)

    def __getitem__(self, idx):
        
        key = self.tuple_keys[idx]
        st_idx, end_idx = self.sample_tuples[idx]
        
        vid_feats = self.features[key][st_idx:end_idx]
        
        label = self.lab_values[self.lab_keys.index(key)]

        return vid_feats, key, st_idx, end_idx, label
    
    