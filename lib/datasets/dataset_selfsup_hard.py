#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 05:12:51 2021

@author: arpan
"""

from __future__ import with_statement
from __future__ import absolute_import
from __future__ import division
import json

import numpy as np
import torch

#from torchvision.datasets import UCF101
from torchvision import transforms
from io import open

import os
import pickle
#import utils
import datasets.folder as folder
import pandas as pd

#from torchvision.datasets.utils import list_dir
#from torchvision.datasets.video_utils import VideoClips
from utils.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
import random


class StrokeFeaturePairsDataset(VisionDataset):
    """
    Cricket Feature Pairs for Seq2Seq training for strokes dataset.
    """

    def __init__(self, feat_path, videos_list, dataset_path, strokes_dir, class_ids_path, 
                 frames_per_clip, extracted_frames_per_clip=16, step_between_clips=1, 
                 future_step = 1, train=True, frame_rate=None, framewiseTransform=True, 
                 transform=None, _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(StrokeFeaturePairsDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.extracted_frames_per_clip = extracted_frames_per_clip
        self.step_between_clips = step_between_clips
        self.future_step = future_step
        assert os.path.isfile(feat_path), "Path does not exist. {}".format(feat_path)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        
        with open(feat_path, "rb") as fp:
            self.features = pickle.load(fp)
        
        self.class_by_idx_label = self.load_labels_index_file(class_ids_path)
        extensions = ('avi', 'mp4', )
        self.train = train

        classes = sorted(list(self.class_by_idx_label.keys()))
        #print("Classes : {}".format(classes))
        
        class_to_idx = {self.class_by_idx_label[i]: i for i in classes}
        self.samples = folder.make_strokes_dataset(videos_list, dataset_path, 
                                                   class_to_idx, train, extensions, 
                                                   is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        vid_strokes = self.read_stroke_labels(video_list)
        self.video_clips = VideoClips(video_list, vid_strokes, frames_per_clip, step_between_clips)
        # self.video_clips_metadata = video_clips.metadata  # in newer torchvision
        # self.indices = self._select_fold(video_list, annotation_path, fold, train)
        # self.video_clips = video_clips.subset(self.indices)
        self.framewiseTransform = framewiseTransform
        self.transform = transform
        self.pairs = self.generateSequencePairs(self.future_step)
        
    def read_stroke_labels(self, video_list):
        
        vid_strokes = []
        for video in video_list:
            vidname = video.rsplit('/', 1)[-1]
            stroke_file = os.path.join(self.strokes_dir, vidname.rsplit('.', 1)[0]+'.json')
            assert os.path.isfile(stroke_file), "File not found {}".format(stroke_file)
            with open(stroke_file, 'r') as fp:
                strokes = json.load(fp)
            vid_strokes.append(strokes[list(strokes.keys())[0]])
        return vid_strokes

    def load_labels_index_file(self, filename):
        """
        Returns a dictionary of {ID: classname, ...}
        """
        if os.path.isfile(filename):
            df = pd.read_csv(filename, sep=" ", names=["id", "classname"])
            d_by_cols = df.to_dict('list')
            d_by_idx_label = dict(zip(d_by_cols['id'],
                              d_by_cols['classname']))
            return d_by_idx_label
        else:
            return None
    
    @property
    def metadata(self):
        return self.video_clips_metadata

    def __len__(self):
        return len(self.pairs)
    
    def generateSequencePairs(self, future_step = 1):
        '''Generate positive and negative pair samples for self supervised 
        training using contrastive loss
        '''
        import pickle
        vid_clip_idx = [self.video_clips.get_clip_location(i) \
                        for i in range(self.video_clips.num_clips())]
        strokes = [self.video_clips.video_paths[vidx]+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][0])+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][1]) \
                   for (vidx, clidx) in vid_clip_idx]
        pairs, pos_pairs, neg_easy_pairs, neg_hard_pairs = [], [], [], []
        
        # create positive pair samples, take future context as target
        prev = strokes[0]
        for i in range(0, len(strokes), future_step): 
            s = strokes[i]
            if i==0:
                continue
            if s == prev:
                pos_pairs.append([vid_clip_idx[i - future_step], vid_clip_idx[i]])
            else:
                prev = s
        
        nSamps = len(pos_pairs)
        pairs_file = None
        if os.path.isfile("hardpairs_train_F"+str(future_step)+"_"+str(2*nSamps)+".pkl"):
            pairs_file = "hardpairs_train_F"+str(future_step)+"_"+str(2*nSamps)+".pkl"
        elif os.path.isfile("hardpairs_val_F"+str(future_step)+"_"+str(2*nSamps)+".pkl"):
            pairs_file = "hardpairs_val_F"+str(future_step)+"_"+str(2*nSamps)+".pkl"
        elif os.path.isfile("hardpairs_test_F"+str(future_step)+"_"+str(2*nSamps)+".pkl"):
            pairs_file = "hardpairs_test_F"+str(future_step)+"_"+str(2*nSamps)+".pkl"
        
        if pairs_file is not None:
            with open(pairs_file, "rb") as fp:
                pairs = pickle.load(fp)
            return pairs
        
        # create negative pair samples, take clips from other strokes as target
        for i in range(0, len(strokes), future_step):
            neg_easy_pairs.append(self.sample_easy_neg(vid_clip_idx, strokes, i, 
                                                       self.frames_per_clip, 
                                                       self.extracted_frames_per_clip))
            neg_hard_pairs.append(self.sample_hard_neg(vid_clip_idx, strokes, i, 
                                                       self.frames_per_clip, 
                                                       self.extracted_frames_per_clip))
#            print("{} / {}".format(i, len(strokes)))
        neg_easy_pairs.extend(neg_hard_pairs)
        neg_pairs = random.sample(neg_easy_pairs, len(pos_pairs))
        for i in range(min(len(pos_pairs), len(neg_pairs))):
            pairs.append((pos_pairs[i], 1))
            pairs.append((neg_pairs[i], 0))
        # write the pairs list to disk as it takes lot of time.
        if self.train:
            with open("hardpairs_train_F"+str(future_step)+"_"+str(2*nSamps)+".pkl", "wb") as fp:
                pickle.dump(pairs, fp)
        else:
            with open("hardpairs_val_F"+str(future_step)+"_"+str(2*nSamps)+".pkl", "wb") as fp:
                pickle.dump(pairs, fp)
        return pairs
    
    def sample_easy_neg(self, vid_clip_idx, strokes, ref_clip_idx, seq_size, ex_fpc):
        '''Get clip from other stroke and > 2*seq distance from ref_clip
        '''
        neg_clip = None
        ref_clip = vid_clip_idx[ref_clip_idx]
        potential_clips_bool = [abs(ref_clip[1]-c[1]) > 2*seq_size for c in vid_clip_idx]
        # sampling to be done from True samples
        potential_clips = [i for i, b in enumerate(potential_clips_bool) if b]        
        neg_clip = random.sample(potential_clips, 1)[0]
        return [vid_clip_idx[ref_clip_idx], vid_clip_idx[neg_clip]]
        
    def sample_hard_neg(self, vid_clip_idx, strokes, ref_clip_idx, seq_size, ex_fpc):
        '''Get clip from same stroke but > seq and <=2*seq distance from ref_clip
        '''
        neg_clip = None
        ref_clip = vid_clip_idx[ref_clip_idx]
        vid_clips_valid = list(filter(lambda x: x[0] == ref_clip[0], vid_clip_idx))
        potential_clips_bool = [(abs(ref_clip[1]-c[1]) > self.future_step and 
                                abs(ref_clip[1]-c[1]) <= 2*seq_size) 
                                for c in vid_clips_valid]
        # sampling to be done from True samples
        potential_clips = [i for i, b in enumerate(potential_clips_bool) if b]        
        neg_clip = random.sample(potential_clips, 1)[0]        
        return [vid_clip_idx[ref_clip_idx], vid_clips_valid[neg_clip]]

        
    def __getitem__(self, idx):
        
        [(video1_idx, clip1_idx), (video2_idx, clip2_idx)], label = self.pairs[idx]
#        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video1_path = self.video_clips.video_paths[video1_idx]
        video2_path = self.video_clips.video_paths[video2_idx]
        stroke1_tuple = self.video_clips.stroke_tuples[video1_idx]
        stroke2_tuple = self.video_clips.stroke_tuples[video2_idx]
        clip1_pts = self.video_clips.clips[video1_idx][clip1_idx]
        clip2_pts = self.video_clips.clips[video2_idx][clip2_idx]
        start1_pts = clip1_pts[0].item()
        start2_pts = clip2_pts[0].item()
#        end_pts = clip_pts[-1].item()
        
        # form feature key 
        key1 = video1_path.rsplit('/', 1)[1].rsplit('.', 1)[0]+'_'+\
                str(stroke1_tuple[0])+'_'+str(stroke1_tuple[1])
        key2 = video2_path.rsplit('/', 1)[1].rsplit('.', 1)[0]+'_'+\
                str(stroke2_tuple[0])+'_'+str(stroke2_tuple[1])
                
        vid1_feats = self.features[key1]
        vid2_feats = self.features[key2]
        
        start1_idx = start1_pts - stroke1_tuple[0]
        start2_idx = start2_pts - stroke2_tuple[0]
        seq_len = self.frames_per_clip - self.extracted_frames_per_clip + 1
        
        # retrieve the sequence of vectors from the stroke sequences
        sequence1 = vid1_feats[start1_idx:(start1_idx+seq_len), :]
        sequence2 = vid2_feats[start2_idx:(start2_idx+seq_len), :]
        
#        if self.train:
#            for s in self.samples:
#                if video_path == s[0]:
#                    label = s[1]
#                    break
#            #label = self.samples[video_idx][1]
#            label = self.classes.index(label)
#        else:
#            label = -1

        return sequence1, sequence2, video1_path, stroke1_tuple, video2_path, \
                stroke2_tuple, label