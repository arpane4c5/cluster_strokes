#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 4 10:39:42 2021

@author: arpan

@Description: Dataset Classes for generating positive and negative samples 
for self-supervised tasks.

"""
from __future__ import with_statement
from __future__ import absolute_import
from __future__ import division
import json

import numpy as np
import torch
from torchnet.dataset import BatchDataset
from torchnet.dataset.dataset import Dataset

#from torchvision.datasets import UCF101
from torchvision import transforms
from io import open

import glob
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


class CricketStrokeClipsDataset(VisionDataset):
    """
    `Cricket Dataset for strokes dataset.


    Args:
        root (string): Root directory of the Cricket Dataset.
        class_ids_path (str): path to the class IDs file
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

    def __init__(self, videos_list, dataset_path, strokes_dir, class_ids_path, 
                 frames_per_clip, extracted_frames_per_clip=16, step_between_clips=1,
                 future_step=1, train=True, 
                 frame_rate=None, framewiseTransform=True, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(CricketStrokeClipsDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.extracted_frames_per_clip = extracted_frames_per_clip
        self.step_between_clips = step_between_clips
        #assert os.path.exists(root), "Path does not exist. {}".format(root)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        self.future_step = future_step
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
        vid_clip_idx = [self.video_clips.get_clip_location(i) \
                        for i in range(self.video_clips.num_clips())]
        strokes = [self.video_clips.video_paths[vidx]+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][0])+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][1]) \
                   for (vidx, clidx) in vid_clip_idx]
        pairs, pos_pairs, neg_pairs = [], [], []
        # create positive pair samples, take future context as target
        prev = strokes[0]
        for i in range(0, len(strokes), future_step): #  enumerate(strokes):
            s = strokes[i]
            if i==0:
                continue
            if s == prev:
                pos_pairs.append([vid_clip_idx[i-1], vid_clip_idx[i]])
            else:
                prev = s
#        neg_idx_all = random.sample(list(range(len(strokes))), len(strokes))
        # create negative pair samples, take clips from other strokes as target
        for i in range(0, len(strokes), future_step):
            neg_pairs.append(self.sample_easy_neg(vid_clip_idx, strokes, i, 
                                                  self.extracted_frames_per_clip))
        for i in range(len(pos_pairs)):
            pairs.append((pos_pairs[i], 1))
            pairs.append((neg_pairs[i], 0))
            
        return pairs
    
    def sample_easy_neg(self, vid_clip_idx, strokes, ref_clip_idx, seq):
        '''Get clip from other stroke and more than seq distance apart from ref_clip
        '''
        neg_clip = None            
        s = strokes[ref_clip_idx]
        strokes_idx = list(range(len(strokes)))
        
        while neg_clip is None:
            random_clip_idx = random.sample(strokes_idx, 1)[0]
            if strokes[random_clip_idx] != s:
                neg_clip = random_clip_idx
        
        return [vid_clip_idx[ref_clip_idx], vid_clip_idx[neg_clip]]

    def __getitem__(self, idx):
        [(video1_idx, clip1_idx), (video2_idx, clip2_idx)], label = self.pairs[idx]
        # reverse map (video_idx, clip_idx) to the global sample idx
        if video1_idx != 0:
            clip1_idx = self.video_clips.cumulative_sizes[video1_idx - 1] + clip1_idx
        if video2_idx != 0:
            clip2_idx = self.video_clips.cumulative_sizes[video2_idx - 1] + clip2_idx
        video1, vid_path1, stroke1, audio1, info1, video_idx1 = self.video_clips.get_clip(clip1_idx)
        video2, vid_path2, stroke2, audio2, info2, video_idx2 = self.video_clips.get_clip(clip2_idx)
        
        if self.transform is not None:
            if self.framewiseTransform:
                if isinstance(self.transform, transforms.Compose):
                    # transform frame-wise (takes input as HxWxC)
                    video1 = torch.stack([self.transform(i) for i in video1])
                    video2 = torch.stack([self.transform(i) for i in video2])
            else:   # clip level transform (takes input as TxHxWxC)
                video1 = self.transform(video1)
                video2 = self.transform(video2)
            
        if isinstance(video1, list):
            video1, video2 = torch.stack(video1), torch.stack(video2)
            
#        if self.frames_per_clip == 1:       # for single frame sequences
#            video = np.squeeze(video, axis=0)
        
        return video1, video2, vid_path1, stroke1, vid_path2, stroke2, label


class StrokeFeatureSequencesDataset(VisionDataset):
    """
    Taking corresponding sequences from two different features for the same stroke.

    Args:
        root (string): Root directory of the Cricket Dataset.
        class_ids_path (str): path to the class IDs file
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

    def __init__(self, feat_path, feat_path2, videos_list, dataset_path, strokes_dir, 
                 class_ids_path, frames_per_clip, extracted_frames_per_clip=16, 
                 step_between_clips=1, train=True, frame_rate=None, framewiseTransform=True, 
                 transform=None, _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(StrokeFeatureSequencesDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.extracted_frames_per_clip = extracted_frames_per_clip
        self.step_between_clips = step_between_clips
        assert os.path.isfile(feat_path), "Path does not exist. {}".format(feat_path)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        
        with open(feat_path, "rb") as fp:
            self.features = pickle.load(fp)
        with open(feat_path2, "rb") as fp:
            self.features2 = pickle.load(fp)
        
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
        return self.video_clips.num_clips()        

    def __getitem__(self, idx):
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        stroke_tuple = self.video_clips.stroke_tuples[video_idx]
        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        start_pts = clip_pts[0].item()
#        end_pts = clip_pts[-1].item()
        
        # form feature key 
        key = video_path.rsplit('/', 1)[1].rsplit('.', 1)[0]+'_'+\
                str(stroke_tuple[0])+'_'+str(stroke_tuple[1])
                
        vid_feats = self.features[key]
        vid_feats2 = self.features2[key]
        
        seq_start_idx = start_pts - stroke_tuple[0]
        seq_len = self.frames_per_clip - self.extracted_frames_per_clip + 1
        
        # retrieve the sequence of vectors from the stroke sequences
        sequence = vid_feats[seq_start_idx:(seq_start_idx+seq_len), :]
        sequence2 = vid_feats2[seq_start_idx:(seq_start_idx+seq_len), :]
        
        if self.train:
            for s in self.samples:
                if video_path == s[0]:
                    label = s[1]
                    break
            #label = self.samples[video_idx][1]
            label = self.classes.index(label)
        else:
            label = -1

        return sequence, video_path, stroke_tuple, label, sequence2


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
        vid_clip_idx = [self.video_clips.get_clip_location(i) \
                        for i in range(self.video_clips.num_clips())]
        strokes = [self.video_clips.video_paths[vidx]+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][0])+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][1]) \
                   for (vidx, clidx) in vid_clip_idx]
        pairs, pos_pairs, neg_pairs = [], [], []
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
#        neg_idx_all = random.sample(list(range(len(strokes))), len(strokes))
        # create negative pair samples, take clips from other strokes as target
        for i in range(0, len(strokes), future_step):
            neg_pairs.append(self.sample_easy_neg(vid_clip_idx, strokes, i, 
                                                  self.frames_per_clip, 
                                                  self.extracted_frames_per_clip))
        for i in range(min(len(pos_pairs), len(neg_pairs))):
            pairs.append((pos_pairs[i], 1))
            pairs.append((neg_pairs[i], 0))
            
        return pairs
    
    def sample_easy_neg(self, vid_clip_idx, strokes, ref_clip_idx, seq_size, ex_fpc):
        '''Get clip from other stroke and more than seq distance apart from ref_clip
        '''
        neg_clip = None
        ref_clip = vid_clip_idx[ref_clip_idx]
        potential_clips_bool = [abs(ref_clip[1]-c[1]) > 2*seq_size for c in vid_clip_idx]
        # sampling to be done from True samples
        potential_clips = [i for i, b in enumerate(potential_clips_bool) if b]
        
        neg_clip = random.sample(potential_clips, 1)[0]
        
        #######################################################
#        s = strokes[ref_clip_idx]
#        strokes_idx = list(range(len(strokes)))
#        
#        while neg_clip is None:
#            random_clip_idx = random.sample(strokes_idx, 1)[0]
#            if strokes[random_clip_idx] != s:
#                neg_clip = random_clip_idx
        
        #######################################################
#        def get_negative_sample(s):
#            random_clip_idx = random.sample(strokes_idx, 1)
#            if strokes[random_clip_idx] != s:
#                return random_clip_idx
#            else:
#                return get_negative_sample(s)
#        
#        neg_clip = get_negative_sample(s)
        return [vid_clip_idx[ref_clip_idx], vid_clip_idx[neg_clip]]
        
        
#        if ref_clip_idx - seq < 0:
#        random.randrange(0, ref_clip_idx-seq) vid_clip_idx
        
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
    
class StrokeMultiFeaturePairsDataset(VisionDataset):
    """
    Cricket Feature Pairs for Seq2Seq training for strokes dataset.
    """

    def __init__(self, feat_path, videos_list, dataset_path, strokes_dir, class_ids_path, 
                 frames_per_clip, extracted_frames_per_clip=16, step_between_clips=1, 
                 train=True, frame_rate=None, framewiseTransform=True, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(StrokeMultiFeaturePairsDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.extracted_frames_per_clip = extracted_frames_per_clip
        self.step_between_clips = step_between_clips
        for ft_path in feat_path:
            assert os.path.isfile(ft_path), "Path does not exist. {}".format(ft_path)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        
        self.features = []
        for ft_path in feat_path:
            with open(ft_path, "rb") as fp:
                self.features.append(pickle.load(fp))
        
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
        self.pairs = self.generateSequencePairs()
        
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
    
    def generateSequencePairs(self):
        
        vid_clip_idx = [self.video_clips.get_clip_location(i) \
                        for i in range(self.video_clips.num_clips())]
        strokes = [self.video_clips.video_paths[vidx]+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][0])+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][1]) \
                   for (vidx, clidx) in vid_clip_idx]
        pairs = []
        prev = strokes[0]
        for i, s in enumerate(strokes):
            if i==0:
                continue
            if s == prev:
                pairs.append([vid_clip_idx[i-1], vid_clip_idx[i]])
            else:
                prev = s                
        return pairs
        
    def __getitem__(self, idx):
        
        (video_idx, clip_idx), (_, tar_clip_idx) = self.pairs[idx]
#        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        video_path = self.video_clips.video_paths[video_idx]
        stroke_tuple = self.video_clips.stroke_tuples[video_idx]
        src_clip_pts = self.video_clips.clips[video_idx][clip_idx]
        tar_clip_pts = self.video_clips.clips[video_idx][tar_clip_idx]
        src_start_pts = src_clip_pts[0].item()
        tar_start_pts = tar_clip_pts[0].item()
#        end_pts = clip_pts[-1].item()
        
        # form feature key 
        key = video_path.rsplit('/', 1)[1].rsplit('.', 1)[0]+'_'+\
                str(stroke_tuple[0])+'_'+str(stroke_tuple[1])
                
        vid_feats = []
        for feats_dict in self.features:
            vid_feats.append(feats_dict[key])
        
        src_start_idx = src_start_pts - stroke_tuple[0]
        tar_start_idx = tar_start_pts - stroke_tuple[0]
        seq_len = self.frames_per_clip - self.extracted_frames_per_clip + 1
        
        # retrieve the sequence of vectors from the stroke sequences
        src_seqs, tar_seqs = [], []
        for vid_ft in vid_feats:
            src_seqs.append(vid_ft[src_start_idx:(src_start_idx+seq_len), :])
            tar_seqs.append(vid_ft[tar_start_idx:(tar_start_idx+seq_len), :])
            
        nrows = sum([s.shape[0] for s in src_seqs])
        ncols = sum([s.shape[1] for s in src_seqs])
        src_sequences = np.zeros((nrows, ncols))
        tar_sequences = np.zeros((nrows, ncols))
        for i, src_seq in enumerate(src_seqs):
            rows = np.array(list(range(i, nrows, len(vid_feats))))
            cols = np.where(src_seq == 1)[1]
            cols = cols + (i * src_seq.shape[1])
            src_sequences[rows, cols] = 1
        for i, tar_seq in enumerate(tar_seqs):
            rows = np.array(list(range(i, nrows, len(vid_feats))))
            cols = np.where(tar_seq == 1)[1]
            cols = cols + (i * tar_seq.shape[1])
            tar_sequences[rows, cols] = 1
        if self.train:
            for s in self.samples:
                if video_path == s[0]:
                    label = s[1]
                    break
            #label = self.samples[video_idx][1]
            label = self.classes.index(label)
        else:
            label = -1

        return src_sequences, video_path, stroke_tuple, tar_sequences, label


class StrokeFeaturePairsDatasetv2(VisionDataset):
    """
    Cricket Feature Pairs for Seq2Seq training for strokes dataset. Use dynamic margin labels
    """

    def __init__(self, feat_path, videos_list, dataset_path, strokes_dir, class_ids_path, 
                 frames_per_clip, extracted_frames_per_clip=16, step_between_clips=1, 
                 future_step = 1, train=True, frame_rate=None, framewiseTransform=True, 
                 transform=None, _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(StrokeFeaturePairsDatasetv2, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.extracted_frames_per_clip = extracted_frames_per_clip
        self.step_between_clips = step_between_clips
        self.future_step = future_step
        assert os.path.isfile(feat_path), "Path does not exist. {}".format(feat_path)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        self.nSamples = 50000
        
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
        self.max_dist = max([p[1] for p in self.pairs])
        
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
        vid_clip_idx = [self.video_clips.get_clip_location(i) \
                        for i in range(self.video_clips.num_clips())]
        strokes = [self.video_clips.video_paths[vidx]+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][0])+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][1]) \
                   for (vidx, clidx) in vid_clip_idx]
        pairs, pos_pairs, neg_pairs = [], [], []
        pos_tDist, neg_tDist = [], []
        # create clip pairs such that the distance between them is the temporal distance
        # between them
        for i in range(self.nSamples):
            random_pair = random.sample(vid_clip_idx, 2)
            pairs.append((random_pair, abs(random_pair[0][1] - random_pair[1][1])))
            
#        prev = strokes[0]
#        for i in range(0, len(strokes), future_step): 
#            s = strokes[i]
#            if i==0:
#                continue
#            if s == prev:
#                pos_pairs.append([vid_clip_idx[i - future_step], vid_clip_idx[i]])
#                pos_tDist.append(future_step)
#            else:
#                prev = s
##        neg_idx_all = random.sample(list(range(len(strokes))), len(strokes))
#        # create negative pair samples, take clips from other strokes as target
#        for i in range(0, len(strokes), future_step):
#            neg_sample = self.sample_easy_neg(vid_clip_idx, strokes, i, 
#                                            self.frames_per_clip,
#                                            self.extracted_frames_per_clip)
#            neg_pairs.append(neg_sample[0])
#            neg_tDist.append(neg_sample[1])
#        for i in range(min(len(pos_pairs), len(neg_pairs))):
#            pairs.append((pos_pairs[i], 1, pos_tDist[i]))
#            pairs.append((neg_pairs[i], 0, neg_tDist[i]))
            
        return pairs
    
#    def sample_easy_neg(self, vid_clip_idx, strokes, ref_clip_idx, seq_size, ex_fpc):
#        '''Get clip from other stroke and more than seq distance apart from ref_clip
#        '''
#        neg_clip = None
#        ref_clip = vid_clip_idx[ref_clip_idx]
#        potential_clips_bool = [abs(ref_clip[1]-c[1]) > 2*seq_size for c in vid_clip_idx]
#        # sampling to be done from True samples
#        potential_clips = [i for i, b in enumerate(potential_clips_bool) if b]
#        
#        neg_clip = random.sample(potential_clips, 1)[0]
#        
#        #######################################################
##        s = strokes[ref_clip_idx]
##        strokes_idx = list(range(len(strokes)))
##        
##        while neg_clip is None:
##            random_clip_idx = random.sample(strokes_idx, 1)[0]
##            if strokes[random_clip_idx] != s:
##                neg_clip = random_clip_idx
#        
#        #######################################################
##        def get_negative_sample(s):
##            random_clip_idx = random.sample(strokes_idx, 1)
##            if strokes[random_clip_idx] != s:
##                return random_clip_idx
##            else:
##                return get_negative_sample(s)
##        
##        neg_clip = get_negative_sample(s)
#        return [vid_clip_idx[ref_clip_idx], vid_clip_idx[neg_clip]]
        
        
#        if ref_clip_idx - seq < 0:
#        random.randrange(0, ref_clip_idx-seq) vid_clip_idx
        
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
                stroke2_tuple, label/self.max_dist



def get_negative_sample(cuts):
    idx = random.randint(0, len(cuts) - 1)
    if cuts[idx] == 0:
        return idx
    else:
        return get_negative_sample(cuts)
    
    