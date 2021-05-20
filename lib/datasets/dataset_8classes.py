from __future__ import with_statement
from __future__ import absolute_import
from __future__ import division
import json

import numpy as np
import torch

from torchvision import transforms
from io import open

import glob
import os
import pickle
import cv2
#import utils
import datasets.folder as folder
import pandas as pd

from torchvision.datasets.utils import list_dir
#from torchvision.datasets.video_utils import VideoClips
from utils.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset


class CricketStrokesFirstNDataset(VisionDataset):
    """
    `Cricket Dataset for strokes dataset. Sampling from first N frames of strokes.

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
                 frames_per_clip, step_between_clips=1, nclips=10, train=True, 
                 frame_rate=None, framewiseTransform=True, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(CricketStrokesFirstNDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        #assert os.path.exists(root), "Path does not exist. {}".format(root)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        self.class_by_idx_label = self.load_labels_index_file(class_ids_path)
        extensions = ('avi', 'mp4', )
        self.train = train
        self.NClips = nclips  # take first 10 clips only, if step = 4, then first 40 frames

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
        self.pairs = self.generateSequences()
        
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
    
    def generateSequences(self):
        '''Generate samples having first N clips of strokes
        '''
        vid_clip_idx = [self.video_clips.get_clip_location(i) \
                        for i in range(self.video_clips.num_clips())]
        vid_clip_idx = list(filter(lambda x: x[1] <= self.NClips, vid_clip_idx))
#        strokes = [self.video_clips.video_paths[vidx]+"_"+\
#                   str(self.video_clips.stroke_tuples[vidx][0])+"_"+\
#                   str(self.video_clips.stroke_tuples[vidx][1]) \
#                   for (vidx, clidx) in vid_clip_idx]
            
        return vid_clip_idx
    

    def __getitem__(self, idx):
        (video_idx, clip_idx) = self.pairs[idx]
        # reverse map (video_idx, clip_idx) to the global sample idx
        if video_idx != 0:
            clip_idx = self.video_clips.cumulative_sizes[video_idx - 1] + clip_idx
        
        video, vid_path, stroke, audio, info, video_idx = self.video_clips.get_clip(clip_idx)
        
        if self.transform is not None:
            if self.framewiseTransform:
                if isinstance(self.transform, transforms.Compose):
                    # transform frame-wise (takes input as HxWxC)
                    video = torch.stack([self.transform(i.permute(2, 0, 1)) for i in video])
            else:   # clip level transform (takes input as TxHxWxC)
                video = video.permute(0,3,1,2)      # Imp: To be used if ToPILImage is used
                video = self.transform(video)
            
        if isinstance(video, list):
            video = torch.stack(video)
            
#        if self.frames_per_clip == 1:       # for single frame sequences
#            video = np.squeeze(video, axis=0)
        
        return video, vid_path, stroke, 0
    
    

class CricketStrokesWithPoseDataset(VisionDataset):
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
                 pose_keys, pose_values, frames_per_clip, step_between_clips=1, 
                 train=True, frame_rate=None, framewiseTransform=True, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(CricketStrokesWithPoseDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        #assert os.path.exists(root), "Path does not exist. {}".format(root)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        self.pose_keys = pose_keys
        self.pose_values = pose_values
        
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
        video, vid_path, stroke, audio, info, video_idx = self.video_clips.get_clip(idx)
        
        if self.train:
            for s in self.samples:
                if vid_path == s[0]:
                    label = s[1]
                    break
            #label = self.samples[video_idx][1]
            label = self.classes.index(label)
        else:
            label = -1
        
        key = vid_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        key = key + '_' + str(stroke[0]) + '_' + str(stroke[1])
        pose_val = self.pose_values[self.pose_keys.index(key)]
        # If stroke with left handed batsman, make it right handed by HFlip
        if pose_val == 0:
            video = torch.flip(video, [2])      # B x 360 x 640 x 3
        
        if self.transform is not None:
            if self.framewiseTransform:
                if isinstance(self.transform, transforms.Compose):
                    # transform frame-wise (takes input as HxWxC)
                    video = torch.stack([self.transform(i.permute(2, 0, 1)) for i in video])
            else:   # clip level transform (takes input as TxHxWxC)
                video = video.permute(0,3,1,2)      # Imp: To be used if ToPILImage is used
                video = self.transform(video)
            
        if isinstance(video, list):
            video = torch.stack(video)
#        if self.frames_per_clip == 1:       # for single frame sequences
#            video = np.squeeze(video, axis=0)
        
        return video, vid_path, stroke, label


class CricketStrokesFirstNWithPoseDataset(VisionDataset):
    """
    `Cricket Dataset for strokes dataset. Sampling from first N frames of strokes.

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
                 pose_keys, pose_values, frames_per_clip, step_between_clips=1, 
                 nclips=10, train=True, frame_rate=None, framewiseTransform=True, 
                 transform=None, _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(CricketStrokesFirstNWithPoseDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        #assert os.path.exists(root), "Path does not exist. {}".format(root)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        self.class_by_idx_label = self.load_labels_index_file(class_ids_path)
        extensions = ('avi', 'mp4', )
        self.train = train
        self.NClips = nclips  # take first 10 clips only, if step = 4, then first 40 frames
        self.pose_keys = pose_keys
        self.pose_values = pose_values

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
        self.pairs = self.generateSequences()
        
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
    
    def generateSequences(self):
        '''Generate samples having first N clips of strokes
        '''
        vid_clip_idx = [self.video_clips.get_clip_location(i) \
                        for i in range(self.video_clips.num_clips())]
        vid_clip_idx = list(filter(lambda x: x[1] <= self.NClips, vid_clip_idx))
#        strokes = [self.video_clips.video_paths[vidx]+"_"+\
#                   str(self.video_clips.stroke_tuples[vidx][0])+"_"+\
#                   str(self.video_clips.stroke_tuples[vidx][1]) \
#                   for (vidx, clidx) in vid_clip_idx]
            
        return vid_clip_idx
    
    def __getitem__(self, idx):
        (video_idx, clip_idx) = self.pairs[idx]
        # reverse map (video_idx, clip_idx) to the global sample idx
        if video_idx != 0:
            clip_idx = self.video_clips.cumulative_sizes[video_idx - 1] + clip_idx
        
        video, vid_path, stroke, audio, info, video_idx = self.video_clips.get_clip(clip_idx)
        
        key = vid_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        key = key + '_' + str(stroke[0]) + '_' + str(stroke[1])
        pose_val = self.pose_values[self.pose_keys.index(key)]
        # If stroke with left handed batsman, make it right handed by HFlip
        if pose_val == 0:
            video = torch.flip(video, [2])      # B x 360 x 640 x 3
        
        if self.transform is not None:
            if self.framewiseTransform:
                if isinstance(self.transform, transforms.Compose):
                    # transform frame-wise (takes input as HxWxC)
                    video = torch.stack([self.transform(i.permute(2, 0, 1)) for i in video])
            else:   # clip level transform (takes input as TxHxWxC)
                video = video.permute(0,3,1,2)      # Imp: To be used if ToPILImage is used
                video = self.transform(video)
            
        if isinstance(video, list):
            video = torch.stack(video)
            
#        if self.frames_per_clip == 1:       # for single frame sequences
#            video = np.squeeze(video, axis=0)
        
        return video, vid_path, stroke, 0

class CricketStrokesBBoxesDataset(VisionDataset):
    """
    `Cricket Dataset for spatio-temporal feature extraction from bboxes in strokes dataset.

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

    def __init__(self, videos_list, dataset_path, strokes_dir, bbox_feats_paths, bbox_snames_paths,
                 class_ids_path, frames_per_clip, step_between_clips=1, train=True, 
                 frame_rate=None, framewiseTransform=True, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(CricketStrokesBBoxesDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        #assert os.path.exists(root), "Path does not exist. {}".format(root)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        self.bbox_feats_paths = bbox_feats_paths
        self.bbox_snames_paths = bbox_snames_paths
        self.read_bbox_coords()
        
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
        
        self.framewiseTransform = framewiseTransform
        self.transform = transform
        self.pairs, self.strokes = self.generateSequences()
        
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
    
    def read_bbox_coords(self):
        assert os.path.isfile(self.bbox_feats_paths[0]), "File doesn't exist!"
        self.bbox_feats, self.bbox_snames = {}, []
        for f in self.bbox_feats_paths:
            with open(f, 'rb') as fp:
                feats = pickle.load(fp)
            self.bbox_feats.update(feats)
        for f in self.bbox_snames_paths:
            with open(f, 'rb') as fp:
                snames = pickle.load(fp)
            self.bbox_snames.extend(snames)
        # get lengths of strokes having bboxes, 
        self.bb_nframes = [self.bbox_feats[sn].shape[0] for sn in self.bbox_snames]
            
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
    
    def generateSequences(self):
        '''Generate samples having first N clips of strokes
        '''
        vid_clip_idx = [self.video_clips.get_clip_location(i) \
                        for i in range(self.video_clips.num_clips())]
        strokes = [self.video_clips.video_paths[vidx].rsplit('/',1)[1].rsplit('.',1)[0]\
                   +"_"+ str(self.video_clips.stroke_tuples[vidx][0])+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][1]) \
                   for (vidx, clidx) in vid_clip_idx]

        # filter the clips having 
        vid_clip_idx_bool = [cid <= ((self.bbox_feats[strokes[i]].shape[0] \
                                     - self.frames_per_clip) / self.step_between_clips) \
                             for i, (vid, cid) in enumerate(vid_clip_idx)]
        
        vid_clip_idx = [vid_clip_idx[i] for i in range(len(vid_clip_idx)) if vid_clip_idx_bool[i]]
        strokes = [strokes[i] for i in range(len(strokes)) if vid_clip_idx_bool[i]]
        
        return vid_clip_idx, strokes
    
    @property
    def metadata(self):
        return self.video_clips_metadata

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (video_idx, clip_idx) = self.pairs[idx]
        # reverse map (video_idx, clip_idx) to the global sample idx
        if video_idx != 0:
            clip_idx = self.video_clips.cumulative_sizes[video_idx - 1] + clip_idx
        
        video, vid_path, stroke, audio, info, video_idx = self.video_clips.get_clip(clip_idx)
        
        key = self.strokes[idx]
        clip_size = video.shape[0]
#        key = vid_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
#        key = key + '_' + str(stroke[0]) + '_' + str(stroke[1])
        bbox_values = self.bbox_feats[key][self.pairs[idx][1]:(self.pairs[idx][1] + clip_size)]

        vid = []
        for i in range(clip_size):
            v_frame = video[i]
            x0, y0, x1, y1 = bbox_values[i]
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > (v_frame.shape[1] - 1):
                x1 = v_frame.shape[1] - 1
            if y1 > (v_frame.shape[0] - 1):
                y1 = v_frame.shape[0] - 1
            vid.append(v_frame[y0:(y1+1), x0:(x1+1),...])
#        # If stroke with left handed batsman, make it right handed by HFlip
#        if pose_val == 0:
#            video = torch.flip(video, [2])      # SEQ x 360 x 640 x 3
        
        if self.transform is not None:
            if self.framewiseTransform:
                if isinstance(self.transform, transforms.Compose):
                    # transform frame-wise (takes input as HxWxC)
                    video = torch.stack([self.transform(i.permute(2, 0, 1)) for i in vid])
            else:   # clip level transform (takes input as TxHxWxC)     # don't use for spatio-temporal
                video = video.permute(0,3,1,2)      # Imp: To be used if ToPILImage is used
                video = self.transform(video)
            
        if isinstance(video, list):
            video = torch.stack(video)
            
#        if self.frames_per_clip == 1:       # for single frame sequences
#            video = np.squeeze(video, axis=0)
        
        return video, vid_path, stroke, 0



class CricketStrokesBBoxesWithPoseDataset(VisionDataset):
    """
    `Cricket Dataset for spatio-temporal feature extraction from bboxes in strokes dataset.

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

    def __init__(self, videos_list, dataset_path, strokes_dir, bbox_feats_paths, 
                 bbox_snames_paths, class_ids_path, pose_keys, pose_values, 
                 frames_per_clip, step_between_clips=1, train=True, 
                 frame_rate=None, framewiseTransform=True, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(CricketStrokesBBoxesWithPoseDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        #assert os.path.exists(root), "Path does not exist. {}".format(root)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        self.bbox_feats_paths = bbox_feats_paths
        self.bbox_snames_paths = bbox_snames_paths
        self.pose_keys = pose_keys
        self.pose_values = pose_values
        self.read_bbox_coords()
        
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
        
        self.framewiseTransform = framewiseTransform
        self.transform = transform
        self.pairs, self.strokes = self.generateSequences()
        
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
    
    def read_bbox_coords(self):
        assert os.path.isfile(self.bbox_feats_paths[0]), "File doesn't exist!"
        self.bbox_feats, self.bbox_snames = {}, []
        for f in self.bbox_feats_paths:
            with open(f, 'rb') as fp:
                feats = pickle.load(fp)
            self.bbox_feats.update(feats)
        for f in self.bbox_snames_paths:
            with open(f, 'rb') as fp:
                snames = pickle.load(fp)
            self.bbox_snames.extend(snames)
        # get lengths of strokes having bboxes, 
        self.bb_nframes = [self.bbox_feats[sn].shape[0] for sn in self.bbox_snames]
            
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
    
    def generateSequences(self):
        '''Generate samples having first N clips of strokes
        '''
        vid_clip_idx = [self.video_clips.get_clip_location(i) \
                        for i in range(self.video_clips.num_clips())]
        strokes = [self.video_clips.video_paths[vidx].rsplit('/',1)[1].rsplit('.',1)[0]\
                   +"_"+ str(self.video_clips.stroke_tuples[vidx][0])+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][1]) \
                   for (vidx, clidx) in vid_clip_idx]

        # filter the clips having 
        vid_clip_idx_bool = [cid <= ((self.bbox_feats[strokes[i]].shape[0] \
                                     - self.frames_per_clip) / self.step_between_clips) \
                             for i, (vid, cid) in enumerate(vid_clip_idx)]
        
        vid_clip_idx = [vid_clip_idx[i] for i in range(len(vid_clip_idx)) if vid_clip_idx_bool[i]]
        strokes = [strokes[i] for i in range(len(strokes)) if vid_clip_idx_bool[i]]
        
        return vid_clip_idx, strokes
    
    @property
    def metadata(self):
        return self.video_clips_metadata

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (video_idx, clip_idx) = self.pairs[idx]
        # reverse map (video_idx, clip_idx) to the global sample idx
        if video_idx != 0:
            clip_idx = self.video_clips.cumulative_sizes[video_idx - 1] + clip_idx
        
        video, vid_path, stroke, audio, info, video_idx = self.video_clips.get_clip(clip_idx)
        
        key = self.strokes[idx]
        clip_size = video.shape[0]
#        key = vid_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
#        key = key + '_' + str(stroke[0]) + '_' + str(stroke[1])
        bbox_values = self.bbox_feats[key][self.pairs[idx][1]:(self.pairs[idx][1] + clip_size)]
        pose_val = self.pose_values[self.pose_keys.index(key)]
        
        vid = []
        for i in range(clip_size):
            v_frame = video[i]
            x0, y0, x1, y1 = bbox_values[i]
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1 > (v_frame.shape[1] - 1):
                x1 = v_frame.shape[1] - 1
            if y1 > (v_frame.shape[0] - 1):
                y1 = v_frame.shape[0] - 1
            v_frame = v_frame[y0:(y1+1), x0:(x1+1),...]
            # If stroke with left handed batsman, make it right handed by HFlip
            if pose_val == 0:
                v_frame = torch.flip(v_frame, [1])      # SEQ x 360 x 640 x 3
            vid.append(v_frame)
        
        if self.transform is not None:
            if self.framewiseTransform:
                if isinstance(self.transform, transforms.Compose):
                    # transform frame-wise (takes input as HxWxC)
                    video = torch.stack([self.transform(i.permute(2, 0, 1)) for i in vid])
            else:   # clip level transform (takes input as TxHxWxC)     # don't use for spatio-temporal
                video = video.permute(0,3,1,2)      # Imp: To be used if ToPILImage is used
                video = self.transform(video)
            
        if isinstance(video, list):
            video = torch.stack(video)
        
        return video, vid_path, stroke, 0
    
class CricketStrokesBBoxesAugDataset(VisionDataset):
    """
    `Cricket Dataset for spatio-temporal feature extraction from bboxes in strokes dataset.
    Add augmentations by taking surrounding clips.

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

    def __init__(self, videos_list, dataset_path, strokes_dir, bbox_feats_paths, bbox_snames_paths,
                 class_ids_path, frames_per_clip, step_between_clips=1, train=True, 
                 frame_rate=None, framewiseTransform=True, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(CricketStrokesBBoxesAugDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        #assert os.path.exists(root), "Path does not exist. {}".format(root)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        self.bbox_feats_paths = bbox_feats_paths
        self.bbox_snames_paths = bbox_snames_paths
        self.read_bbox_coords()
        if train:
            self.sq_crop = 112  #pass as param
        else:
            self.sq_crop = 56
        
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
        
        self.framewiseTransform = framewiseTransform
        self.transform = transform
        self.pairs, self.strokes = self.generateSequences()
        
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
    
    def read_bbox_coords(self):
        assert os.path.isfile(self.bbox_feats_paths[0]), "File doesn't exist!"
        self.bbox_feats, self.bbox_snames = {}, []
        for f in self.bbox_feats_paths:
            with open(f, 'rb') as fp:
                feats = pickle.load(fp)
            self.bbox_feats.update(feats)
        for f in self.bbox_snames_paths:
            with open(f, 'rb') as fp:
                snames = pickle.load(fp)
            self.bbox_snames.extend(snames)
        # get lengths of strokes having bboxes, 
        self.bb_nframes = [self.bbox_feats[sn].shape[0] for sn in self.bbox_snames]
            
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
    
    def generateSequences(self):
        '''Generate samples having first N clips of strokes
        '''
        vid_clip_idx = [self.video_clips.get_clip_location(i) \
                        for i in range(self.video_clips.num_clips())]
        strokes = [self.video_clips.video_paths[vidx].rsplit('/',1)[1].rsplit('.',1)[0]\
                   +"_"+ str(self.video_clips.stroke_tuples[vidx][0])+"_"+\
                   str(self.video_clips.stroke_tuples[vidx][1]) \
                   for (vidx, clidx) in vid_clip_idx]

        # filter the clips having 
        vid_clip_idx_bool = [cid <= ((self.bbox_feats[strokes[i]].shape[0] \
                                     - self.frames_per_clip) / self.step_between_clips) \
                             for i, (vid, cid) in enumerate(vid_clip_idx)]
        
        vid_clip_idx = [vid_clip_idx[i] for i in range(len(vid_clip_idx)) if vid_clip_idx_bool[i]]
        strokes = [strokes[i] for i in range(len(strokes)) if vid_clip_idx_bool[i]]
        
        return vid_clip_idx, strokes
    
    @property
    def metadata(self):
        return self.video_clips_metadata

    def __len__(self):
        return len(self.pairs)

    def get_augmented_crop(self, video, bbox):
        '''get video clip and bbox_values for clip, and generate augmentation crop
        video : tensor of shape (SEQ, H, W, C)
        bbox_values : tensor of shape (SEQ, 4) having (x0, y0, x1, y1) are rows
        '''
        seq, h, w, c = video.shape
        bbox[bbox < 0] = 0
        bbox[:, 2][bbox[:, 2] >= w] = w - 1
        bbox[:, 3][bbox[:, 3] >= h] = h - 1
        bbox_ctr_x = int(np.mean((bbox[:, 2] + bbox[:, 0]) / 2.))
        bbox_ctr_y = int(np.mean((bbox[:, 3] + bbox[:, 1]) / 2.))
#        min_x, max_x = np.min(bbox[:, 0]), np.max(bbox[:, 2])
#        min_y, max_y = np.min(bbox[:, 1]), np.max(bbox[:, 3])
#        x_width = max_x - min_x
#        y_width = max_y - min_y
        X0, Y0 = (bbox_ctr_x - self.sq_crop//2) , (bbox_ctr_y - self.sq_crop//2)
        X1, Y1 = (bbox_ctr_x + self.sq_crop//2) , (bbox_ctr_y + self.sq_crop//2)
        if X0 < 0:          # shift right
            X1 += (-X0)
            X0 = 0
        if Y0 < 0:          # shift down
            Y1 += (-Y0)
            Y0 = 0
        if X1 > w:          # shift left
            X0 -= (X1 - w)
            X1 = w
        if Y1 > h:          # shift up
            Y0 -= (Y1 - h)
            Y1 = h
        # crop the square clip with center of bboxes, FiveCrop, Jitter, GaussianBlur
        vid = video[:, Y0:Y1, X0:X1, :]
        
        return vid

    def __getitem__(self, idx):
        (video_idx, clip_idx) = self.pairs[idx]
        # reverse map (video_idx, clip_idx) to the global sample idx
        if video_idx != 0:
            clip_idx = self.video_clips.cumulative_sizes[video_idx - 1] + clip_idx
        
        video, vid_path, stroke, audio, info, video_idx = self.video_clips.get_clip(clip_idx)
        
        key = self.strokes[idx]
        clip_size = video.shape[0]
#        key = vid_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
#        key = key + '_' + str(stroke[0]) + '_' + str(stroke[1])
        bbox_values = self.bbox_feats[key][self.pairs[idx][1]:(self.pairs[idx][1] + clip_size)]
        
        vid = self.get_augmented_crop(video, bbox_values)

#        # If stroke with left handed batsman, make it right handed by HFlip
#        if pose_val == 0:
#            video = torch.flip(video, [2])      # SEQ x 360 x 640 x 3
        
        if self.transform is not None:
            if self.framewiseTransform:
                if isinstance(self.transform, transforms.Compose):
                    # transform frame-wise (takes input as HxWxC)
                    video = torch.stack([self.transform(i) for i in vid])
            else:   # clip level transform (takes input as TxHxWxC)     # don't use for spatio-temporal
                vid = vid.permute(0,3,1,2)      # Imp: To be used if ToPILImage is used
                video = self.transform(vid)
            
        if isinstance(video, list):
            video = torch.stack(video)
            
#        if self.frames_per_clip == 1:       # for single frame sequences
#            video = np.squeeze(video, axis=0)
        
        return video, vid_path, stroke, 0


class StrokeFeatureSequenceDataset(VisionDataset):
    """
    `Cricket Feature Sequence Dataset for strokes dataset.

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

    def __init__(self, feat_path, videos_list, dataset_path, strokes_dir, class_ids_path, 
                 frames_per_clip, extracted_frames_per_clip=16, step_between_clips=1, 
                 train=True, frame_rate=None, framewiseTransform=True, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(StrokeFeatureSequenceDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.extracted_frames_per_clip = extracted_frames_per_clip
        self.step_between_clips = step_between_clips
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
        
        seq_start_idx = start_pts - stroke_tuple[0]
        seq_len = self.frames_per_clip - self.extracted_frames_per_clip + 1
        
        # retrieve the sequence of vectors from the stroke sequences
        sequence = vid_feats[seq_start_idx:(seq_start_idx+seq_len), :]
        
        if self.train:
            for s in self.samples:
                if video_path == s[0]:
                    label = s[1]
                    break
            #label = self.samples[video_idx][1]
            label = self.classes.index(label)
        else:
            label = -1

        return sequence, video_path, stroke_tuple, label


#    def get_bboxes_for_stroke(self, gt_rows, start, end):
#        '''Return the boxes for the stroke starting from 'start' to 'end'
#        Parameters:
#        -----------
#        start: int
#            starting frame number
#        end: int
#            ending frame number
#        gt_rows : list of tuples
#            annotations for the batsman bboxes. Contains all the annotations for the video
#            
#        Returns: 
#        --------
#        list of tuples [[x0, y0, x1, y1], ....]  
#        '''
#        boxes = []
#        for tup in gt_rows:
#            fno, lab_id, x0, y0, x1, y1, label = tup
#            if int(fno) >= start and int(fno) <= end and label=='Batsman':
#                boxes.append([int(x0), int(y0), int(x1), int(y1)])
#                
#        return boxes