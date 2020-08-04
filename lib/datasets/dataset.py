from __future__ import with_statement
from __future__ import absolute_import
from __future__ import division
import json

import numpy as np
import torch
from torchnet.dataset import BatchDataset
from torchnet.dataset.dataset import Dataset

from torchvision.datasets import UCF101
from torchvision import transforms
from io import open

import glob
import os
import pickle
#import utils
import datasets.folder as folder
import pandas as pd

from torchvision.datasets.utils import list_dir
#from torchvision.datasets.video_utils import VideoClips
from utils.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset


class CricketStrokesDataset(VisionDataset):
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
                 frames_per_clip, step_between_clips=1, train=True, 
                 frame_rate=None, framewiseTransform=True, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(CricketStrokesDataset, self).__init__(dataset_path)
        
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        #assert os.path.exists(root), "Path does not exist. {}".format(root)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        assert os.path.exists(strokes_dir), "Path does not exist. {}".format(strokes_dir)
        self.strokes_dir = strokes_dir
        
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
        
        if self.transform is not None:
            if self.framewiseTransform:
                if isinstance(self.transform, transforms.Compose):
                    # transform frame-wise (takes input as HxWxC)
                    video = torch.stack([self.transform(i) for i in video])
            else:   # clip level transform (takes input as TxHxWxC)
                video = self.transform(video)
            
        if isinstance(video, list):
            video = torch.stack(video)
#        if self.frames_per_clip == 1:       # for single frame sequences
#            video = np.squeeze(video, axis=0)

        return video, vid_path, stroke, label

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

class StrokeFeaturePairsDataset(VisionDataset):
    """
    Cricket Feature Pairs for Seq2Seq training for strokes dataset.
    """

    def __init__(self, feat_path, videos_list, dataset_path, strokes_dir, class_ids_path, 
                 frames_per_clip, extracted_frames_per_clip=16, step_between_clips=1, 
                 train=True, frame_rate=None, framewiseTransform=True, transform=None, 
                 _precomputed_metadata=None, num_workers=1, _video_width=0, 
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(StrokeFeaturePairsDataset, self).__init__(dataset_path)
        
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
                
        vid_feats = self.features[key]
        
        src_start_idx = src_start_pts - stroke_tuple[0]
        tar_start_idx = tar_start_pts - stroke_tuple[0]
        seq_len = self.frames_per_clip - self.extracted_frames_per_clip + 1
        
        # retrieve the sequence of vectors from the stroke sequences
        src_sequence = vid_feats[src_start_idx:(src_start_idx+seq_len), :]
        tar_sequence = vid_feats[tar_start_idx:(tar_start_idx+seq_len), :]
        
        if self.train:
            for s in self.samples:
                if video_path == s[0]:
                    label = s[1]
                    break
            #label = self.samples[video_idx][1]
            label = self.classes.index(label)
        else:
            label = -1

        return src_sequence, tar_sequence, video_path, stroke_tuple, label

class THUMOS14Dataset(VisionDataset):
    """
    `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.

    UCF101 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the UCF101 Dataset.
        class_ids_path (str): path to the class IDs file
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int, optional): number of frames between each clip.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, class_ids_path, frames_per_clip, step_between_clips=1,
                 train=True, frame_rate=None , transform=None,
                 _precomputed_metadata=None, num_workers=1, _video_width=0,
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(THUMOS14Dataset, self).__init__(root)
        
        assert os.path.exists(root), "Path does not exist. {}".format(root)
        assert os.path.isfile(class_ids_path), "File does not exist. {}".format(class_ids_path)
        
#        if not 1 <= fold <= 3:
#            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.class_by_idx_label = self.load_labels_index_file(class_ids_path)
        extensions = ('avi', 'mp4', )
        self.train = train

        classes = sorted(list(self.class_by_idx_label.keys()))
        #print("Classes : {}".format(classes))
        
        class_to_idx = {self.class_by_idx_label[i]: i for i in classes}
        self.samples = folder.make_dataset(self.root, class_to_idx, train, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(video_list, frames_per_clip, step_between_clips)
        # self.video_clips_metadata = video_clips.metadata  # in newer torchvision
        # self.indices = self._select_fold(video_list, annotation_path, fold, train)
        # self.video_clips = video_clips.subset(self.indices)
        self.transform = transform

    def load_labels_index_file(self, filename):
        """
        Returns a dictionary of {ID: classname, ...}
        """
        if os.path.isfile(filename):
            df = pd.read_table(filename, sep=" ", names=["id", "classname"])
            d_by_cols = df.to_dict('list')
            d_by_idx_label = dict(zip(d_by_cols['id'],
                              d_by_cols['classname']))
            return d_by_idx_label
        else:
            return None
    
    @property
    def metadata(self):
        return self.video_clips_metadata

#    def _select_fold(self, video_list, annotation_path, fold, train):
#        name = "train" if train else "test"
#        name = "{}list{:02d}.txt".format(name, fold)
#        f = os.path.join(annotation_path, name)
#        selected_files = []
#        with open(f, "r") as fid:
#            data = fid.readlines()
#            data = [x.strip().split(" ") for x in data]
#            data = [x[0] for x in data]
#            selected_files.extend(data)
#        selected_files = set(selected_files)
#        indices = [i for i in range(len(video_list)) if video_list[i][len(self.root) + 1:] in selected_files]
#        return indices

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        
        if self.train:
            label = self.samples[video_idx][1]
            label = self.classes.index(label)
        else:
            label = None

        if self.transform is not None:
            video = self.transform(video)

        return video, label

