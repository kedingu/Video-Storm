# -*- coding: utf-8 -*-
# [MOD] MMCV→MMEngine migrations + I/O robustness + PyTorch collate + CSV/TXT reading

from logging import Logger
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import numpy as np
from functools import partial
import random

import io
import os
import os.path as osp
import shutil
import warnings
from collections.abc import Mapping, Sequence
# [MOD] Registry/build_from_cfg compatibility + file I/O (prefer MMEngine, fallback to MMCV 1.x)
try:
    from mmengine.registry import Registry, build_from_cfg  # ✅ MMEngine / MMCV 2
    from mmengine.fileio import load as file_load, dump as file_dump
    HAVE_MMENGINE = True
except ImportError:
    from mmcv.utils import Registry, build_from_cfg          # ✅ MMCV 1.x
    import mmcv
    file_load = mmcv.load
    file_dump = mmcv.dump
    HAVE_MMENGINE = False

from torch.utils.data import Dataset
import copy
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
import tarfile
from .pipeline import *  # assumes a Compose/registry based on PIPELINES below
from torch.utils.data import DataLoader
# [MOD] Use PyTorch collate (mmcv.parallel.collate removed in MMCV 2)
from torch.utils.data.dataloader import default_collate as torch_default_collate
import pandas as pd
import csv  # [MOD] for robust .csv reading

# [MOD] Local registry for pipeline steps
PIPELINES = Registry('pipeline')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 ann_file,
                 pipeline,
                 repeat=1,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0,
                 dynamic_length=False,):
        super().__init__()
        # [MOD] Robustness when data_prefix == None
        self.use_tar_format = isinstance(data_prefix, str) and (".tar" in data_prefix)
        data_prefix_clean = (data_prefix.replace(".tar", "") if isinstance(data_prefix, str) else data_prefix)

        self.ann_file = ann_file
        self.repeat = repeat
        self.data_prefix = osp.realpath(data_prefix_clean) if (data_prefix_clean is not None and osp.isdir(data_prefix_clean)) else data_prefix_clean
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.dynamic_length = dynamic_length

        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        # [MOD] file_load (MMEngine) or mmcv.load (fallback)
        video_infos = file_load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        # [MOD] file_dump (MMEngine) or mmcv.dump (fallback)
        return file_dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        aug1 = self.pipeline(results)
        if self.repeat > 1:
            aug2 = self.pipeline(results)
            ret = {
                "imgs": torch.cat((aug1['imgs'], aug2['imgs']), 0),
                "label": aug1['label'].repeat(2),
            }
            return ret
        else:
            return aug1

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)


class VideoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        self.labels_file = labels_file

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        # [MOD] Robust CSV vs TXT reading
        if self.ann_file.endswith('.csv'):
            with open(self.ann_file, 'r', newline='') as fin:
                reader = csv.reader(fin)
                for row in reader:
                    if not row:
                        continue
                    if self.multi_class:
                        assert self.num_classes is not None
                        filename, labels = row[0], row[1:]
                        label = list(map(int, labels))
                    else:
                        if len(row) < 2:
                            # [MOD] ignore/warn empty or malformed lines
                            warnings.warn(f"[load_annotations] ligne CSV invalide: {row}")
                            continue
                        filename, label = row[0], int(row[1])
                    if self.data_prefix is not None:
                        filename = osp.join(self.data_prefix, filename)
                    video_infos.append(dict(filename=filename, label=label, tar=self.use_tar_format))
        else:
            with open(self.ann_file, 'r') as fin:
                for line in fin:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if self.multi_class:
                        assert self.num_classes is not None
                        filename, labels = parts[0], parts[1:]
                        label = list(map(int, labels))
                    else:
                        if len(parts) < 2:
                            warnings.warn(f"[load_annotations] ligne TXT invalide: {parts}")
                            continue
                        filename, label = parts[0], int(parts[1])
                    if self.data_prefix is not None:
                        filename = osp.join(self.data_prefix, filename)
                    video_infos.append(dict(filename=filename, label=label, tar=self.use_tar_format))
        return video_infos


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


# [MOD] Simplified collate via PyTorch (no dependency on mmcv.parallel.collate)
def mmcv_collate(batch, samples_per_gpu=1):
    return torch_default_collate(batch)


def build_dataloader(logger, config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)

    train_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='Flip', flip_ratio=config.AUG.FLIP_RATIO),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label']),
    ]

    train_data = VideoDataset(
        ann_file=config.DATA.TRAIN_FILE,
        data_prefix=config.DATA.ROOT,
        labels_file=config.DATA.LABEL_LIST,
        pipeline=train_pipeline
    )
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_data,
        sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )

    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[1] = dict(
            type='SampleFrames',
            clip_len=1,
            frame_interval=1,
            num_clips=config.DATA.NUM_FRAMES,
            multiview=config.TEST.NUM_CLIP
        )

    val_data = VideoDataset(
        ann_file=config.DATA.VAL_FILE,
        data_prefix=config.DATA.ROOT,
        labels_file=config.DATA.LABEL_LIST,
        pipeline=val_pipeline
    )
    indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)
    val_loader = DataLoader(
        val_data,
        sampler=sampler_val,
        batch_size=2,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=2),
    )

    return train_data, val_data, train_loader, val_loader
