from __future__ import print_function, division
import torch
# import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

from relative_baseline.omg_emotion import project_paths as PP

# --------------
# for the nvidia dali loader

import os
import logging


from .video_iterator import get_video_list

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# --------------


class JesterDataset(Dataset):

    def __init__(self, which, project_variable):
        phases = ['train', 'val', 'test']
        assert which in phases
        idx_which = phases.index(which)

        label_path = os.path.join(PP.jester_location, 'labels_%s.npy' % which)
        self.labels = np.load(label_path)[:project_variable.data_points[idx_which]]
        self.root_dir = os.path.join(PP.jester_data_50_75)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()



        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


jester_dataset = JesterDataset()
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())


# nvidia dali loader

class VideoPipe(Pipeline):
    def __init__(self, batch_size, num_threads=6, device_id=0, filenames="", shuffle=False, sequence_length=16, step=-1,
                 stride=1, initial_fill=1024, seed=0):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)
        # initial_prefetch_size = initial_prefetch_size
        # https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/supported_ops.html#videoreader
        self.input = ops.VideoReader(device="gpu", filenames=filenames,
                                     sequence_length=sequence_length, step=step, stride=stride,
                                     shard_id=0, num_shards=1,
                                     random_shuffle=shuffle, initial_fill=initial_fill)
        #self.augmentations = {}
        #self.augmentations["resize"] = ops.Resize(device="gpu", resize_shorter=32)

    def define_graph(self):
        output = self.input(name="Reader")
        return output
        # RuntimeError: Critical error in pipeline: [/opt/dali/dali/pipeline/operator/op_schema.h:306] The layout "FHWC" does not match any of the allowed layouts for input 0. Valid layouts are:
        # HWC
        # clip = self.input(name="Reader")
        # n = len(self.augmentations)
        # aug_list = list(self.augmentations.values())
        # # outputs[0] is the original cropped image
        # output = clip
        # for i in range(n):
        #     output = aug_list[i](output)
        # return output


def create_dali_iters(name, batch_size, num_workers=16, **kwargs):
    data_root = '/media/research/Data/{}'.format(name)

    train_video_files = get_video_list(data_root, kwargs['trainlist'], kwargs['max_videos'])
    test_video_files = get_video_list(data_root, kwargs['testlist'], kwargs['max_videos'])

    train_pipe = VideoPipe(batch_size=batch_size,
                           filenames=train_video_files,
                           shuffle=True,
                           sequence_length=kwargs['clip_length'],
                           step=kwargs['clip_step'],
                           stride=kwargs['clip_stride'],
                           initial_fill=10 * batch_size)  # ,
                           # seed=kwargs['seed'])
    train_pipe.build()

    test_pipe = VideoPipe(batch_size=batch_size,
                          filenames=test_video_files,
                          shuffle=False,
                          sequence_length=kwargs['clip_length'],
                          step=kwargs['clip_step'],
                          stride=kwargs['clip_stride'],
                          initial_fill=10 * batch_size,
                          seed=kwargs['seed'])
    test_pipe.build()

    # class nvidia.dali.plugin.pytorch.DALIGenericIterator(pipelines, output_map, size, auto_reset=False, fill_last_batch=True, dynamic_shape=False, last_batch_padded=False
    train_dali_iter = DALIGenericIterator([train_pipe], ['data'], train_pipe.epoch_size("Reader"), auto_reset=True)
    test_dali_iter = DALIGenericIterator([test_pipe], ['data'], test_pipe.epoch_size("Reader"), auto_reset=True)

    # # Testing iterations
    # import matplotlib.pyplot as plt
    # import numpy as np
    # epochs = 5
    # for e in range(epochs):
    #     rows = None
    #     batches = 5
    #     for i, data in enumerate(train_dali_iter):
    #         if i >= batches:
    #             break
    #         row = None
    #         for d in data:
    #             clips = d["data"] # B T H W C
    #             for c in range(clips.shape[0]):
    #                 if row is None:
    #                     row = clips[c,0,:,:,:]
    #                 else:
    #                     row = torch.cat((row, clips[c,0,:,:,:]), 1)
    #         if rows is None:
    #             rows = row
    #         else:
    #             rows = torch.cat((rows, row), 0)
    #
    #     plt.figure
    #     plt.imshow(rows.data.cpu())
    #     plt.show()

    logging.debug("VideoIter:: clip_length = {}, clip_step = {}, clip_stride = {}, batch_size = {}, number of batches = [train: {}, val: {}]".format( \
        kwargs['clip_length'], kwargs['clip_step'], kwargs['clip_stride'], batch_size, train_dali_iter._size, test_dali_iter._size,))

    return train_dali_iter, test_dali_iter