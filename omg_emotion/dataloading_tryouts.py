import numpy as np
import os

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from relative_baseline.omg_emotion import project_paths as PP


class VideoPipe(Pipeline):
    def __init__(self, batch_size, num_threads=6, device_id=0, filenames="", shuffle=False, sequence_length=30, step=-1,
                 stride=1, initial_fill=1024, seed=0):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)

        # filenames assumes the selection of which files to load has already been made
        # TODO: does it load in order specified?
        self.input = ops.VideoReader(device="gpu", filenames=filenames,
                                     sequence_length=sequence_length, step=step, stride=stride,
                                     shard_id=0, num_shards=1,
                                     random_shuffle=shuffle, initial_fill=initial_fill)

    def define_graph(self):
        output = self.input(name="Reader")
        return output


def create_dali_iters(batch_size, file_names, num_workers):

    train_pipe = VideoPipe(batch_size=batch_size,
                           filenames=file_names,
                           shuffle=False,
                           initial_fill=2 * batch_size,
                           num_threads=num_workers)
    train_pipe.build()

    # class nvidia.dali.plugin.pytorch.DALIGenericIterator(pipelines, output_map, size, auto_reset=False, fill_last_batch=True, dynamic_shape=False, last_batch_padded=False
    train_dali_iter = DALIGenericIterator([train_pipe], ['data'], train_pipe.epoch_size("Reader"), auto_reset=True)

    return train_dali_iter


def tryout():

    jester_data_path = PP.jester_data_50_75
    jester_labels_val_path = os.path.join(PP.jester_location, 'labels_val.npy')

    labels_val = np.load(jester_labels_val_path)
    video_folders = os.listdir(jester_data_path)

    num_epochs = 10

    # TODO: get filenames
    file_names = []

    val_iter = create_dali_iters(batch_size=2*27, file_names=file_names, num_workers=4)

    for epoch in range(num_epochs):

        num_steps = 5

        for i, data in enumerate(val_iter):
            if i >= num_steps:
                break

            print(type(data), data.shape)

