import numpy as np
import os

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIGenericIterator

from relative_baseline.omg_emotion import project_paths as PP


class VideoPipe(Pipeline):
    def __init__(self, batch_size, num_threads=6, device_id=0, file_list="", shuffle=False, sequence_length=30, step=-1,
                 stride=1, initial_fill=1024, seed=0):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=seed)

        # filenames assumes the selection of which files to load has already been made
        # TODO: does it load in order specified?
        # self.input = ops.VideoReader(device="gpu", filenames=filenames,
        self.input = ops.VideoReader(device="gpu", file_list=file_list,
                                     sequence_length=sequence_length, step=step, stride=stride,
                                     shard_id=0, num_shards=1,
                                     random_shuffle=shuffle, initial_fill=initial_fill)


    def define_graph(self):
        # output, labels = self.input()
        output, labels = self.input(name="Reader")
        return output, labels

# def create_dali_iters(batch_size, file_names, num_workers):
def create_dali_iters(batch_size, file_list, num_workers):

    train_pipe = VideoPipe(batch_size=batch_size,
                           file_list=file_list,
                           # filenames=file_names,
                           shuffle=False,
                           initial_fill=2 * batch_size,
                           num_threads=num_workers)
    train_pipe.build()

    train_dali_iter = DALIGenericIterator([train_pipe], ['data', 'labels'], train_pipe.epoch_size("Reader"), auto_reset=True)

    # fill_last_batch = True, last_batch_padded = False  -> last batch = ``[7, 1]``, next iteration will return ``[2, 3]``

    return train_dali_iter


def tryout():
    b, e = 0, 100

    the_filelist_val = os.path.join(PP.jester_location, 'filelist_val.txt')
    # jester_labels_val_path = os.path.join(PP.jester_location, 'labels_val.npy')

    # labels_val = np.load(jester_labels_val_path)[b:e]

    num_epochs = 10

    # file_names = [os.path.join(PP.jester_data_50_75_avi, '%s.avi' % i) for i in labels_val[:, 0]]

    val_iter = create_dali_iters(batch_size=2*27, file_list=the_filelist_val, num_workers=4)

    for epoch in range(num_epochs):

        num_steps = 5

        for i, data in enumerate(val_iter):
            if i >= num_steps:
                break

            print('well here we are')


tryout()
