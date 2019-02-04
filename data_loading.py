import numpy as np
import os
import subprocess
import random

# assume we have 2 matrices, one val one train
# each cell is a story-subject indicating number of frames
# generate 2 pairs of story-sunjects
# for the number in each cell, generate a number between 0 and num_frames
# create label
# fetch 2 jpgs



def wc_l(file_name):
    command = "cat %s | wc -l" % file_name
    frames = subprocess.check_output(command, shell=True).decode('utf-8')
    frames = int(frames)
    return frames - 1
    # return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')


def make_frame_matrix():
    # TODO: add frames for testing
    path = '/scratch/users/gabras/data/omg_empathy/'
    _shape = (10, 8)  # story, subject
    frame_matrix = np.zeros(_shape, dtype=int)

    valid_story_idx_train = [2-1, 4-1, 5-1, 8-1]
    valid_story_idx_val = [1-1]
    valid_story_idx_test = [3-1, 6-1, 7-1]

    data_folders = os.listdir(path)
    for f in data_folders:
        annotation_path = os.path.join(path, f, 'Annotations')
        if os.path.exists(annotation_path):
            csv_list = os.listdir(annotation_path)
            for csv_file in csv_list:
                csv_path = os.path.join(annotation_path, csv_file)
                subject = int(csv_file.split('_')[1]) - 1  # -1 because array idx at 0
                story = int(csv_file.split('_')[-1].split('.')[0]) - 1
                frames = wc_l(csv_path)
                frame_matrix[subject][story] = frames

    valid_story_idx = [valid_story_idx_train, valid_story_idx_val, valid_story_idx_test]

    return frame_matrix, valid_story_idx


def dummy_load_data():
    #  data = np.zeros((len(keys), 3, C2.H, C2.W), dtype=np.float32)
    num_samples = 10
    label = np.array([random.randint(-1, 1) for i in range(num_samples)], dtype=np.float32)
    label = np.expand_dims(label, -1)

    img_left = np.random.randint(low=0, high=255, size=(num_samples, 3, 320, 640)).astype(np.float32)
    img_right = np.random.randint(low=0, high=255, size=(num_samples, 3, 320, 640)).astype(np.float32)

    return label, np.array([img_left, img_right], dtype=np.float32)


# TODO
def load_data(which, frame_matrix, val_idx):
    '''
    generate 32 points from val_idx using poisson disk
    for each point:
        s1, s2 sample from frames

    data_left = fetch_data(which, list_subj_story, frames) -> shape = (batchsize, 3, 320, 640)
    data_right = fetch_data(which, list_subj_story, frames) -> shape = (batchsize, 3, 320, 640)
    labels =

    '''


    pass


# TODO
def update_step_logs(which, loss, experiment_number):
    if which == 'train':
        path = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/train/steps'

    elif which == 'val':
        path = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/val/steps'

    elif which == 'test':
        raise NotImplemented




