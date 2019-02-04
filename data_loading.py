import numpy as np
import os
import subprocess

# assume we have 2 matrices, one val one train
# each cell is a story-subject indicating number of frames
# generate 2 pairs of story-sunjects
# for the number in each cell, generate a number between 0 and num_frames
# create label
# fetch 2 jpgs


# TODO: add frames for testing
def wc_l(file_name):
    # command = "cat %s | wc -l" % file_name
    # subprocess.call(command, shell=True)
    frames = subprocess.check_output(['cat', file_name, '|', 'wc', 'l']).decode('utf-8')
    frames = int(frames)
    return frames - 1
    # return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')


def make_frame_matrix():
    path = '/scratch/users/gabras/data/omg_empathy/'
    _shape = (10, 8)  # story, subject
    frame_matrix = np.zeros(_shape, dtype=int)

    valid_story_idx_train = [2, 4, 5, 8]
    valid_story_idx_val = [1]
    valid_story_idx_test = [3, 6, 7]

    data_folders = os.listdir(path)
    for f in data_folders:
        annotation_path = os.path.join(path, f, 'Annotations')
        if os.path.exists(annotation_path):
            csv_list = os.listdir(annotation_path)
            for csv_file in csv_list:
                csv_path = os.path.join(annotation_path, csv_file)
                subject = int(csv_file.split('_')[1]) - 1  # -1 because array idx at 0
                story = int(csv_file.split('_')[-1].split('.')[0]) - 1
                frames = wc_l(csv_file)
                frame_matrix[subject][story] = frames

    valid_story_idx = [valid_story_idx_train, valid_story_idx_val, valid_story_idx_test]

    return frame_matrix, valid_story_idx


