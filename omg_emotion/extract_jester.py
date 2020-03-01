import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import shutil
import random
import subprocess
import tqdm

from relative_baseline.omg_emotion import project_paths as PP


def get_label_mapping():
    labels_path = os.path.join(PP.jester_location, 'label_number.txt')
    labels = np.genfromtxt(labels_path, delimiter='\n', dtype=str)
    mapping = {}
    for i in range(len(labels)):
        mapping[labels[i]] = i

    return mapping


def change_text_to_numbered_labels():
    which = ['train', 'test', 'val']
    mapping = get_label_mapping()

    for i in which:
        txt_path = os.path.join(PP.jester_location, 'jester-v1-%s.txt' % i)

        new_path = os.path.join(PP.jester_location, 'labels_%s.npy' % i)

        which_labels = np.genfromtxt(txt_path, delimiter=';', dtype=str)

        for j in range(which_labels.shape[0]):
            which_labels[j][1] = str(mapping[which_labels[j][1]])

        # save as array
        np.save(new_path, which_labels)


def load_labels(which):
    base_path = os.path.join(PP.jester_location, 'labels_%s.npy' % which)
    labs = np.load(base_path).astype(int)
    return labs


def wc_l(path):
    ps = subprocess.Popen(('ls', path), stdout=subprocess.PIPE)
    output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout)
    output = int(output.decode('utf-8'))
    return output

    # return subprocess.check_output(['ls', path, '|', 'wc', '-l']).decode('utf-8')
    #
    # command = "ls %s | wc -l" % path
    # subprocess.call(command, shell=True)


def get_information():
    base_path = os.path.join(PP.jester_location, 'data')
    vid_folders = os.listdir(base_path)
    vid_folders.sort()

    print('getting number of frames...')
    frames = []
    for vid in tqdm.tqdm(range(len(vid_folders))):

        vid_path = os.path.join(base_path, vid_folders[vid])
        imgs = wc_l(vid_path)
        frames.append(imgs)

        frames.append(len(imgs))

    frames = np.array(frames)
    avg_frames = np.mean(frames)
    print('average number of frames:    %d\n'
          'max number of frames:        %d\n'
          'min number of frames:        %d\n'
          % (int(avg_frames), frames.max(), frames.min()))

    # TODO: number of frames less than x

    # which = ['train', 'test', 'val']
    # for i in which:
    #     print('\n getting class balance %s...' % i)
    #
    #     labs = load_labels(i)[:, 1]
    #     print('(perfect class balance at %d per class)' % (len(labs) // 27))
    #
    #     for j in range(27):
    #         print('class %d:    %d' % (j, sum(labs == j)))
    #     print('\n ------')


get_information()

