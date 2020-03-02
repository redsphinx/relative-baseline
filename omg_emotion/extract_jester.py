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


def get_num_frames():
    total_videos = 148092
    frames = np.zeros(shape=(total_videos, 2), dtype=int)
    continue_from = 0
    base_path = PP.jester_data
    vid_folders = os.listdir(base_path)
    vid_folders.sort()

    if os.path.exists(PP.jester_frames):
        tmp = np.genfromtxt(PP.jester_frames, delimiter=',', dtype=int)
        if tmp.shape[0] == total_videos:
            return tmp
        else:
            frames[:tmp.shape[0]] = tmp
            continue_from = vid_folders.index(str(tmp[-1][0])) + 1

    if os.path.exists(PP.jester_zero):
        zeros = list(np.genfromtxt(PP.jester_zero, delimiter='\n', dtype=int))
    else:
        zeros = []

    print('getting number of frames, this may take a while...')
    for vid in tqdm.tqdm(range(continue_from, len(vid_folders))):
        vid_path = os.path.join(base_path, vid_folders[vid])
        imgs = wc_l(vid_path)
        frames[vid] = [vid_folders[vid], imgs]

        line = '%s,%i\n' % (vid_folders[vid], imgs)
        with open(PP.jester_frames, 'a') as my_file:
            my_file.write(line)

        if imgs == 0:
            print('video %s has zero frames\n' % vid_folders[vid])
            zeros.append(vid_folders[vid])

            line = '%s\n' % vid_folders[vid]
            with open(PP.jester_zero, 'a') as a_file:
                a_file.write(line)

        if vid > 0 and vid % 1000 == 0:
            tmp = np.genfromtxt(PP.jester_frames, delimiter=',', dtype=int)
            avg_frames = np.mean(tmp[:, 1])
            print('based on %d videos:'
                  'average number of frames:    %d\n'
                  'max number of frames:        %d\n'
                  'min number of frames:        %d\n\n'
                  % (len(tmp), int(avg_frames), tmp[:, 1].max(), tmp[:, 1].min()))

    return frames


def get_information():
    base_path = os.path.join(PP.jester_location, 'data')
    vid_folders = os.listdir(base_path)
    vid_folders.sort()

    print('getting number of frames, this will take a while...')
    frames = get_num_frames()

    avg_frames = np.mean(frames[:, 1])
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

