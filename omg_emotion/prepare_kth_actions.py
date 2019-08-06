import matplotlib.pyplot as plt
import os
import subprocess
import tqdm
import numpy as np

from relative_baseline.omg_emotion import project_paths as PP

def prepare_data():
    def avi_to_png(vid_path, img_path):
        command = 'ffmpeg -loglevel panic -i ' + vid_path + ' -f image2 ' + img_path + '/%d.png'
        subprocess.call(command, shell=True)

    # AVI to PNG
    # sort into which folders
    train = [11, 12, 13, 14, 15, 16, 17, 18]
    val = [19, 20, 21, 23, 24, 25, 1, 4]
    test = [22, 2, 3, 5, 6, 7, 8, 9, 10]

    classes = os.listdir(PP.kth_location)

    for c in classes:
        print(c)
        folder_path = os.path.join(PP.kth_location, c)
        all_avis = os.listdir(folder_path)
        for avi in tqdm.tqdm(all_avis):
            avi_path = os.path.join(folder_path, avi)
            which  = None
            person = int(avi.split('person')[-1][:2])
            if person in train:
                which = 'train'
            elif person in val:
                which = 'val'
            elif person in test:
                which = 'test'
            save_path = os.path.join(PP.kth_png, which, c, avi.split('_uncomp')[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            avi_to_png(avi_path, save_path)


def prepare_frames_metadata():
    src_path = '/huge/gabras/kth_actions/00sequences.txt'
    dest_path = '/huge/gabras/kth_actions/metadata.txt'

    with open(src_path, 'r') as my_file:
        content = my_file.readlines()

    content = content[21:]

    for i in range(len(content)):
        line = content[i]
        print(i, line)

        if line.strip() != '' and i != 306:
            person = int(line.strip().split('\t')[0].split('_')[0][-2:])
            cls = line.strip().split('\t')[0].split('_')[1]
            event = int(line.strip().split('\t')[0].split('_')[-1].strip()[-1])
            frames = int(line.strip()[-3:])

            new_line = '%d,%s,%d,%d\n' % (person, cls, event, frames)

            with open(dest_path, 'a') as my_file:
                my_file.write(new_line)


def get_statistics():
    '''
    boxing:         lowest 250,     highest: 701,   average: 449
    handclapping:   lowest 284,     highest: 558,   average: 430
    handwaving:     lowest 330,     highest: 824,   average: 534
    jogging:        lowest 284,     highest: 660,   average: 438
    running:        lowest 32,      highest: 686,   average: 364
    walking:        lowest 60,      highest: 960,   average: 617
    '''
    metadata = np.genfromtxt(PP.kth_metadata, delimiter=',', dtype=str)
    translate = {'boxing':0, 'handclapping':1, 'handwaving':2, 'jogging':3, 'running':4, 'walking':5}
    translate_back = {0:'boxing', 1:'handclapping', 2:'handwaving', 3:'jogging', 4:'running', 5:'walking'}
    meta_dict =  {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}

    for i in range(len(metadata)):
        meta_dict[translate[metadata[i][1]]].append(int(metadata[i][-1]))

    for c in range(6):
        print('%s: lowest %d, highest: %d, average: %d'
              % (translate_back[c], min(meta_dict[c]), max(meta_dict[c]), int(np.mean(meta_dict[c])))
              )

    return meta_dict


def plot_statistics():
    meta_dict = get_statistics()
    translate_back = {0: 'boxing', 1: 'handclapping', 2: 'handwaving', 3: 'jogging', 4: 'running', 5: 'walking'}
    save_location = '/huge/gabras/kth_actions/histograms'
    n_bins = 50

    for i in meta_dict.keys():
        fig = plt.figure()
        plt.hist(meta_dict[i], bins=n_bins)
        plt.title(translate_back[i])
        plt.savefig(os.path.join(save_location, '%s_histogram.jpg' % (translate_back[i])))
        del fig

# plot_statistics()
