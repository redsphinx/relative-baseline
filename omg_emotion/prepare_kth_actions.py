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


