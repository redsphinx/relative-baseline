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
          % (int(avg_frames), frames[:, 1].max(), frames[:, 1].min()))

    less_than = 30

    frames = frames[:, 1]
    num_frames_less_than = sum(frames < less_than)
    num_frames_exactly = sum(frames == less_than)
    num_frames_gret_than = sum(frames > less_than)

    print('number of videos with less than %i frames: %d' % (less_than, num_frames_less_than))
    print('number of videos with exactly %i frames: %d' % (less_than, num_frames_exactly))
    print('number of videos with greater than %i frames: %d' % (less_than, num_frames_gret_than))

    which = ['train', 'test', 'val']
    for i in which:
        print('\n getting class balance %s...' % i)

        labs = load_labels(i)[:, 1]
        print('(perfect class balance at %d per class)' % (len(labs) // 27))

        for j in range(27):
            print('class %d:    %d' % (j, sum(labs == j)))
        print('\n ------')

'''
average number of frames:    35
max number of frames:        70
min number of frames:        12

number of videos with less than 30 frames: 2633
number of videos with exactly 30 frames: 948
number of videos with greater than 30 frames: 144511
'''

def adjust_frame(image, h, w, c):
    # resize to height of h
    or_w, or_h = image.size
    new_w = int(h * or_w / or_h)
    image = image.resize((new_w, h)) # w, h

    if new_w > w:
        delta_w = (new_w - w) // 2
        image = image.crop((delta_w, 0, new_w-delta_w-1, h)) # l, u, r, d
    elif new_w < w:
        delta_w = (w - new_w) // 2
        image = np.array(image)
        pixel_mean = np.mean(np.mean(image, axis=0), axis=0)
        pixel_mean =np.array(pixel_mean, dtype=int)
        canvas = np.ones(shape=(h, w, c), dtype=np.uint8)
        canvas = canvas * pixel_mean
        # paste it
        canvas[:, delta_w:new_w+delta_w, :] = image
        image = canvas

    image = np.array(image, dtype=np.uint8)
    assert image.shape == (h, w, c)

    return image


def standardize_clips(b, e):
    print(b, e)
    height = 50
    width = 75
    channels = 3
    frames = 30

    base_path = PP.jester_data
    new_path = os.path.join(PP.jester_location, 'data_50_75')

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    all_videos = os.listdir(base_path)
    all_videos.sort()

    for vid in tqdm.tqdm(range(b, e)):
        vid_path = os.path.join(base_path, all_videos[vid])
        new_vid_path = os.path.join(new_path, all_videos[vid])
        if not os.path.exists(new_vid_path):
            os.mkdir(new_vid_path)

        all_frames = os.listdir(vid_path)
        # new_video = np.zeros(shape=(num_fames, channels, height, width), dtype=int)

        num_frames = len(all_frames)
        if num_frames < frames:
            missing_frames = frames - num_frames
            dupl_1 = [0 for n in range(missing_frames // 2)]
            dupl_2 = [num_frames-1 for n in range(missing_frames - missing_frames // 2)]
            dupl_mid = [n for n in range(num_frames)]
            frames_to_copy = dupl_1 + dupl_mid + dupl_2
            assert len(frames_to_copy) == frames

        elif num_frames > frames:
            frames_to_remove = [n for n in range(0, num_frames, num_frames // (num_frames - frames))]
            leftover = num_frames - len(frames_to_remove)

            if leftover < frames:
                random_indices = random.sample(frames_to_remove, k=(frames - leftover))
                for n in random_indices:
                    frames_to_remove.remove(n)

                assert num_frames - len(frames_to_remove) == frames

            elif leftover > frames:
                print('leftover > frames')

            frames_to_copy = [n for n in range(num_frames)]
            for n in frames_to_remove:
                frames_to_copy.remove(n)

            assert len(frames_to_copy) == frames

        else:
            frames_to_copy = [n for n in range(num_frames)]

        cntr = 0
        for i in frames_to_copy:
            frame_path = os.path.join(vid_path, all_frames[i])

            frame = Image.open(frame_path)
            frame = adjust_frame(frame, height, width, channels)

            frame_name = '%05d.jpg' % cntr

            new_frame_path = os.path.join(new_vid_path, frame_name)
            frame = Image.fromarray(frame, mode='RGB')
            frame.save(new_frame_path)
            cntr = cntr + 1

# 148092
# DONE  standardize_clips(0, 3)
# standardize_clips(3, 10000)
# standardize_clips(10000, 20000)
# standardize_clips(20000, 30000)
# standardize_clips(30000, 40000)
# standardize_clips(40000, 50000)
# standardize_clips(50000, 60000)
# standardize_clips(60000, 70000)
# standardize_clips(70000, 80000)
# standardize_clips(80000, 90000)
# standardize_clips(90000, 100000)
# standardize_clips(100000, 110000)
# standardize_clips(110000, 120000)
# standardize_clips(120000, 130000)
# standardize_clips(130000, 140000)
# standardize_clips(140000, 148092)


def triple_check_num_frames_in_folders():
    path = PP.jester_data_50_75
    save_path = os.path.join(PP.jester_location, 'missing_frames.txt')

    all_folders = os.listdir(path)
    all_folders.sort()
    for i in tqdm.tqdm(range(len(all_folders))):
        p1 = os.path.join(path, all_folders[i])
        num_frames = wc_l(p1)
        if num_frames < 30:
            print(all_folders[i], num_frames)
            with open(save_path, 'a') as my_file:
                my_file.write('%s,%d\n' % (all_folders[i], num_frames))


def redo_folders_with_few_frames():
    folders_missing_frames = os.path.join(PP.jester_data, 'missing_frames.txt')
    folder_names = np.genfromtxt(folders_missing_frames, str, delimiter=',')[:0]

    base_path = PP.jester_data
    all_videos = os.listdir(base_path)
    all_videos.sort()

    for fn in folder_names:
        fn_index = all_videos.index(fn)

        folder_path = os.path.join(PP.jester_data_50_75, fn)
        # confirm that folder has less than 30 frames
        assert(wc_l(folder_path) < 30)

        # remove existing folder
        os.remove(folder_path)

        # create the folder again
        standardize_clips(fn_index, fn_index+1)

        # check that the folder has the correct number of frames
        assert(wc_l(folder_path) == 30)



