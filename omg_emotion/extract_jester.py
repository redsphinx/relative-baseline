import numpy as np
import math
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
from relative_baseline.omg_emotion.xai_tools.paper_plots import find_top_xai_videos


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

    # less_than = 30
    less_than = 60

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


# get_information()

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
    image = image.resize((new_w, h), resample=Image.BICUBIC) # w, h

    if new_w > w:
        delta_w = (new_w - w) // 2
        delta_w_2 = w + delta_w
        image = image.crop((delta_w, 0, delta_w_2, h))  # l, u, r, d
    elif new_w < w:
        delta_w = (w - new_w) // 2
        image = np.array(image)
        pixel_mean = np.mean(np.mean(image, axis=0), axis=0)
        pixel_mean = np.array(pixel_mean, dtype=int)
        canvas = np.ones(shape=(h, w, c), dtype=np.uint8)
        canvas = canvas * pixel_mean
        # paste it
        canvas[:, delta_w:new_w+delta_w, :] = image
        image = canvas

    image = np.array(image, dtype=np.uint8)
    assert image.shape == (h, w, c)

    return image


def standardize_clips(b, e, he, wi, loc, video_list=None, frames=30, base_path=None):
    print(b, e)
    height = he # 50
    width = wi  # 75
    channels = 3
    # frames = 30

    new_path = os.path.join(PP.jester_location, loc)  # 'data_50_75')
    if base_path is None:
        base_path = PP.jester_data

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    if video_list is not None:
        b = 0
        e = len(video_list)
        all_videos = video_list

    else:
        all_videos = os.listdir(base_path)

    all_videos.sort()

    for vid in tqdm.tqdm(range(b, e)):
        # print('vid = ', vid)

        # business as usual:
        vid_path = os.path.join(base_path, all_videos[vid])
        new_vid_path = os.path.join(new_path, all_videos[vid])

        # to fi_x specific videos
        # vid_path = os.path.join(base_path, str(vid))
        # new_vid_path = os.path.join(new_path, str(vid))
        if not os.path.exists(new_vid_path):
            os.mkdir(new_vid_path)

        all_frames = os.listdir(vid_path)
        all_frames.sort()
        # new_video = np.zeros(shape=(num_fames, channels, height, width), dtype=int)

        num_frames = len(all_frames)
        if num_frames < frames:
            missing_frames = frames - num_frames
            dupl_1 = [0] * (missing_frames // 2)
            dupl_2 = [num_frames-1] * (missing_frames - missing_frames // 2)
            dupl_mid = [n for n in range(num_frames)]
            frames_to_copy = dupl_1 + dupl_mid + dupl_2
            assert len(frames_to_copy) == frames

        elif num_frames > frames:
            frames_to_remove = [n for n in range(0, num_frames, int(math.ceil(num_frames / (num_frames - frames))))]
            leftover = num_frames - len(frames_to_remove)

            if leftover < frames:
                random_indices = random.sample(frames_to_remove, k=(frames - leftover))
                for n in random_indices:
                    frames_to_remove.remove(n)

                assert num_frames - len(frames_to_remove) == frames

            elif leftover > frames:
                to_add = leftover - frames

                if to_add == 1:
                    frames_to_remove.append(frames_to_remove[-1]-1)
                else:
                    selection_list = [i for i in range(num_frames)]
                    tmp = []
                    ind = 0
                    while len(tmp) != num_frames:
                        tmp.append(selection_list.pop(ind))
                        if ind == 0:
                            ind = -1
                        else:
                            ind = 0

                    for i in range(len(tmp)):
                        if tmp[i] not in frames_to_remove:
                            selection_list.append(tmp[i])

                    for a_t in range(to_add):
                        frames_to_remove.append(selection_list[a_t])

                frames_to_remove.sort()


            frames_to_copy = [n for n in range(num_frames)]
            for n in frames_to_remove:
                frames_to_copy.remove(n)

            assert len(frames_to_copy) == frames

        else:
            frames_to_copy = [n for n in range(num_frames)]


        cntr = 1
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
# standardize_clips(0, 3, he=150, wi=224, loc='data_150_224')

# standardize_clips(0, 10000, he=150, wi=224, loc='data_150_224')
# standardize_clips(10000, 20000, he=150, wi=224, loc='data_150_224')
# standardize_clips(20000, 30000, he=150, wi=224, loc='data_150_224')
# standardize_clips(30000, 40000, he=150, wi=224, loc='data_150_224')

# standardize_clips(40000, 50000, he=150, wi=224, loc='data_150_224')
# standardize_clips(50000, 60000, he=150, wi=224, loc='data_150_224')
# standardize_clips(60000, 70000, he=150, wi=224, loc='data_150_224')
# standardize_clips(70000, 80000, he=150, wi=224, loc='data_150_224')

# standardize_clips(80000, 90000, he=150, wi=224, loc='data_150_224')
# standardize_clips(90000, 100000, he=150, wi=224, loc='data_150_224')
# standardize_clips(100000, 110000, he=150, wi=224, loc='data_150_224')
# standardize_clips(110000, 120000, he=150, wi=224, loc='data_150_224')

# standardize_clips(120000, 130000, he=150, wi=224, loc='data_150_224')
# standardize_clips(130000, 140000, he=150, wi=224, loc='data_150_224')
# standardize_clips(140000, 148092, he=150, wi=224, loc='data_150_224')

# standardize_clips(101138, 101138+1, he=224, wi=336, loc='data_150_224')
# standardize_clips(101139, 101139+1, he=224, wi=336, loc='data_150_224')

def triple_check_num_frames_in_folders():
    # path = PP.jester_data_50_75
    path = PP.jester_data_224_336
    # save_path = os.path.join(PP.jester_location, 'missing_frames.txt')
    save_path = os.path.join(PP.jester_location, 'missing_frames_224x336.txt')

    all_folders = os.listdir(path)
    all_folders.sort()
    for i in tqdm.tqdm(range(len(all_folders))):
        p1 = os.path.join(path, all_folders[i])
        num_frames = wc_l(p1)
        if num_frames != 30:
            print(all_folders[i], num_frames)
            with open(save_path, 'a') as my_file:
                my_file.write('%s,%d\n' % (all_folders[i], num_frames))

# triple_check_num_frames_in_folders()

def redo_folders_with_few_frames():
    folders_missing_frames = os.path.join(PP.jester_location, 'missing_frames.txt')
    folder_names = np.genfromtxt(folders_missing_frames, dtype=int, delimiter=',')[:, 0]

    base_path = PP.jester_data
    all_videos = os.listdir(base_path)
    all_videos.sort()

    tot_folders = len(folder_names)
    cntr = 1

    for fn in folder_names:
        fn_index = all_videos.index(str(fn))

        folder_path = os.path.join(PP.jester_data_50_75, str(fn))
        # confirm that folder has less than 30 frames
        if os.path.exists(folder_path):
            assert(wc_l(folder_path) < 30)

        # remove existing folder
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        # create the folder again
        standardize_clips(fn_index, fn_index+1)

        # check that the folder has the correct number of frames
        assert(wc_l(folder_path) == 30)

        print('%d/%d DONEEE' % (cntr, tot_folders))
        cntr = cntr + 1


def fix_file_names_in_folder(folder_path, num_frames=30):

    list_files = os.listdir(folder_path)
    list_files.sort()

    good_list = ['%05d.jpg' % i for i in range(1, num_frames+1)]

    if list_files != good_list:
        for i in range(1, len(list_files) + 1):
            name = int(list_files[i-1].split('.')[0])
            if name != i:
                name = '%05d.jpg' % i

                src = os.path.join(folder_path, list_files[i-1])
                dest = os.path.join(folder_path, name)
                shutil.move(src, dest)


def jpg_to_avi(jpg_folder, avi_dest):
    command = "ffmpeg -loglevel fatal -f image2 -i %s/%s.jpg %s" % (jpg_folder, '%05d', avi_dest)
    subprocess.call(command, shell=True)


def to_avi(which, b, e):
    print('start and finish: ', b, e)
    assert which in ['test', 'val', 'train', 'xai']

    if which == 'xai':
        jpg_path = PP.fast_jester_data_150_224_65_frames
        avi_path = PP.fast_jester_data_150_224_65_avi

        list_files_to_convert = os.listdir(jpg_path)
        list_files_to_convert = [os.path.join(jpg_path, i) for i in list_files_to_convert]
        b = 0
        e = len(list_files_to_convert)
    else:
        jpg_path = PP.fast_jester_data_150_224
        avi_path = PP.fast_jester_data_150_224_avi

        labels_path = os.path.join(PP.jester_location, 'labels_%s.npy' % which)
        labels = np.load(labels_path)
        labels = labels[b:e]

        list_files_to_convert = [os.path.join(jpg_path, i) for i in labels[:, 0]]

    if not os.path.exists(avi_path):
        os.mkdir(avi_path)

    for file_path in tqdm.tqdm(list_files_to_convert):

        fix_file_names_in_folder(file_path, num_frames=65)
        dest = os.path.join(avi_path, '%s.avi' % (file_path.split('/')[-1]))
        jpg_to_avi(file_path, dest)

# to_avi('xai', 0, 0)

# to_avi('val', 0, 3)
# to_avi('val', 3, 10000)  # 7393
# to_avi('train', 0, 10000)
# to_avi('train', 10000, 20000) # 2069


# to_avi('train', 20000, 30000)
# to_avi('train', 30000, 40000)
# to_avi('train', 40000, 50000)


# to_avi('train', 50000, 60000)
# to_avi('train', 60000, 70000)
# to_avi('train', 70000, 80000)


# to_avi('train', 80000, 90000)
# to_avi('train', 90000, 100000)
# to_avi('train', 100000, 110000)

# to_avi('train', 110000, 120000)
# to_avi('test', 0, 10000)


def create_file_list(which, dest, jester_data_path):
    assert which in ['test', 'val', 'train']

    # dest_path = os.path.join(PP.jester_location, 'filelist_%s.txt' % which)
    dest_path = os.path.join(PP.jester_location, 'filelist_%s_%s.txt' % (which, dest))

    labels_path = os.path.join(PP.jester_location, 'labels_%s.npy' % which)
    labels = np.load(labels_path)

    with open(dest_path, 'a') as my_file:
        for i in range(len(labels)):
            fname = os.path.join(os.path.join(jester_data_path, '%s.avi' % labels[i, 0]))
            # fname = os.path.join(os.path.join(PP.jester_data_50_75_avi, '%s.avi' % labels[i, 0]))
            the_label = int(labels[i, 1]) + 1

            # NOTE: there *HAS* to be a space between the image path and the label

            line = '%s %d\n' % (fname, the_label)
            # print(line)
            my_file.write(line)


# create_file_list('test', dest='224_336_fast', jester_data_path=PP.fast_jester_data_224_336_avi)
# create_file_list('test', dest='150_224_fast', jester_data_path=PP.fast_jester_data_150_224_avi)



def calculate_weights_for_loss(which='train'):
    assert which in ['test', 'val', 'train']

    weights = np.zeros(shape=27)
    # get the number of datapoints for each class

    labs = load_labels(which)[:, 1]

    for j in range(27):
        weights[j] = sum(labs == j)

    total = sum(weights)

    weights = weights / total

    weights = 1 / weights

    total = sum(weights)

    weights = weights / total

    print(weights)


def ffmpeg_clean(src, dest):
    # command = "ffmpeg -loglevel fatal -i %s -map 0 -map -0:a -c copy -y %s" % (src, dest)
    command = "ffmpeg -loglevel fatal -i %s -c copy -an %s" % (src, dest)
    subprocess.call(command, shell=True)


def clean_files(b, e, which):
    assert which in ['test', 'val', 'train']
    print('b: %d    e: %d' % (b, e))

    if not os.path.exists(PP.jester_data_50_75_avi_clean):
        os.mkdir(PP.jester_data_50_75_avi_clean)

    which_path = os.path.join(PP.jester_location, 'filelist_%s.txt' % which)
    all_paths = np.genfromtxt(which_path, dtype=str, delimiter=' ')[b:e, 0]

    for i in tqdm.tqdm(range(len(all_paths))):
        dest = os.path.join(PP.jester_data_50_75_avi_clean, all_paths[i].split('/')[-1])
        # print(dest)

        ffmpeg_clean(all_paths[i], dest)


# clean_files(0, 500, 'val')

def short_filelist():
    src = os.path.join(PP.jester_location, 'filelist_val.txt')
    dest = os.path.join(PP.jester_location, 'filelist_val_TEST.txt')


    lines = np.genfromtxt(src, dtype=str, delimiter=' ')

    for i in range(500):
        line = lines[i][0]
        name = line.split('/')[-1]
        path = os.path.join(PP.jester_data_50_75_avi_clean, name)
        mod_line = '%s %s\n' % (path, lines[i][1])

        with open(dest, 'a') as my_file:
            my_file.write(mod_line)

# short_filelist()

def short_balanced_selection(which, data_per_class=500):
    assert which in ['test', 'val', 'train']

    num_classes = 27

    all_lines = np.genfromtxt(os.path.join(PP.jester_location, 'filelist_%s.txt' % which), dtype=str, delimiter=' ')
    labels = all_lines[:, 1]

    chosen = []
    for i in range(num_classes):
        # need to start at 1
        indices = np.arange(len(all_lines))[labels == str(i+1)]
        chosen.extend(indices[:data_per_class])

    chosen_lines = all_lines[chosen]

    new_file = os.path.join(PP.jester_location, 'filelist_%s_%dperclass.txt' % (which, data_per_class))
    with open(new_file, 'a') as my_file:
        for i in range(len(chosen_lines)):
            # must be separated by space
            line = '%s %s\n' % (chosen_lines[i][0], chosen_lines[i][1])
            # print(line)
            my_file.write(line)


# short_balanced_selection('val', data_per_class=200)


# selection for the visualization of kernel parameters
def xai_selection():
    which = 'test'
    data_per_class = 10

    num_classes = 27
    all_lines = np.genfromtxt(os.path.join(PP.jester_location, 'filelist_%s_224_336_fast.txt' % which), dtype=str, delimiter=' ')
    labels = all_lines[:, 1]

    chosen = []
    for i in range(num_classes):
        # need to start at 1
        indices = np.arange(len(all_lines))[labels == str(i + 1)]
        chosen.extend(indices[:data_per_class])

    chosen_lines = all_lines[chosen]

    new_file = os.path.join(PP.jester_location, 'filelist_%s_xai.txt' % (which))
    with open(new_file, 'a') as my_file:
        for i in range(len(chosen_lines)):
            # must be separated by space
            line = '%s %s\n' % (chosen_lines[i][0], chosen_lines[i][1])
            # print(line)
            my_file.write(line)

# xai_selection()


def xai_150_224_selection():
    which = 'test'
    data_per_class = 10

    num_classes = 27
    all_lines = np.genfromtxt(os.path.join(PP.jester_location, 'filelist_%s_150_224_fast.txt' % which), dtype=str, delimiter=' ')
    labels = all_lines[:, 1]

    chosen = []
    for i in range(num_classes):
        # need to start at 1
        indices = np.arange(len(all_lines))[labels == str(i + 1)]
        chosen.extend(indices[:data_per_class])

    chosen_lines = all_lines[chosen]

    new_file = os.path.join(PP.jester_location, 'filelist_%s_xai_150_224.txt' % (which))
    with open(new_file, 'a') as my_file:
        for i in range(len(chosen_lines)):
            # must be separated by space
            line = '%s %s\n' % (chosen_lines[i][0], chosen_lines[i][1])
            # print(line)
            my_file.write(line)


def longer_videos(dataset, number_videos=20, total_frames=65, prediction_type='correct'):
    '''
    get list of og videos locations
    only choose the xai ones
    standardize clip to (total_frames, 150, 224)
    save them
    '''

    xai_videos = find_top_xai_videos(dataset, prediction_type, combine=True, model=[23, 25])
    xai_videos = [i for i in xai_videos.keys()]
    xai_video_list = []

    if number_videos == 'all':
        number_videos = len(xai_videos)

    for v in range(number_videos):
        name = xai_videos[v].split('/')[-1].split('.')[0]
        # og_vid_path = os.path.join(base_path, name)
        xai_video_list.append(name)

    # save as frames
    standardize_clips(b=0, e=0, he=150, wi=224, video_list=xai_video_list,
                      loc='data_xai_150_224_%d_frames' % total_frames, frames=total_frames)

# longer_videos(dataset='jester', number_videos='all')

