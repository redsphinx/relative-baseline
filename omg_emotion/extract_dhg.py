import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import shutil
import random

from relative_baseline.omg_emotion import project_paths as PP


def rescale(old, min, max):
    return int((old - min) / (max - min) * 255)


def tes_image():
    test_path = os.path.join(PP.dhg_location, "gesture_1/finger_1/subject_1/essai_1/depth_1.png")
    test_image = Image.open(test_path)
    test_image = test_image.convert('RGB')

    test_data_coords_path = os.path.join(PP.dhg_location, "gesture_1/finger_1/subject_1/essai_1/skeleton_image.txt")
    test_data_coords = np.genfromtxt(test_data_coords_path, delimiter=" ", dtype=int)[0]

    test_data_depth_path = os.path.join(PP.dhg_location, "gesture_1/finger_1/subject_1/essai_1/skeleton_world.txt")
    test_data_depth = np.genfromtxt(test_data_depth_path, delimiter=" ", dtype=float)[0]

    # make data easy
    test_data_coords = test_data_coords.reshape((22, 2))
    test_data_depth = test_data_depth.reshape((22, 3))[:,2] # take last column
    min = test_data_depth.min()
    max = test_data_depth.max()

    # map coordinated on image
    output_path = "/home/gabras/deployed/relative_baseline/omg_emotion/test.png"
    d = ImageDraw.Draw(test_image)

    for i in range(22):
        x = test_data_coords[i][0]
        y = test_data_coords[i][1]
        depth_color = (255, rescale(test_data_depth[i], min, max), 0)
        d.point([(x, y)], fill=depth_color)


    test_image.save(output_path, "PNG")


def get_min_max_coords(coord_arr):
    left = min(coord_arr[:, 0])
    right = max(coord_arr[:, 0])
    top = min(coord_arr[:, 1])
    bot = max(coord_arr[:, 1])
    return left, top, right, bot


def add_margin(coords): # left, top, right, bot
    margin = 40
    new_coords = [0, 0, 0, 0]

    for i in range(2):
        if coords[i] - margin < 0:
            new_coords[i] = 0
        else:
            new_coords[i] = coords[i] - margin

    if coords[2] + margin > 639:
        new_coords[2] = 639
    else:
        new_coords[2] = coords[2] + margin
    
    if coords[3] + margin > 479:
        new_coords[3] = 479
    else:
        new_coords[3] = coords[3] + margin

    return new_coords


def clean_up_hand_only(img, coords):
    # TODO
    return img


def extract_hand_from_1_sequence(gesture, finger, subject, essai):
    base_path = PP.dhg_location

    path = os.path.join(base_path, 'gesture_%d/finger_%d/subject_%s/essai_%d' % (gesture, finger, subject, essai))
    coords_path = os.path.join(path, 'skeleton_image.txt')
    all_coords = np.genfromtxt(coords_path, delimiter=" ", dtype=int)

    num_imgs = len(os.listdir(path)) - 3

    save_path = os.path.join(PP.dhg_hand_only_28_28, 'gesture_%d/finger_%d/subject_%s/essai_%d'
                             % (gesture, finger, subject, essai))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(num_imgs):
        img_path = os.path.join(path, 'depth_%d.png' % (i + 1))

        img = Image.open(img_path)
        img = img.convert('RGB')
        coords = all_coords[i].reshape((22, 2))

        coords = get_min_max_coords(coords)
        coords_boundary = add_margin(coords)

        img = img.crop((coords_boundary[0], coords_boundary[1], coords_boundary[2], coords_boundary[3]))
        img = clean_up_hand_only(img, coords)
        img = img.resize((28, 28))

        img_path = os.path.join(save_path, 'depth_%d.png'% (i + 1))

        img.save(img_path, 'PNG')


def extract_whole_DHG():
    for i in range(1, 15):
        for j in range(1, 3):
            for k in range(1, 21):
                for l in range(1, 6):
                    extract_hand_from_1_sequence(i, j, k, l)


# split in train, val, test
def generate_labels():
    splits = ['train', 'val', 'test']

    for s in range(3):

        base_path = PP.dhg_hand_only_28_28
        label_txt_path = os.path.join(base_path, 'labels_%s.txt' % splits[s])

        if s == 0:
            kb, ke = 1, 15 # train
        elif s ==1:
            kb, ke = 15, 17 # val
        else:
            kb, ke = 17, 21 # test

        with open(label_txt_path, 'w') as my_file:
            for i in range(1, 15):
                for j in range(1, 3):
                    for k in range(kb, ke):
                        for l in range(1, 6):
                            line = '%d,%d,%d,%d\n' % (i, j, k, l)
                            my_file.write(line)


def create_difference_frames_graph(a_path, name):
    all_names = os.listdir(a_path)
    all_avg_diff = []
    x = [i for i in range(1, len(all_names))]

    for i in range(1, len(all_names)):
        current_img_path = os.path.join(a_path, all_names[i])
        current_img = Image.open(current_img_path)
        current_img = np.array(current_img.convert('L'))

        prev_img_path = os.path.join(a_path, all_names[i-1])
        prev_img = Image.open(prev_img_path)
        prev_img = np.array(prev_img.convert('L'))

        diff = current_img - prev_img
        all_avg_diff.append(np.mean(diff))

    all_avg_diff = np.array(all_avg_diff)
    all_avg_diff = all_avg_diff - np.mean(all_avg_diff)


    new_avg_diff = []

    for i in range(len(all_avg_diff)):
        if all_avg_diff[i] > 0:
            new_avg_diff.append(all_avg_diff[i])
        else:
            new_avg_diff.append(0)

    title = 'difference per frame %s' % name
    save_path = '/home/gabras/deployed/relative_baseline/omg_emotion/images'
    save_path = os.path.join(save_path, 'diff_frame_%s.png' % name)


    fig = plt.figure()
    # plt.plot(x, all_avg_diff)
    plt.plot(x, new_avg_diff)
    plt.title(title)
    plt.xlabel('frame_num')
    plt.ylabel('avg diff per pixel')
    plt.savefig(save_path)


# statistics
def get_statistics():
    # avg num frames per sequence
    base_path = PP.dhg_hand_only_28_28
    len_seq = []
    short = 0
    limit = 50


    for i in range(1, 15):
        for j in range(1, 3):
            for k in range(1, 21):
                for l in range(1, 6):
                    path = os.path.join(base_path, 'gesture_%d/finger_%d/subject_%s/essai_%d' % (i, j, k, l))
                    len_seq.append(len(os.listdir(path)))

                    if len_seq[-1] == 26:
                        print('short', [i, j, k, l])
                        create_difference_frames_graph(path, '%d_%d_%d_%d' % (i, j, k, l))

                    if len_seq[-1] == 280:
                        print('long', [i, j, k, l])
                        create_difference_frames_graph(path, '%d_%d_%d_%d' % (i, j, k, l))

                    if len(os.listdir(path)) < limit:
                        short += 1


    print('min num frames in seq:   %d\n'
          'max num frames in seq:   %d\n'
          'avg num frames in seq:   %d\n'
          'frames shorter than %d:  %d'
          % (min(len_seq), max(len_seq), int(np.mean(len_seq)), limit, short))


    # plot histogram num frames
    # fig = plt.figure()
    # save_path = '/home/gabras/deployed/relative_baseline/omg_emotion/images'
    # save_path = os.path.join(save_path, 'frames_distribution.jpg')
    # plt.hist(len_seq, bins=20)
    # plt.title('frames distribution')
    # plt.savefig(save_path)

"""
min num frames in seq:   26
max num frames in seq:   280
avg num frames in seq:   80
frames shorter than 50:  222
"""


# get_statistics()

def create_fixed_sequence(frames=50):
    base_path = PP.dhg_hand_only_28_28
    save_path = PP.dhg_hand_only_28_28_50_frames

    for i in range(1, 15):
        for j in range(1, 3):
            for k in range(1, 21):
                for l in range(1, 6):
                    path = os.path.join(base_path, 'gesture_%d/finger_%d/subject_%s/essai_%d' % (i, j, k, l))
                    num_frames = len(os.listdir(path))

                    if num_frames < frames:
                        missing_frames = frames - num_frames
                        # duplicate first frame and last frame
                        dupl_1 = [1 for n in range(missing_frames//2)]
                        dupl_2 = [num_frames for n in range(missing_frames-missing_frames//2)]
                        dupl_mid = [n+1 for n in range(num_frames)]
                        frames_to_copy = dupl_1 + dupl_mid + dupl_2

                        if len(frames_to_copy) != frames:
                            print('num frames not good')

                    elif num_frames > frames:
                        frames_to_remove = [n for n in range(1, num_frames, num_frames//(num_frames-frames))]
                        # frames_to_copy = [n for n in range(1, num_frames, num_frames//frames+1)]
                        leftover = num_frames - len(frames_to_remove)

                        if leftover < frames:
                            random_indices = random.sample(frames_to_remove, k=(frames-leftover))
                            for n in random_indices:
                                frames_to_remove.remove(n)

                            assert(num_frames - len(frames_to_remove) == frames)

                        elif leftover > frames:
                            pass

                        frames_to_copy = [n+1 for n in range(num_frames)]
                        for n in frames_to_remove:
                            frames_to_copy.remove(n)

                        if len(frames_to_copy) != frames:
                            print('num frames not good')

                    else:
                        frames_to_copy = [n + 1 for n in range(num_frames)]

                    # copy correct frames to new folder
                    new_path = os.path.join(save_path, 'gesture_%d/finger_%d/subject_%s/essai_%d' % (i, j, k, l))
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)

                    for n in range(len(frames_to_copy)):
                        src_path = os.path.join(path, 'depth_%d.png' % frames_to_copy[n])
                        dest_path = os.path.join(new_path, 'depth_%d.png' % (n+1))

                        shutil.copy(src_path, dest_path)



