import numpy as np
import os
from PIL import Image, ImageDraw

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


for i in range(1, 15):
    for j in range(1, 3):
        for k in range(1, 21):
            for l in range(1, 6):
                extract_hand_from_1_sequence(i, j, k, l)