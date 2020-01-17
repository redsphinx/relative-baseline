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


def extract_hand(gesture, finger, subject, essai):
