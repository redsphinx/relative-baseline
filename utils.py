import os
import numpy as np
from PIL import Image
import cv2


def pose_to_bbox(txt_path):
    try:
        poses_array = np.genfromtxt(txt_path, 'str')
    except UnicodeDecodeError:
        print(txt_path)
        return None

    frame_pose = []

    for line in range(len(poses_array)):
        _tmp = poses_array[line].split(',')[1:]
        _tmp = [float(_tmp[i]) for i in range(len(_tmp))]
        frame_pose.extend(_tmp)

    odds = [i for i in range(len(frame_pose)) if i % 2 == 1]
    evens = list(set([i for i in range(len(frame_pose))]) - set(odds))
    evens.sort()

    all_x = np.array(list(set(np.array(frame_pose)[evens]) - {0.0}), dtype=int)
    all_y = np.array(list(set(np.array(frame_pose)[odds]) - {0.0}), dtype=int)

    # TODO: add extra space for top of head
    min_x = np.min(all_x)
    max_x = np.max(all_x)
    min_y = np.min(all_y)
    max_y = np.max(all_y)

    return [min_x, max_x, min_y, max_y]


# TODO: fix the x, y, v stuff
def get_avg_body_bbox():

    poses_path = '/scratch/users/gabras/data/omg_empathy/saving_data/poses'

    which = ['Training', 'Validation']

    body_bbox = [10000, 10000, 0, 0]

    # body_bbox = np.zeros((1, 4))

    for dataset in which:
        folder_path = os.path.join(poses_path, dataset)
        txt_files = os.listdir(folder_path)

        for txt in txt_files:
            txt_path = os.path.join(folder_path, txt)
            # MOD: add the fix here such that pose_to_bbox can remain unchanged
            bbox = pose_to_bbox(txt_path)

            if bbox is not None:
                if bbox[0] < body_bbox[0]: body_bbox[0] = bbox[0]  # x1
                if bbox[1] < body_bbox[1]: body_bbox[1] = bbox[1]  # y1
                if bbox[2] > body_bbox[2]: body_bbox[2] = bbox[2]  # x2
                if bbox[3] > body_bbox[3]: body_bbox[3] = bbox[3]  # y2

    return body_bbox


def draw_points(grid, all_x, all_y, all_v, save_path, names):
    assert len(all_x) == len(all_y)
    colr = (0, 255, 0)  # green
    # canvas = Image.fromarray(grid)
    
    for i in range(len(all_x)):
        if all_v[i] != 0:
            grid = cv2.circle(grid, (all_x[i], all_y[i]), 3, colr, -1)
            # grid[all_x[i], all_y[i]] = colr
            # grid[all_x[i] + 1, all_y[i]] = colr
            # grid[all_x[i], all_y[i] + 1] = colr
            # grid[all_x[i] + 1, all_y[i] + 1] = colr
            # grid[all_x[i] - 0, all_y[i]] = colr
            # grid[all_x[i], all_y[i] - 0] = colr
            # grid[all_x[i] - 0, all_y[i] - 0] = colr

    img = Image.fromarray(grid, 'RGB')
    img_save_path = os.path.join(save_path, names)
    img.save(img_save_path)
    # canvas.save(img_save_path)
    
# TODO: repeat this with the new poses
def save_bbox_image():
    img_folder = '/scratch/users/gabras/data/omg_empathy/Validation/jpg_participant_1280_720/Subject_10_Story_1'
    poses_path = '/scratch/users/gabras/data/omg_empathy/saving_data/poses/Validation/Subject_10_Story_1.txt'

    poses_array = np.genfromtxt(poses_path, 'str')

    num_imgs = 100

    for n in range(num_imgs):
        jpg_name = poses_array[n].split(',')[0]
        img_path = os.path.join(img_folder, jpg_name)
        img_arr = np.array(Image.open(img_path), dtype=np.uint8)

        frame_pose = []
        _tmp = poses_array[n].split(',')[1:]
        _tmp = [float(_tmp[i]) for i in range(len(_tmp))]
        frame_pose.extend(_tmp)

        x_nums = [i for i in range(len(frame_pose)) if i % 3 == 0]
        y_nums = [x_nums[i]+1 for i in range(len(x_nums))]
        v_nums = [y_nums[i] + 1 for i in range(len(y_nums))]

        # odds = [i for i in range(len(frame_pose)) if i % 2 == 1]
        # evens = list(set([i for i in range(len(frame_pose))]) - set(odds))
        # evens.sort()

        # all_x = np.array(frame_pose, dtype=int)[evens]
        # all_y = np.array(frame_pose, dtype=int)[odds]
        all_x = np.array(frame_pose, dtype=int)[x_nums]
        all_y = np.array(frame_pose, dtype=int)[y_nums]
        all_v = np.array(frame_pose, dtype=int)[v_nums]

        save_path = '/scratch/users/gabras/data/omg_empathy/saving_data/testing_pose_extraction'

        draw_points(img_arr, all_x, all_y, all_v, save_path, jpg_name)


# save_bbox_image()
