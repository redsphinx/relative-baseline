import os
import numpy as np
from PIL import Image
import cv2
import sys


# use when txt_path only contains (x, y) poses
def pose_to_bbox_old(txt_path):
    try:
        poses_array = np.genfromtxt(txt_path, 'str')
        # poses_array = np.genfromtxt(txt_path, 'str')
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


def single_pose_to_bbox(person_pose):
    unit_length = 80

    top_joint_priority = [4, 5, 6, 12, 16, 7, 13, 17, 8, 10, 14, 9, 11, 15, 2, 3, 0, 1, sys.maxsize]
    bottom_joint_priority = [9, 6, 7, 14, 16, 8, 15, 17, 4, 2, 0, 5, 3, 1, 10, 11, 12, 13, sys.maxsize]

    top_joint_index = len(top_joint_priority) - 1
    bottom_joint_index = len(bottom_joint_priority) - 1
    left_joint_index = 0
    right_joint_index = 0
    top_pos = sys.maxsize
    bottom_pos = 0
    left_pos = sys.maxsize
    right_pos = 0

    for i, joint in enumerate(person_pose):
        joint = [int(float(joint[0])), int(float(joint[1])), int(float(joint[2]))]

        if joint[2] > 0:
            if top_joint_priority[i] < top_joint_priority[top_joint_index]:
                top_joint_index = i
            elif bottom_joint_priority[i] < bottom_joint_priority[bottom_joint_index]:
                bottom_joint_index = i
            if joint[1] < top_pos:
                top_pos = joint[1]
            elif joint[1] > bottom_pos:
                bottom_pos = joint[1]

            if joint[0] < left_pos:
                left_pos = joint[0]
                left_joint_index = i
            elif joint[0] > right_pos:
                right_pos = joint[0]
                right_joint_index = i

    top_padding_radio = [0.9, 1.9, 1.9, 2.9, 3.7, 1.9, 2.9, 3.7, 4.0, 5.5, 7.0, 4.0, 5.5, 7.0, 0.7, 0.8, 0.7, 0.8]
    bottom_padding_radio = [6.9, 5.9, 5.9, 4.9, 4.1, 5.9, 4.9, 4.1, 3.8, 2.3, 0.8, 3.8, 2.3, 0.8, 7.1, 7.0, 7.1, 7.0]

    left = int(left_pos - 0.3 * unit_length)
    right = int(right_pos + 0.3 * unit_length)
    top = int(top_pos - top_padding_radio[top_joint_index] * unit_length)
    bottom = int(bottom_pos + bottom_padding_radio[bottom_joint_index] * unit_length)
    bbox = (left, top, right, bottom)

    return bbox


def max_bbox(txt_path):
    try:
        poses_array = np.genfromtxt(txt_path, 'str', delimiter=',')
    except UnicodeDecodeError:
        print(txt_path)
        return None
    
    bbox = [10000, 10000, 0, 0]
    
    all_names = poses_array[:, 0]

    person_pose = poses_array[:, 1:].reshape((300, 18, 3))
    
    for f in range(person_pose.shape[0]):
        single_pose = person_pose[f]
        single_bbox = single_pose_to_bbox(single_pose)
        if single_bbox[0] < bbox[0]: bbox[0] = single_bbox[0]  # x1
        if single_bbox[1] < bbox[1]: bbox[1] = single_bbox[1]  # y1
        if single_bbox[2] > bbox[2]: bbox[2] = single_bbox[2]  # x2
        if single_bbox[3] > bbox[3]: bbox[3] = single_bbox[3]  # y2

    return bbox
    

def get_avg_body_bbox():

    # poses_path = '/scratch/users/gabras/data/omg_empathy/saving_data/poses'
    poses_path = '/scratch/users/gabras/data/omg_empathy/saving_data/poses_fixed'

    which = ['Training', 'Validation']

    body_bbox = [10000, 10000, 0, 0]

    # body_bbox = np.zeros((1, 4))

    for dataset in which:
        folder_path = os.path.join(poses_path, dataset)
        txt_files = os.listdir(folder_path)

        for txt in txt_files:
            txt_path = os.path.join(folder_path, txt)
            bbox = max_bbox(txt_path)
            # bbox = pose_to_bbox(txt_path)

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


def find_change_points(which, subject, story):
    name = 'Subject_%d_Story_%d' % (subject, story)
    path = '/scratch/users/gabras/data/omg_empathy/%s' % which

    full_name = os.path.join(path, 'Annotations', name + '.csv')
    all_labels = np.genfromtxt(full_name, dtype=np.float32, skip_header=True)

    change_points = []

    for l in range(len(all_labels)-1):

        f1 = all_labels[l]
        f2 = all_labels[l+1]

        if f2 - f1 != 0:
            change_points.append([l, l+1])

    print('number of change_points in %s: %d' % (name, len(change_points)))

    return change_points


for i in range(1, 11):
    cp = find_change_points('Training', i, 2)
    cp = find_change_points('Training', i, 4)
    cp = find_change_points('Training', i, 5)
    cp = find_change_points('Training', i, 8)
