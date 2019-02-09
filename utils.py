import os
import numpy as np


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


def get_avg_body_bbox():

    poses_path = '/scratch/users/gabras/data/omg_empathy/saving_data/poses'

    which = ['Training', 'Validation']

    body_bbox = [10000, 0, 10000, 0]

    # body_bbox = np.zeros((1, 4))

    for dataset in which:
        folder_path = os.path.join(poses_path, dataset)
        txt_files = os.listdir(folder_path)

        for txt in txt_files:
            txt_path = os.path.join(folder_path, txt)
            bbox = pose_to_bbox(txt_path)

            if bbox is not None:
                if bbox[0] < body_bbox[0]: body_bbox[0] = bbox[0]
                if bbox[1] > body_bbox[1]: body_bbox[1] = bbox[1]
                if bbox[2] < body_bbox[2]: body_bbox[2] = bbox[2]
                if bbox[3] > body_bbox[3]: body_bbox[3] = bbox[3]

    return body_bbox


# bb = get_avg_body_bbox()
