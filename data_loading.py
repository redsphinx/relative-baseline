import numpy as np
import os
import subprocess
import random
from deepimpression2.chalearn20 import poisson_disc as pd
from PIL import Image


def wc_l(file_name):
    command = "cat %s | wc -l" % file_name
    frames = subprocess.check_output(command, shell=True).decode('utf-8')
    frames = int(frames)
    return frames - 1
    # return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')


def make_frame_matrix():
    # TODO: add frames for testing
    path = '/scratch/users/gabras/data/omg_empathy/'
    _shape = (10, 8)  # subject, story
    frame_matrix = np.zeros(_shape, dtype=int)

    valid_story_idx_train = [2, 4, 5, 8]
    valid_story_idx_val = [1]
    valid_story_idx_test = [3, 6, 7]

    data_folders = os.listdir(path)
    for f in data_folders:
        annotation_path = os.path.join(path, f, 'Annotations')
        if os.path.exists(annotation_path):
            csv_list = os.listdir(annotation_path)
            for csv_file in csv_list:
                csv_path = os.path.join(annotation_path, csv_file)
                subject = int(csv_file.split('_')[1]) - 1  # -1 because array idx at 0
                story = int(csv_file.split('_')[-1].split('.')[0]) - 1
                frames = wc_l(csv_path)
                frame_matrix[subject][story] = frames

    valid_story_idx = [valid_story_idx_train, valid_story_idx_val, valid_story_idx_test]

    return frame_matrix, valid_story_idx


def dummy_load_data():
    #  data = np.zeros((len(keys), 3, C2.H, C2.W), dtype=np.float32)
    num_samples = 10
    label = np.array([random.randint(-1, 1) for i in range(num_samples)], dtype=np.float32)
    label = np.expand_dims(label, -1)

    img_left = np.random.randint(low=0, high=255, size=(num_samples, 3, 320, 640)).astype(np.float32)
    img_right = np.random.randint(low=0, high=255, size=(num_samples, 3, 320, 640)).astype(np.float32)

    return label, np.array([img_left, img_right], dtype=np.float32)


# makes pairs with 2 random people across different stories
def get_left_right_pair_random_person(val_idx, frame_matrix, batch_size=32):
    if len(val_idx) == 4:
        _r = 4.6
    elif len(val_idx) == 1:
        _r = 1

    _len = 0
    while _len not in range(batch_size, batch_size+3):
        samples = pd.poisson_disc_samples(10*len(val_idx), 10*len(val_idx), r=_r)
        _len = len(samples)

    if _len != batch_size:
        samples = samples[:batch_size]

    def convert(p):
        p1, p2 = p
        subj_1 = p1 / len(val_idx) + 1
        subj_2 = p2 / len(val_idx) + 1
        story_1 = val_idx[p1 % len(val_idx)]
        story_2 = val_idx[p2 % len(val_idx)]
        name_1 = 'Subject_%d_Story_%d' % (subj_1, story_1)
        name_2 = 'Subject_%d_Story_%d' % (subj_2, story_2)
        return [name_1, int(subj_1)-1, story_1-1], [name_2, int(subj_2)-1, story_2-1]

    left_all = []
    right_all = []

    for i in samples:
        left, right,= convert(i)
        # get total frames
        left_num_frames = frame_matrix[left[1]][left[2]]
        right_num_frames = frame_matrix[right[1]][right[2]]
        # grab a random frame
        left_frame = random.randint(0, left_num_frames - 1) # jpgs start at 0.jpg
        right_frame = random.randint(0, right_num_frames - 1)

        left_all.append('%s/%d.jpg' % (left[0], left_frame))
        right_all.append('%s/%d.jpg' % (right[0], right_frame))

    zips = list(zip(left_all, right_all))
    random.shuffle(zips)
    left_all, right_all = zip(*zips)

    return left_all, right_all


# makes pairs with the same person across different stories
def get_left_right_pair_same_person(val_idx, frame_matrix, batch_size=32):
    num_subjects = 10
    sample_per_person = int(batch_size / num_subjects)

    left_all = []
    right_all = []

    def make_pairs(subject_number, left, right, spp):

        sample_idx = [random.randint(0, 3) for i in range(2 * spp)]
        stories = [val_idx[sample_idx[i]] - 1 for i in range(len(sample_idx))]
        frames = [random.randint(0, frame_matrix[sub][stories[i]] - 1) for i in range(len(sample_idx))]
        sample_names = ['Subject_%d_Story_%d/%d.jpg' % (subject_number+1, stories[i]+1, frames[i]) for i in range(len(sample_idx))]
        for i in range(len(sample_names)//2):
            left.append(sample_names[i])
        for i in range(len(sample_names) // 2, len(sample_names)):
            right.append(sample_names[i])
        return left, right

    for sub in range(num_subjects):
        left_all, right_all = make_pairs(sub, left_all, right_all, sample_per_person)

    # add 2 extras to reach batchsize 32
    leftovers = batch_size - num_subjects * sample_per_person

    for _l in range(leftovers):
        sub = random.randint(0, 9)
        left_all, right_all = make_pairs(sub, left_all, right_all, spp=1)

    zips = list(zip(left_all, right_all))
    random.shuffle(zips)
    left_all, right_all = zip(*zips)

    return left_all, right_all


def cat_head_tail(fname, frame_num):
    command = "cat %s | head -n %d | tail -n 1" % (fname, frame_num)
    value = subprocess.check_output(command, shell=True).decode('utf-8')
    return value


def get_valence(which, full_name):
    name = full_name.split('/')[0]
    frame = int(full_name.split('/')[-1].split('.jpg')[0]) + 2 # lines start at 1 + skip first line

    if which == 'train':
        csv_path = os.path.join('/scratch/users/gabras/data/omg_empathy/Training/Annotations', name + '.csv')
    elif which == 'val':
        csv_path = os.path.join('/scratch/users/gabras/data/omg_empathy/Validation/Annotations', name + '.csv')
    elif which == 'test':
        raise NotImplemented

    try:
        valence = float(cat_head_tail(csv_path, frame))
    except ValueError:
        print(name, frame)
        print(ValueError)

    return valence


def load_data(which, frame_matrix, val_idx, batch_size):
    if which == 'train':
        path = '/scratch/users/gabras/data/omg_empathy/Training/jpg_participant_640_360'
    elif which == 'val':
        path = '/scratch/users/gabras/data/omg_empathy/Validation/jpg_participant_640_360'
    elif which == 'test':
        path = '/scratch/users/gabras/data/omg_empathy/Test/jpg_participant_640_360'

    # left_all, right_all = get_left_right_pair_random_person(val_idx, frame_matrix, batch_size)
    left_all, right_all = get_left_right_pair_same_person(val_idx, frame_matrix, batch_size)

    left_data = np.zeros((batch_size, 3, 360, 640), dtype=np.float32)
    right_data = np.zeros((batch_size, 3, 360, 640), dtype=np.float32)

    labels = np.zeros((batch_size, 1), dtype=np.float32)

    assert len(left_all) == len(right_all)
    for i in range(len(left_all)):
        _tmp_labels = np.zeros(2)
        # get left data
        jpg_path = os.path.join(path, left_all[i])
        try:
            jpg = np.array(Image.open(jpg_path), dtype=np.float32).transpose((2, 0, 1))
        except FileNotFoundError:
            print(jpg_path)
            print(FileNotFoundError)
        left_data[i] = jpg
        # left valence
        _tmp_labels[0] = get_valence(which, left_all[i])

        # get right data
        jpg_path = os.path.join(path, right_all[i])
        jpg = np.array(Image.open(jpg_path), dtype=np.float32).transpose((2, 0, 1))
        right_data[i] = jpg
        # right valence
        _tmp_labels[1] = get_valence(which, right_all[i])

        diff = _tmp_labels[0] - _tmp_labels[1]
        if diff == 0:
            labels[i] = 0
        elif diff < 0:
            labels[i] = 1
        else:
            labels[i] = -1

    # labels = np.expand_dims(labels, -1)
    return left_data, right_data, labels


def update_logs(which, loss, epoch, model_num, experiment_number):
    path = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/%s/epochs/model_%d_experiment_%d.txt' \
           % (which, model_num, experiment_number)

    with open(path, 'a') as my_file:
        line = '%d,%f\n' % (epoch, loss)
        my_file.write(line)


f_mat, v_idx = make_frame_matrix()
l, r, lab = load_data('train', f_mat, v_idx[0], 32)
