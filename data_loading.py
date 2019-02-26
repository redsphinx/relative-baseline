import numpy as np
import os
import subprocess
import random
from deepimpression2.chalearn20 import poisson_disc as pd
from PIL import Image
import utils as U


def wc_l(file_name):
    command = "cat %s | wc -l" % file_name
    frames = subprocess.check_output(command, shell=True).decode('utf-8')
    frames = int(frames)
    return frames - 1
    # return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')


def make_frame_matrix():
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
def get_left_right_pair_same_person(which, val_idx, frame_matrix, batch_size=32, seed=42):
    if which != 'train':
        random.seed(seed)
    else:
        random.seed()

    num_subjects = 10
    sample_per_person = int(batch_size / num_subjects)

    left_all = []
    right_all = []

    def make_pairs(subject_number, left, right, spp):
        if which == 'val':
            num = 0
        elif which == 'train':
            num = 3
        elif which == 'test':
            num = 2
        else:
            num = None
        sample_idx = [random.randint(0, num) for i in range(2 * spp)]
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

    if which == 'train':
        zips = list(zip(left_all, right_all))
        random.shuffle(zips)
        left_all, right_all = zip(*zips)

    return list(left_all), list(right_all)


def get_left_right_consecutively(which, subject, current_frame):
    if which == 'train':
        path = '/scratch/users/gabras/data/omg_empathy/Training/jpg_participant_662_542'
    elif which == 'val':
        path = '/scratch/users/gabras/data/omg_empathy/Validation/jpg_participant_662_542'
    elif which == 'test':
        path = '/scratch/users/gabras/data/omg_empathy/Test/jpg_participant_662_542'
    
    if current_frame != 0:
        left_path = os.path.join(path, subject, '%d.jpg' % (current_frame - 1))
        right_path = os.path.join(path, subject, '%d.jpg' % (current_frame))
        jpg_left = np.array(Image.open(left_path), dtype=np.float32).transpose((2, 0, 1))
        jpg_right = np.array(Image.open(right_path), dtype=np.float32).transpose((2, 0, 1))
    else:
        print('Something is wrong, frame with value %d should not be passed here' % current_frame)
        jpg_left = None
        jpg_right = None

    return jpg_left, jpg_right


# makes random pairs with the same person with 2 consecutive frames
def get_left_right_pair_same_person_consecutive(which, val_idx, frame_matrix, batch_size=32, step=0):
    if which != 'train':
        random.seed(42+step)
    else:
        random.seed()

    num_subjects = 10
    sample_per_person = int(batch_size / num_subjects)

    left_all = []
    right_all = []
    
    def make_pairs(subject_number, left, right, spp):
        if which == 'val':
            num = 0
        elif which == 'train':
            num = 3
        elif which == 'test':
            num = 2
        else:
            num = None
        sample_idx = [random.randint(0, num) for i in range(spp)]
        stories = [val_idx[sample_idx[i]] - 1 for i in range(len(sample_idx))]
        left_frames = [random.randint(0, frame_matrix[sub][stories[i]] - 2) for i in range(len(sample_idx))] # -2 because of randint and we do +1 for right_frame
        right_frames = [left_frames[i] + 1 for i in range(len(left_frames))]
        
        left_names = ['Subject_%d_Story_%d/%d.jpg' % (subject_number+1, stories[i]+1, left_frames[i]) for i in range(len(sample_idx))]
        right_names = ['Subject_%d_Story_%d/%d.jpg' % (subject_number+1, stories[i]+1, right_frames[i]) for i in range(len(sample_idx))]

        for i in range(len(left_names)):
            left.append(left_names[i])
            right.append(right_names[i])
            
        return list(left), list(right)

    for sub in range(num_subjects):
        left_all, right_all = make_pairs(sub, left_all, right_all, sample_per_person)

    # add extras to reach batchsize
    leftovers = batch_size - num_subjects * sample_per_person

    for _l in range(leftovers):
        sub = random.randint(0, 9)
        left_all, right_all = make_pairs(sub, left_all, right_all, spp=1)

    if which == 'train':
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


def load_data_relative(which, frame_matrix, val_idx, batch_size, label_output='single', seed=42, label_mode='difference', data_mix='far', step=0, mode='default'):
    assert label_mode in ['difference', 'stepwise']
    assert data_mix in ['far', 'close', 'both', 'change_points']  # far = frames are >1 apart, close = frames are 1 apart
    assert label_output in ['single', 'double']
    assert mode in ['default', 'single']  # single=for validation on the same images using single frames (to compare with relative)

    if which == 'train':
        path = '/scratch/users/gabras/data/omg_empathy/Training/jpg_participant_662_542'
    elif which == 'val':
        path = '/scratch/users/gabras/data/omg_empathy/Validation/jpg_participant_662_542'
    elif which == 'test':
        path = '/scratch/users/gabras/data/omg_empathy/Test/jpg_participant_662_542'

    if data_mix == 'far':
        left_all, right_all = get_left_right_pair_same_person(which, val_idx, frame_matrix, batch_size, seed)
    elif data_mix == 'close':
        left_all, right_all = get_left_right_pair_same_person_consecutive(which, val_idx, frame_matrix, batch_size, step)
    elif data_mix == 'both':
        left_all_1, right_all_1 = get_left_right_pair_same_person(which, val_idx, frame_matrix, batch_size=batch_size//2)
        left_all_2, right_all_2 = get_left_right_pair_same_person_consecutive(which, val_idx, frame_matrix, batch_size=batch_size//2)
        left_all_1.extend(left_all_2)
        right_all_1.extend(right_all_2)
        zips = list(zip(left_all_1, right_all_1))
        random.shuffle(zips)
        left_all, right_all = zip(*zips)
    elif data_mix == 'change_points':
        change_points = U.get_all_change_points(which, val_idx)
        left_all_1, right_all_1 = get_left_right_pair_change_points(which, change_points, batch_size=batch_size // 2)
        left_all_2, right_all_2 = get_left_right_pair_same_person_consecutive(which, val_idx, frame_matrix,
                                                                              batch_size=batch_size // 2)
        left_all_1.extend(left_all_2)
        right_all_1.extend(right_all_2)
        zips = list(zip(left_all_1, right_all_1))
        random.shuffle(zips)
        left_all, right_all = zip(*zips)

    left_data = np.zeros((batch_size, 3, 542, 662), dtype=np.float32)
    right_data = np.zeros((batch_size, 3, 542, 662), dtype=np.float32)

    if mode == 'default':
        if label_output == 'single':
            if label_mode == 'difference':
                labels = np.zeros((batch_size, 1), dtype=np.float32)
            elif label_mode == 'stepwise':
                labels = np.zeros((batch_size, 3), dtype=np.float32)
        elif label_output == 'double':
            # labels_1 = np.zeros((batch_size, 2), dtype=int)  # classifications
            labels_1 = np.zeros((batch_size, 1), dtype=int)  # classifications
            labels_2 = np.zeros((batch_size, 1), dtype=np.float32)  # regression

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

            if label_output == 'single':
                if label_mode == 'difference':
                    labels[i]= _tmp_labels[1] - _tmp_labels[0]
                elif label_mode == 'stepwise':
                    # [-1, 0, 1] = right is lower, same, right is higher
                    diff = _tmp_labels[0] - _tmp_labels[1]
                    if diff == 0:
                        labels[i] = [0, 1, 0]
                    elif diff < 0:
                        labels[i] = [0, 0, 1]
                    else:
                        labels[i] = [1, 0, 0]
            elif label_output == 'double':
                diff = _tmp_labels[0] - _tmp_labels[1]
                # [a, b] where a = no change and b = change
                # [0, 1] = change
                # [1, 0] = no change
                if diff == 0:
                    # labels_1[i] = [1, 0]
                    labels_1[i] = 0
                else:
                    # labels_1[i] = [0, 1]
                    labels_1[i] = 1

                labels_2[i] = _tmp_labels[1] - _tmp_labels[0]

        if label_output == 'double':
            labels = [labels_1, labels_2]

    # for comparison with relative when using single frames
    else:
        labels = np.zeros((batch_size, 1), dtype=np.float32)

        for i in range(len(left_all)):
            # get only right data
            jpg_path = os.path.join(path, right_all[i])
            jpg = np.array(Image.open(jpg_path), dtype=np.float32).transpose((2, 0, 1))
            right_data[i] = jpg
            # right valence
            labels[i] = get_valence(which, right_all[i])


    # labels = np.expand_dims(labels, -1)
    return left_data, right_data, labels


def load_data_single(which, frame_matrix, val_idx, batch_size, step=0):
    if which == 'train':
        path = '/scratch/users/gabras/data/omg_empathy/Training/jpg_participant_662_542'
    elif which == 'val':
        path = '/scratch/users/gabras/data/omg_empathy/Validation/jpg_participant_662_542'
    elif which == 'test':
        path = '/scratch/users/gabras/data/omg_empathy/Test/jpg_participant_662_542'

    if which != 'train':
        random.seed(42+step)
    else:
        random.seed()

    num_subjects = 10
    sample_per_person = batch_size // num_subjects
    names = []

    def get_names(subject_number, spp):
        if which == 'val':
            num = 0
        elif which == 'train':
            num = 3
        elif which == 'test':
            num = 2
        else:
            num = None

        sample_idx = [random.randint(0, num) for i in range(spp)]
        stories = [val_idx[sample_idx[i]] - 1 for i in range(len(sample_idx))]
        frames = [random.randint(0, frame_matrix[sub][stories[i]] - 1) for i in range(len(sample_idx))]
        sample_names = ['Subject_%d_Story_%d/%d.jpg' % (subject_number+1, stories[i]+1, frames[i]) for i in range(len(sample_idx))]
        return sample_names

    for sub in range(num_subjects):
        names.extend(get_names(sub, sample_per_person))

    leftovers = batch_size - num_subjects * sample_per_person

    for _l in range(leftovers):
        sub = random.randint(0, 9)
        names.extend(get_names(sub, spp=1))

    if which == 'train':
        random.shuffle(names)

    data = np.zeros((batch_size, 3, 542, 662), dtype=np.float32)
    labels = np.zeros((batch_size, 1), dtype=np.float32)

    for i in range(len(names)):
        # get data
        jpg_path = os.path.join(path, names[i])
        try:
            jpg = np.array(Image.open(jpg_path), dtype=np.float32).transpose((2, 0, 1))
        except FileNotFoundError:
            print(jpg_path)
            print(FileNotFoundError)

        data[i] = jpg

        # get valence
        labels[i] = get_valence(which, names[i])

    # labels = np.expand_dims(labels, -1)
    return data, labels


def get_left_right_pair_change_points(which, change_points, batch_size):
    if which != 'train':
        random.seed(42)
    else:
        random.seed()

    # shape = np.shape(change_points)  # (10, 4, 2)

    num_subjects = 10
    sample_per_person = int(batch_size / num_subjects)
    left_all = []
    right_all = []

    def make_pairs(subject_number, left, right, spp):
        if which == 'val':
            num = 0
        elif which == 'train':
            num = 3
        elif which == 'test':
            num = 2
        else:
            num = None
        sample_idx = [random.randint(0, num) for i in range(spp)]

        for story in range(len(sample_idx)):
            pairs_left = change_points[subject_number][story][0]
            pairs_right = change_points[subject_number][story][1]

            assert len(pairs_left) == len(pairs_right)

            random_pair_num = random.randint(0, len(pairs_left)-1)
            left.append(pairs_left[random_pair_num])
            right.append(pairs_right[random_pair_num])

        return list(left), list(right)

    for sub in range(num_subjects):
        left_all, right_all = make_pairs(sub, left_all, right_all, sample_per_person)

    # add extras to reach batch_size
    leftovers = batch_size - num_subjects * sample_per_person

    for _l in range(leftovers):
        sub = random.randint(0, 9)
        left_all, right_all = make_pairs(sub, left_all, right_all, spp=1)

    if which == 'train':
        zips = list(zip(left_all, right_all))
        random.shuffle(zips)
        left_all, right_all = zip(*zips)

    return list(left_all), list(right_all)


def get_single_consecutively(which, subject, current_frame):
    if which == 'train':
        path = '/scratch/users/gabras/data/omg_empathy/Training/jpg_participant_662_542'
    elif which == 'val':
        path = '/scratch/users/gabras/data/omg_empathy/Validation/jpg_participant_662_542'
    elif which == 'test':
        path = '/scratch/users/gabras/data/omg_empathy/Test/jpg_participant_662_542'

    jpg_path = os.path.join(path, subject, '%d.jpg' % current_frame)
    jpg = np.array(Image.open(jpg_path), dtype=np.float32).transpose((2, 0, 1))

    return jpg


def update_logs(which, loss, epoch, model_num, experiment_number):
    path = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/%s/epochs/model_%d_experiment_%d.txt' \
           % (which, model_num, experiment_number)

    with open(path, 'a') as my_file:
        line = '%d,%f\n' % (epoch, loss)
        my_file.write(line)






# f_mat, v_idx = make_frame_matrix()
# print('e')
# l, r, lab = load_data('train', f_mat, v_idx[0], 32)

