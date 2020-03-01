import os
import numpy as np
import relative_baseline.omg_emotion.project_paths as PP
from relative_baseline.omg_emotion import utils as U1
import random
from PIL import Image
import cv2
# from time import time
import torch
from multiprocessing import Pool, Queue
from tqdm import tqdm
# temporary for debugging
# from .settings import ProjectVariable
import time
import torchvision.datasets as datasets
import torchvision

# arousal,valence
# Training: {0: 262, 1: 96, 2: 54, 3: 503, 4: 682, 5: 339, 6: 19}
# Validation: {0: 51, 1: 34, 2: 17, 3: 156, 4: 141, 5: 75, 6: 7}
# Test: {0: 329, 1: 135, 2: 50, 3: 550, 4: 678, 5: 231, 6: 16}
# 0 - Anger
# 1 - Disgust
# 2 - Fear
# 3 - Happy
# 4 - Neutral
# 5 - Sad
# 6 - Surprise

global_queue = Queue()
# pass q to everything
# put things in q
# q.put()
# when q is filled
# q.get()

def get_data(which, everything):
    ind = everything[0].index(which)
    return everything[1][ind]


def get_labels(which, everything):
    ind = everything[0].index(which)
    return everything[2][ind]


def load_labels(which, project_variable):
    # project_variable = ProjectVariable
    assert which in ['Training', 'Validation', 'Test']
    path = os.path.join(PP.data_path, which, 'Annotations', 'annotations.csv')
    annotations = np.genfromtxt(path, delimiter=',', dtype=str)

    if project_variable.debug_mode:
        annotations = annotations[0:project_variable.batch_size]
    if project_variable.train:
        np.random.shuffle(annotations)

    names = np.array(annotations[:, 0:2])
    arousal = annotations[:, 2]
    valence = annotations[:, 3]
    categories = annotations[:, -1]

    labels = [names]

    for i in project_variable.label_type:
        if i == 'arousal':
            arousal = U1.str_list_to_num_arr(arousal, float)
            labels.append(arousal)
        if i == 'valence':
            valence = U1.str_list_to_num_arr(valence, float)
            labels.append(valence)
        if i == 'categories':
            categories = U1.str_list_to_num_arr(categories, int)
            labels.append(categories)

    # labels = [names, arousal, valence, categories]
    return labels


def get_nonzero_frame(frames, utterance_path, cnt):
    index = random.randint(0, len(frames) - 1)
    jpg_path = os.path.join(utterance_path, '%d.jpg' % index)

    # left_data = np.zeros((batch_size, 3, 542, 662), dtype=np.float32)
    jpg_as_arr = Image.open(jpg_path)
    if jpg_as_arr.width != 1280 or jpg_as_arr.height != 720:
        jpg_as_arr = jpg_as_arr.resize((1280, 720))
        cnt += 1
    # ValueError: could not broadcast input array from shape (1280,3,720) into shape (3,720,1280)

    jpg_as_arr = np.array(jpg_as_arr, dtype=np.float32).transpose((2, 0, 1))

    return jpg_as_arr, cnt


def get_image(things):
    utterance_path = things[0]
    index = things[1]

    frames = os.listdir(utterance_path)
    cnt = 0

    jpg_as_arr, cnt = get_nonzero_frame(frames, utterance_path, cnt)

    # scale between 0 and 1 for resnet18
    while int(np.max(jpg_as_arr)) == 0:
        print('max is zero')
        jpg_as_arr, cnt = get_nonzero_frame(frames, utterance_path, cnt)

    # jpg_as_arr /= int(np.max(jpg_as_arr))
    try:
        jpg_as_arr[0] = jpg_as_arr[0] / np.max(jpg_as_arr[0])
        jpg_as_arr[1] = jpg_as_arr[1] / np.max(jpg_as_arr[1])
        jpg_as_arr[2] = jpg_as_arr[2] / np.max(jpg_as_arr[2])
    except RuntimeWarning:
        print('channel has a max of 0')
        return

    import torchvision.transforms as transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    jpg_as_arr = torch.from_numpy(jpg_as_arr)
    jpg_as_arr = normalize(jpg_as_arr)
    jpg_as_arr = jpg_as_arr.numpy()

    global_queue.put([jpg_as_arr, index])


# def parallel_load(path_list, index_list, number_processes=20):
def parallel_load(items, number_processes=20):
    func = get_image
    pool = Pool(processes=number_processes)
    pool.apply_async(func)
    pool.map(func, items)
    # pool.close()
    return pool


def load_omg_emotion(project_variable, seed):
    splits = []
    all_labels = []
    all_data = []
    tp = np.float32
    frames = project_variable.load_num_frames # should be 60 for omg_emotions

    def load(which, dp):
        if which == 'train':
            folder_name = 'Training'
        elif which == 'val':
            folder_name = 'Validation'
        elif which == 'test':
            folder_name = 'Test'
        else:
            print('problem with variable which')
            folder_name = None

        path = os.path.join(PP.data_path, folder_name, PP.omg_emotion_jpg_face)
        label_path = os.path.join(PP.data_path, folder_name, 'easy_labels.txt')
        all_labels = np.genfromtxt(label_path, delimiter=',', dtype=str)[:dp]
        data_names = all_labels[:, 0:2]

        if project_variable.label_type == 'categories':
            labels = all_labels[:, 5].astype(int)
        elif project_variable.label_type == 'arousal':
            labels = all_labels[:, 3].astype(int)
        elif project_variable.label_type == 'valence':
            labels = all_labels[:, 4].astype(int)
        else:
            print('label type not valid')
            labels = None

        num_points = len(labels)

        data = np.zeros(shape=(num_points, 3, frames, 96, 96), dtype=tp) # for cropped faces
        # data = np.zeros(shape=(num_points, 3, frames, 720, 1280), dtype=tp)

        for i in tqdm(range(num_points)):
            num_frames = int(all_labels[i][2])
            if num_frames < frames:
                new_list = [j_ for j_ in range(num_frames)]
                times = frames // num_frames + 1
                new_list = np.tile(new_list, times)
                new_list = new_list[:frames]

                assert(len(new_list) == frames)

                for j in range(frames):
                    frame_path = os.path.join(path, data_names[i][0], data_names[i][1], '%d.jpg' % new_list[j])
                    tmp = np.array(Image.open(frame_path))
                    tmp = tmp.transpose((2, 0, 1))
                    data[i, :, j] = tmp
            else:
                for j in range(frames):
                    frame_path = os.path.join(path, data_names[i][0], data_names[i][1], '%d.jpg' % j)
                    tmp = np.array(Image.open(frame_path))
                    tmp = tmp.transpose((2, 0, 1))
                    data[i, :, j] = tmp

        return data, labels

    def load_random(which, dp, balanced, seed):
        # balancing of classes is only implemented for categorical labels
        assert(project_variable.label_type == 'categories')
        if balanced:
            num_categories = 7
            assert (dp % num_categories == 0)

        total_dp = {'train': 1955, 'val': 481, 'test': 1989}

        if which == 'train':
            folder_name = 'Training'
        elif which == 'val':
            folder_name = 'Validation'
        elif which == 'test':
            folder_name = 'Testing'
        else:
            print('problem with variable which')
            folder_name = None

        path = os.path.join(PP.data_path, folder_name, PP.omg_emotion_jpg_face)
        label_path = os.path.join(PP.data_path, folder_name, 'easy_labels.txt')
        full_labels = np.genfromtxt(label_path, delimiter=',', dtype=str)
        data_names = full_labels[:, 0:2]

        if project_variable.label_type == 'categories':
            labels = full_labels[:, 5].astype(int)
        elif project_variable.label_type == 'arousal':
            labels = full_labels[:, 3].astype(int)
        elif project_variable.label_type == 'valence':
            labels = full_labels[:, 4].astype(int)
        else:
            print('label type not valid')
            labels = None

        if balanced:
            chosen = []
            for i in range(num_categories):
                indices = np.arange(total_dp[which])[labels == i]

                if seed is not None:
                    random.seed(seed)

                num_samples_per_category = dp // num_categories

                if num_samples_per_category > len(indices):
                    chosen.extend(indices)

                    diff = num_samples_per_category - len(indices)
                    assert(len(indices) + diff == num_samples_per_category)
                    choose_indices = random.choices(list(np.arange(len(indices))), k=diff)
                else:
                    choose_indices = random.sample(list(np.arange(len(indices))), num_samples_per_category)

                chosen.extend(indices[choose_indices])
        else:
            chosen = np.arange(total_dp[which])

            if seed is not None:
                random.seed(seed)

            random.shuffle(chosen)
            chosen = chosen[:dp]

        chosen.sort()
        labels = labels[chosen]

        num_points = len(labels)

        data = np.zeros(shape=(num_points, 3, frames, 96, 96), dtype=tp)

        for i in tqdm(range(num_points)):
            choose = chosen[i] # this is the index of the line you need

            num_frames = int(full_labels[choose][2])

            if num_frames < frames:
                new_list = [j_ for j_ in range(num_frames)]
                times = frames // num_frames + 1
                new_list = np.tile(new_list, times)
                new_list = new_list[:frames]

                assert (len(new_list) == frames)

                for j in range(frames):
                    frame_path = os.path.join(path, data_names[choose][0], data_names[choose][1], '%d.jpg' % new_list[j])
                    tmp = np.array(Image.open(frame_path))
                    tmp = tmp.transpose((2, 0, 1))
                    data[i, :, j] = tmp
            else:
                for j in range(frames):
                    frame_path = os.path.join(path, data_names[choose][0], data_names[choose][1], '%d.jpg' % j)
                    tmp = np.array(Image.open(frame_path))
                    tmp = tmp.transpose((2, 0, 1))
                    data[i, :, j] = tmp

        return data, labels

    if project_variable.train:
        if project_variable.randomize_training_data:
            data, labels = load_random('train', project_variable.data_points[0], project_variable.balance_training_data,
                                       seed)
        else:
            data, labels = load('train', project_variable.data_points[0])
        splits.append('train')
        all_data.append(data)
        all_labels.append(labels)

    if project_variable.val:
        data, labels = load('val', project_variable.data_points[1])
        splits.append('val')
        all_data.append(data)
        all_labels.append(labels)

    if project_variable.test:
        data, labels = load('test', project_variable.data_points[2])
        splits.append('test')
        all_data.append(data)
        all_labels.append(labels)

    return splits, all_data, all_labels


def prepare_data(project_variable, full_data, full_labels, device, ts, steps, nice_div):

    if project_variable.train:
        order = np.arange(len(full_labels))
        np.random.shuffle(order)
        full_labels = full_labels[order]
        full_data = full_data[order]

    if ts == steps - 1:
        if nice_div == 0:
            data = full_data[ts * project_variable.batch_size:(ts + 1) * project_variable.batch_size]
            labels = full_labels[ts * project_variable.batch_size:(ts + 1) * project_variable.batch_size]
        else:
            data = full_data[ts * nice_div:(ts + 1) * nice_div]
            labels = full_labels[ts * nice_div:(ts + 1) * nice_div]
    else:
        data = full_data[ts * project_variable.batch_size:(ts + 1) * project_variable.batch_size]
        labels = full_labels[ts * project_variable.batch_size:(ts + 1) * project_variable.batch_size]

    if project_variable.dataset in ['dhg', 'omg_emotion', 'dummy']:
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)

        data = data.cuda(device)

        labels = torch.from_numpy(labels)
        labels = labels.long()
        # https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542

        labels = labels.cuda(device)

    elif project_variable.dataset in ['mnist']:
        data = data.cuda(device)
        labels = labels.cuda(device)

    else:
        data = torch.from_numpy(data).cuda(device)
        labels = torch.from_numpy(labels).cuda(device)

    return data, labels


def load_mnist(project_variable):
    # TODO: do transform?
    image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                       (0.5, 0.5, 0.5)),])

    splits = []
    all_labels = []
    all_data = []

    if project_variable.train:
        mnist = datasets.MNIST(root=PP.mnist_location,
                               train=True,
                               download=False)
                               # ,
                               # transform=image_transform)

        data = mnist.train_data[:50000].unsqueeze(1).type(torch.float)
        labels = mnist.train_labels[:50000]

        splits.append('train')
        all_data.append(data)
        all_labels.append(labels)

    if project_variable.val:
        mnist = datasets.MNIST(root=PP.mnist_location,
                               train=True,
                               download=False)
                               # ,
                               # transform=image_transform)

        data = mnist.train_data[50000:].unsqueeze(1).type(torch.float)
        labels = mnist.train_labels[50000:]

        splits.append('val')
        all_data.append(data)
        all_labels.append(labels)

    if project_variable.test:
        mnist = datasets.MNIST(root=PP.mnist_location,
                               train=False,
                               download=False)
                               # ,
                               # transform=image_transform)

        data = mnist.test_data.unsqueeze(1).type(torch.float)
        labels = mnist.test_labels

        splits.append('test')
        all_data.append(data)
        all_labels.append(labels)

    return splits, all_data, all_labels


def create_dummy_3d_dataset(num_datapoints, c, d, h, w, num_class, pop_with='uniform'):
    """
    options for pop_with    uniform:    random uniform floats between 0 and 1
                            zeros:      zeros
                            ones:       ones
    """
    tp = np.float32
    data = np.zeros(shape=(num_datapoints, c, d, h, w), dtype=tp)
    if pop_with == 'uniform':
        data[:] = np.random.uniform(size=data.shape)
    elif pop_with == 'ones':
        data = np.ones(shape=(num_datapoints, c, d, h, w), dtype=tp)
    else:
        print('Error: %s not a valid value for pop_with' % pop_with)
        data = None

    labels = np.zeros((num_datapoints, num_class), dtype=int)
    _tmp = np.random.randint(low=num_class, size=num_datapoints)
    labels = _tmp

    # for i in range(num_datapoints):
    #     labels[i][_tmp[i]] = 1

    return data, labels


def dummy_uniform_lenet5_3d(project_variable):

    splits, all_data, all_labels = [], [], []

    channels = 1
    time_dim = 30 #10 # TODO: change to 30 see what happens
    # ValueError: Expected input batch_size (100) to match target batch_size (20).

    side = 28
    classes = 10

    TMP = 1000

    if project_variable.train:
        data, labels = create_dummy_3d_dataset(TMP, channels, time_dim, side, side, classes)
        splits.append('train')
        all_data.append(data)
        all_labels.append(labels)
    if project_variable.val:
        data, labels = create_dummy_3d_dataset(TMP, channels, time_dim, side, side, classes)
        splits.append('val')
        all_data.append(data)
        all_labels.append(labels)
    if project_variable.test:
        data, labels = create_dummy_3d_dataset(TMP, channels, time_dim, side, side, classes)
        splits.append('test')
        all_data.append(data)
        all_labels.append(labels)

    return splits, all_data, all_labels


def load_movmnist(project_variable, seed):
    splits = []
    all_labels = []
    all_data = []
    tp = np.float32
    frames = 30

    def load(which, dp):
        path = os.path.join(PP.moving_mnist_png, which)
        label_path = os.path.join(PP.moving_mnist_location, 'labels_%s.csv' % which)
        labels = np.genfromtxt(label_path, dtype=int)[:dp]

        num_points = len(labels)

        data = np.zeros(shape=(num_points, 1, frames, 28, 28), dtype=tp)

        for i in tqdm(range(num_points)):
            for j in range(frames):
                file_path = os.path.join(path, str(i), '%d.png' % j)
                tmp = np.array(Image.open(file_path))
                data[i, 0, j] = tmp

        return data, labels

    def load_random(which, dp, balanced, seed):
        assert(dp % 10 == 0)

        total_dp = {'train':50000, 'val':10000, 'test':10000}

        path = os.path.join(PP.moving_mnist_png, which)
        label_path = os.path.join(PP.moving_mnist_location, 'labels_%s.csv' % which)
        labels = np.genfromtxt(label_path, dtype=int)

        if balanced:
            chosen = []
            for i in range(10):
                indices = np.arange(total_dp[which])[labels == i]

                if seed is not None:
                    random.seed(seed)

                choose_indices = random.sample(list(np.arange(len(indices))), dp//10)
                chosen.extend(indices[choose_indices])
        else:
            chosen = np.arange(total_dp[which])

            if seed is not None:
                random.seed(seed)

            random.shuffle(chosen)
            chosen = chosen[:dp]

        chosen.sort()
        labels = labels[chosen]

        num_points = len(labels)

        data = np.zeros(shape=(num_points, 1, frames, 28, 28), dtype=tp)

        for i in tqdm(range(num_points)):
            choose = chosen[i]
            for j in range(frames):
                file_path = os.path.join(path, str(choose), '%d.png' % j)
                tmp = np.array(Image.open(file_path))
                data[i, 0, j] = tmp

        return data, labels


    if project_variable.train:
        if project_variable.randomize_training_data:
            data, labels = load_random('train', project_variable.data_points[0], project_variable.balance_training_data,
                                       seed)
        else:
            data, labels = load('train', project_variable.data_points[0])
        splits.append('train')
        all_data.append(data)
        all_labels.append(labels)

    if project_variable.val:
        data, labels = load('val', project_variable.data_points[1])
        splits.append('val')
        all_data.append(data)
        all_labels.append(labels)

    if project_variable.test:
        data, labels = load('test', project_variable.data_points[2])
        splits.append('test')
        all_data.append(data)
        all_labels.append(labels)

    return splits, all_data, all_labels


def load_kthactions(project_variable, seed):
    tp = np.float32
    labels_dict = {0:'boxing', 1:'handclapping', 2:'handwaving', 3:'jogging', 4:'running', 5:'walking'}
    # Note: person13_handclapping_d3 missing
    splits = []
    all_labels = []
    all_data = []
    FRAMES = project_variable.load_num_frames

    def load(which, dp, seed):
        total_dp = {'train': 191, 'val': 192, 'test': 216}
        # frames = {'train':project_variable.load_num_frames[0],
        #           'val':project_variable.load_num_frames[1],
        #           'test':project_variable.load_num_frames[2]}

        if dp != total_dp[which]:
            assert(dp % 6 == 0)

        # data = np.zeros(shape=(dp, 1, FRAMES, 120, 160), dtype=tp)
        data = np.zeros(shape=(dp, 1, FRAMES, 60, 60), dtype=tp)

        kth_png_path = os.path.join(PP.kth_png_60_60, which)

        # balanced by default
        if which == 'train' and dp == total_dp[which]:
            _tmp = dp + 1
            labels = np.repeat(list(labels_dict.keys()), _tmp // 6)
        else:
            labels = np.repeat(list(labels_dict.keys()), dp // 6)

        # remove missing datapoint person13_handclapping_d3
        if which == 'train' and dp == total_dp[which]:
            labels = list(labels)
            labels.pop(34)
            assert(len(labels) == dp)
            labels = np.array(labels)

        chosen_paths = []

        # sets seeds for each class such that different people are chosen
        non_train_seeds = ['a', 'b', 'c', 'd', 'e', 'f']
        if seed is not None and which == 'train':
            np.random.seed(seed)
            train_seeds = np.random.randint(10000, size=6)

        for c in range(6):
            class_path = os.path.join(kth_png_path, labels_dict[c])
            options = os.listdir(class_path)
            options.sort()
            if seed is not None and which == 'train':
                random.seed(train_seeds[c])
            elif seed is None and which != 'train':
                random.seed(non_train_seeds[c])

            if dp == total_dp[which]:
               chosen = options
            else:
                chosen = random.sample(options, dp // 6) # samples without replacement

            for i in chosen:
                p = os.path.join(class_path, i)
                chosen_paths.append(p)

        if which == 'train':
            chosen_paths = np.array(chosen_paths)
            shuffle_order = list(np.arange(dp))

            if seed is not None:
                random.seed(seed)

            random.shuffle(shuffle_order)

            labels = labels[shuffle_order]
            chosen_paths = chosen_paths[shuffle_order]

        for i in tqdm(range(dp)):
            choose = chosen_paths[i]

            num_frames = len(os.listdir(choose))

            # pad shorter sequences with zeros
            if FRAMES+1 > num_frames+1:
                end = num_frames+1
            else:
                end = FRAMES+1

            for j in range(1, end):
                file_path = os.path.join(choose, '%d.png' % j)
                tmp = np.array(Image.open(file_path))
                # channels are the same
                data[i, 0, j-1] = tmp[:,:,0]

        return data, labels

    if project_variable.train:
        data, some_labels = load('train', project_variable.data_points[0], seed)
        splits.append('train')
        all_data.append(data)
        all_labels.append(some_labels)

    if project_variable.val:
        data, some_labels = load('val', project_variable.data_points[1], seed)
        splits.append('val')
        all_data.append(data)
        all_labels.append(some_labels)

    if project_variable.test:
        data, some_labels = load('test', project_variable.data_points[2], seed)
        splits.append('test')
        all_data.append(data)
        all_labels.append(some_labels)

    return splits, all_data, all_labels


def load_dhg(project_variable, seed):
    tp = np.float32
    splits = []
    all_labels = []
    all_data = []
    FRAMES = project_variable.load_num_frames


    def load(which, dp):

        label_path = os.path.join(PP.dhg_hand_only_28_28_50_frames, 'labels_%s.txt' % which)
        labels = np.genfromtxt(label_path, delimiter=',', dtype=int)[:dp]
        # TODO: could be that we need startindex 0

        data = np.zeros(shape=(dp, 1, FRAMES, 28, 28), dtype=tp)

        for i in tqdm(range(dp)):
            for j in range(FRAMES):

                img_path = os.path.join(PP.dhg_hand_only_28_28_50_frames, 'gesture_%d/finger_%d/subject_%s/essai_%d/depth_%d.png'
                                        % (labels[i][0], labels[i][1], labels[i][2], labels[i][3], j+1))
                tmp = Image.open(img_path)
                tmp = np.array(tmp.convert('L'))
                data[i, 0, j] = tmp

        labels = labels[:, 0]
        labels = labels - 1

        return data, labels

    def load_random(which, dp, balanced, seed):
        num_categories = 14
        total_dp = {'train': 1960, 'val': 280, 'test': 560}
        if dp != total_dp[which]:
            assert(dp % 14 == 0)

        label_path = os.path.join(PP.dhg_hand_only_28_28_50_frames, 'labels_%s.txt' % which)
        full_labels = np.genfromtxt(label_path, delimiter=',', dtype=int)
        labels = full_labels[:, 0]

        if balanced:
            if seed is not None:
                random.seed(seed)

            chosen = []
            for i in range(num_categories):
                indices = np.arange(total_dp[which])[labels == i+1]

                # if seed is not None:
                #     random.seed(seed)

                num_samples_per_category = dp // num_categories

                if num_samples_per_category > len(indices):
                    chosen.extend(indices)

                    diff = num_samples_per_category - len(indices)
                    assert(len(indices) + diff == num_samples_per_category)
                    choose_indices = random.choices(list(np.arange(len(indices))), k=diff)
                else:
                    choose_indices = random.sample(list(np.arange(len(indices))), num_samples_per_category)

                chosen.extend(indices[choose_indices])
        else:
            chosen = np.arange(total_dp[which])

            if seed is not None:
                random.seed(seed)

            random.shuffle(chosen)
            chosen = chosen[:dp]

        chosen.sort()
        labels = labels[chosen]
        full_labels = full_labels[chosen]

        num_points = len(labels)

        data = np.zeros(shape=(num_points, 1, FRAMES, 28, 28), dtype=tp)

        for i in tqdm(range(num_points)):
            for j in range(FRAMES):
                img_path = os.path.join(PP.dhg_hand_only_28_28_50_frames,
                                        'gesture_%d/finger_%d/subject_%s/essai_%d/depth_%d.png'
                                        % (full_labels[i][0], full_labels[i][1], full_labels[i][2], full_labels[i][3], j + 1))
                tmp = Image.open(img_path)
                tmp = np.array(tmp.convert('L'))
                # tmp = tmp.transpose((2, 0, 1))
                data[i, :, j] = tmp

        labels = labels - 1

        return data, labels


    if project_variable.train:
        if project_variable.randomize_training_data:
            data, some_labels = load_random('train', project_variable.data_points[0],
                                            project_variable.balance_training_data, seed)
        else:
            data, some_labels = load('train', project_variable.data_points[0])
        splits.append('train')
        all_data.append(data)
        all_labels.append(some_labels)

    if project_variable.val:
        data, some_labels = load('val', project_variable.data_points[1])
        splits.append('val')
        all_data.append(data)
        all_labels.append(some_labels)

    if project_variable.test:
        data, some_labels = load('test', project_variable.data_points[2])
        splits.append('test')
        all_data.append(data)
        all_labels.append(some_labels)

    return splits, all_data, all_labels


def load_jester(project_variables, seed):

    pass


def get_mean_std_train_jester():

    pass

def get_mean_std_train_dhg():
    if os.path.exists(PP.dhg_mean_std):
        total = np.load(PP.dhg_mean_std)
        mean = total[0]
        std = total[1]
    else:
        all_train_files = np.zeros(shape=(140, 50, 28, 28))

        label_path = os.path.join(PP.dhg_hand_only_28_28_50_frames, 'labels_train.txt')
        labels = np.genfromtxt(label_path, delimiter=',', dtype=int)[:140]

        for i in tqdm(range(140)):
            for j in range(50):
                img_path = os.path.join(PP.dhg_hand_only_28_28_50_frames,
                                        'gesture_%d/finger_%d/subject_%s/essai_%d/depth_%d.png'
                                        % (labels[i][0], labels[i][1], labels[i][2], labels[i][3], j + 1))
                tmp = Image.open(img_path)
                tmp = np.array(tmp.convert('L'))
                all_train_files[i, j] = tmp

        # get the mean as array.mean(axis=0)
        mean = all_train_files.mean(axis=0)
        # get the std as array.std(axis=0)
        std = all_train_files.std(axis=0)

    # save files for next time
    total = np.array([mean, std])
    np.save(PP.dhg_mean_std, total)

    return mean, std


def get_mean_std_train_mov_mnist():
    if os.path.exists(PP.mov_mnist_mean_std):
        total = np.load(PP.mov_mnist_mean_std)
        mean = total[0]
        std = total[1]
    else:
        all_train_files = np.zeros(shape=(1000, 30, 28, 28))

        train_data_path = os.path.join(PP.moving_mnist_location, 'png', 'train')

        for i in tqdm(range(1000)):
            for j in range(30):
                img_path = os.path.join(train_data_path, str(i), '%s.png' % str(j))
                tmp = Image.open(img_path)
                tmp = np.array(tmp.convert('L'))
                all_train_files[i, j] = tmp

        # get the mean as array.mean(axis=0)
        mean = all_train_files.mean(axis=0)
        # get the std as array.std(axis=0)
        std = all_train_files.std(axis=0)

    # save files for next time
    total = np.array([mean, std])
    np.save(PP.mov_mnist_mean_std, total)

    return mean, std



def load_data(project_variable, seed):
    if project_variable.dataset == 'omg_emotion':
        return load_omg_emotion(project_variable, seed)
    elif project_variable.dataset == 'mnist':
        return load_mnist(project_variable)
    elif project_variable.dataset == 'dummy':
        return dummy_uniform_lenet5_3d(project_variable)
    elif project_variable.dataset == 'mov_mnist':
        return load_movmnist(project_variable, seed)
    elif project_variable.dataset == 'kth_actions':
        return load_kthactions(project_variable, seed)
    elif project_variable.dataset == 'dhg':
        return load_dhg(project_variable, seed)
    elif project_variable.dataset == 'jester':
        return load_jester(project_variable, seed)
    else:
        print('Error: dataset %s not supported' % project_variable.dataset)
        return None


