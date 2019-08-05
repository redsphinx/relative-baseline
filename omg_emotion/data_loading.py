import os
import numpy as np
from . import project_paths as PP
from relative_baseline.omg_emotion import utils as U1
import random
from PIL import Image
import cv2
# from time import time
import torch
from multiprocessing import Pool, Queue
from tqdm import tqdm
# temporary for debugging
from .settings import ProjectVariable
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


def load_omg_emotion(project_variable):
    # TODO: https://pytorch.org/docs/stable/torchvision/models.html
    # normalize the data
    # project_variable = ProjectVariable()

    all_labels = []
    splits = []

    if project_variable.train:
        labels = load_labels('Training', project_variable)
        all_labels.append(labels)
        splits.append('train')

    if project_variable.val:
        labels = load_labels('Validation', project_variable)
        all_labels.append(labels)
        splits.append('val')

    if project_variable.test:
        labels = load_labels('Test', project_variable)
        all_labels.append(labels)
        splits.append('test')

    # all_labels = [[names, arousal, valence, categories],
    #               [names, arousal, valence, categories],
    #               [names, arousal, valence, categories]]
    # splits = ['train', 'val', 'test']
    # names = [[f, u], [f, u], ...]

    final_data = []
    final_labels = []

    for i, s in enumerate(splits):
        cnt = 0
        datapoints = len(all_labels[i][0])
        data = np.zeros(shape=(datapoints, 3, 720, 1280), dtype=np.float32)

        if s == 'train':
            which = 'Training'
        elif s == 'val':
            which = 'Validation'
        else:
            which = 'Test'

        if s == 'val' or s == 'test':
            random.seed(project_variable.seed)

        # if s == 'train':
        if s == 'debug_this_shit':
            start = time.time()
            items_packaged = []

            all_utterance_paths = []
            for j in range(datapoints):
                # '../omg_emotion/Validation/jpg.../xxxxxxx/utterance_xx/'
                utterance_path = os.path.join(PP.data_path,
                                              which,
                                              PP.omg_emotion_jpg,
                                              all_labels[i][0][j][0],
                                              all_labels[i][0][j][1].split('.')[0])
                all_utterance_paths.append(utterance_path)
                items_packaged.append([utterance_path, j])

            # parallel_load(all_utterance_paths, index_list)
            pool = parallel_load(items_packaged)
            # print('training data parallel loaded')

            assert global_queue.qsize() == datapoints
            indices = []

            for d in tqdm(range(datapoints)):
                item = global_queue.get()
                indices.append(item[1])
                data[item[1]] = item[0]

            # close queue
            pool.terminate()

            print('parallel loading: %f' % (time.time() - start))

        else:
            start = time.time()
            for j in range(datapoints):

                # '../omg_emotion/Validation/jpg.../xxxxxxx/utterance_xx/'
                utterance_path = os.path.join(PP.data_path,
                                              which,
                                              PP.omg_emotion_jpg,
                                              all_labels[i][0][j][0],
                                              all_labels[i][0][j][1].split('.')[0])

                # select random frame
                frames = os.listdir(utterance_path)

                jpg_as_arr, cnt = get_nonzero_frame(frames, utterance_path, cnt)

                # scale between 0 and 1 for resnet18
                if project_variable.model_number == 0:
                    while int(np.max(jpg_as_arr)) == 0:
                        print('max is zero')
                        jpg_as_arr, cnt = get_nonzero_frame(frames, utterance_path, cnt)

                    try:
                        jpg_as_arr[0] = jpg_as_arr[0] / np.max(jpg_as_arr[0])
                        jpg_as_arr[1] = jpg_as_arr[1] / np.max(jpg_as_arr[1])
                        jpg_as_arr[2] = jpg_as_arr[2] / np.max(jpg_as_arr[2])
                    except RuntimeWarning:
                        print('channel has a max of 0')
                        return
                    # jpg_as_arr /= int(np.max(jpg_as_arr))

                    import torchvision.transforms as transforms
                    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
                    jpg_as_arr = torch.from_numpy(jpg_as_arr)
                    jpg_as_arr = normalize(jpg_as_arr)
                    jpg_as_arr = jpg_as_arr.numpy()

                data[j] = jpg_as_arr
            print('normal loading: %f' % (time.time() - start))

        final_data.append(data)

        tmp = all_labels[i][:][1:]

        final_labels.append(tmp)

        # print('items to resize %s: %d' % (s, cnt))

    # splits = ['train', 'val', 'test']
    # final_data = [[img0, img1,...],
    #               [img0, img1,...],
    #               [img0, img1,...]]
    # final_labels = [[arousal, valence, categories],
    #                 [arousal, valence, categories],
    #                 [arousal, valence, categories]]

    return splits, final_data, final_labels


def prepare_data(project_variable, full_data, full_labels, device, ts, steps, nice_div):
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

    if project_variable.dataset in ['omg_emotion', 'dummy']:
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)
        # normalize image data
        # import torchvision.transforms as transforms
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # data = torch.from_numpy(data)
        #
        # if ts == steps - 1:
        #     if nice_div == 0:
        #         for _b in range(project_variable.batch_size):
        #             data[_b] = normalize(data[_b])
        #     else:
        #         for _b in range(nice_div):
        #             data[_b] = normalize(data[_b])
        # else:
        #     for _b in range(project_variable.batch_size):
        #         data[_b] = normalize(data[_b])

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


    # TODO: implement
    def load_faster(which):
        # first load all data into cpu memory
        # move data from cpu to gpu as you need it
        pass


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


def load_data(project_variable, seed):
    if project_variable.dataset == 'omg_emotion':
        return load_omg_emotion(project_variable)
    elif project_variable.dataset == 'mnist':
        return load_mnist(project_variable)
    elif project_variable.dataset == 'dummy':
        return dummy_uniform_lenet5_3d(project_variable)
    elif project_variable.dataset == 'mov_mnist':
        return load_movmnist(project_variable, seed)
    else:
        print('Error: dataset %s not supported' % project_variable.dataset)
        return None
















































