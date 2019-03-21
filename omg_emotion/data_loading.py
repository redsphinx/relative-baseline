import os
import numpy as np
from . import project_paths as PP
from relative_baseline.omg_emotion import utils as U1
import random
from PIL import Image
import cv2
from time import time

# temporary for debugging
from .settings import ProjectVariable

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

def load_labels(which, project_variable):
    # TODO: one-hot-encode labels
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
    # TODO: get rid of this, fix the jpgs
    if jpg_as_arr.width != 1280 or jpg_as_arr.height != 720:
        jpg_as_arr = jpg_as_arr.resize((1280, 720))
        cnt += 1
    # ValueError: could not broadcast input array from shape (1280,3,720) into shape (3,720,1280)

    jpg_as_arr = np.array(jpg_as_arr, dtype=np.float32).transpose((2, 0, 1))

    return jpg_as_arr, cnt


def load_data(project_variable):
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
        start = time()
        cnt = 0
        datapoints = len(all_labels[i][0])
        data = np.zeros(shape=(datapoints, 3, 720, 1280), dtype=np.float32)

        # TODO: https://pytorch.org/docs/stable/torchvision/models.html

        if s == 'train': which = 'Training'
        elif s == 'val': which = 'Validation'
        else: which = 'Test'

        if s == 'val' or s == 'test':
            random.seed(project_variable.seed)

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

            # index = random.randint(0, len(frames)-1)
            # jpg_path = os.path.join(utterance_path, '%d.jpg' % index)
            #
            # # left_data = np.zeros((batch_size, 3, 542, 662), dtype=np.float32)
            # jpg_as_arr = Image.open(jpg_path)
            # # TODO: get rid of this, fix the jpgs
            # if jpg_as_arr.width != 1280 or jpg_as_arr.height != 720:
            #     jpg_as_arr = jpg_as_arr.resize((1280, 720))
            #     cnt += 1
            # # ValueError: could not broadcast input array from shape (1280,3,720) into shape (3,720,1280)
            #
            # jpg_as_arr = np.array(jpg_as_arr, dtype=np.float32).transpose((2, 0, 1))

            # scale between 0 and 1 for resnet18
            if project_variable.model_number == 0:
                while int(np.max(jpg_as_arr)) == 0:
                    print('max is zero')
                    jpg_as_arr, cnt = get_nonzero_frame(frames, utterance_path, cnt)

                jpg_as_arr /= int(np.max(jpg_as_arr))

                # if int(np.max(jpg_as_arr)) > 0:
                #     jpg_as_arr /= int(np.max(jpg_as_arr))
                # else:
                #     print('something wrong with: %s' % jpg_path)
                #     jpg_as_arr = np.zeros(shape=jpg_as_arr.shape)

            data[j] = jpg_as_arr

        end = time() - start


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






