import os
import deepimpression2.constants as C
from relative_baseline import utils as U
from relative_baseline import data_loading as D
import numpy as np
import random
from . import project_paths as PP
from relative_baseline.omg_emotion import utils as U1
import torch
import random
from PIL import Image

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
    # project_variable = ProjectVariable
    assert which in ['Training', 'Validation', 'Test']
    path = os.path.join(PP.data_path, which, 'Annotations', 'annotations.csv')
    annotations = np.genfromtxt(path, delimiter=',', dtype=str)

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



def load_data(project_variable):
    # project_variable = ProjectVariable()

    all_labels = []
    splits = []

    if project_variable.train:
        labels = load_labels('Training', project_variable)
        # TODO: shuffle

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
        datapoints = len(all_labels[i][0])
        data = np.zeros(shape=(datapoints, 3, 1280, 720), dtype=np.float32)

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
                                          all_labels[i][0][0][j],
                                          all_labels[i][0][1][j].split('.')[0])

            # select random frame
            frames = os.listdir(utterance_path)
            index = random.randint(0, len(frames)-1)
            jpg_path = os.path.join(utterance_path, frames[index])

            jpg_as_arr = Image.open(jpg_path)
            data[j] = jpg_as_arr

        final_data.append(data)

        tmp = all_labels[i][:, 1:]

        final_labels.append(tmp)

    # splits = ['train', 'val', 'test']
    # final_data = [[img0, img1,...],
    #               [img0, img1,...],
    #               [img0, img1,...]]
    # final_labels = [[arousal, valence, categories],
    #                 [arousal, valence, categories],
    #                 [arousal, valence, categories]]
    return splits, final_data, final_labels






