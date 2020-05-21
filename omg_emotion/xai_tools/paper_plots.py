import os
import numpy as np
from PIL import Image
import cv2 as cv
from scipy import stats
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import functional as F

from relative_baseline.omg_emotion.settings import ProjectVariable as pv
from relative_baseline.omg_emotion import setup
import relative_baseline.omg_emotion.data_loading as DL
import relative_baseline.omg_emotion.project_paths as PP

from relative_baseline.omg_emotion.xai_tools.misc_functions import preprocess_image, recreate_image, save_clip
from relative_baseline.omg_emotion.models import deconv_3DTTN, deconv_3D
from relative_baseline.omg_emotion import utils as U
from relative_baseline.omg_emotion import visualization as VZ


def init(dataset, model):
    proj_var = pv(debug_mode=True)
    proj_var.dataset = dataset
    proj_var.load_model = model
    proj_var.model_number = model[1]
    proj_var.xai_only_mode = True
    proj_var.use_dali = True
    proj_var.batch_size_val_test = 1
    proj_var.dali_workers = 32
    proj_var.device = 0
    proj_var.load_num_frames = 30
    if dataset == 'jester':
        proj_var.label_size = 27
    elif dataset == 'ucf101':
        proj_var.label_size = 101
    return proj_var


def prepare_data(data, labels, dataset):
    data = data.permute(0, 4, 1, 2, 3)
    og_data = data.clone()
    # convert to floattensor
    data = data.type(torch.float32)
    data = data / 255
    data[:, 0, :, :, :] = (data[:, 0, :, :, :] - 0.485) / 0.229
    data[:, 1, :, :, :] = (data[:, 1, :, :, :] - 0.456) / 0.224
    data[:, 2, :, :, :] = (data[:, 2, :, :, :] - 0.406) / 0.225

    labels = labels.type(torch.long)
    labels = labels.flatten()
    if dataset == 'jester':
        labels = labels - 1

    return og_data, data, labels


def sort_dict(the_dict, reverse=True, axis=1):
    the_dict = {k: v for k, v in sorted(the_dict.items(), reverse=reverse, key=lambda item: item[axis])}
    return the_dict


def map_to_names(the_dict, name_path, dataset):
    new_dict = dict()
    if dataset == 'jester':
        names = np.genfromtxt(name_path, dtype=str, delimiter=' ')
        names = names[:, 0]
    else:
        folders = os.listdir(name_path)
        folders.sort()
        names = []
        for fol in folders:
            fol_path = os.path.join(name_path, fol)
            nam = os.listdir(fol_path)
            nam.sort()
            names.extend(nam)

    assert len(the_dict) == len(names)

    for k in the_dict:
        new_key = names[k]
        new_dict[new_key] = the_dict[k]

    return new_dict


def get_max_activation(model, model_number, data, device):

    if model_number in [21, 20]: # resnet18
        conv_layers = [i+1 for i in range(20) if (i+1) not in [6, 11, 16]]

    else: # googlenet
        conv_layers = [1, 3, 6, 8, 12, 14, 18, 20, 24, 26, 31, 33, 37, 39, 43, 45, 50, 52, 56, 58]

    model_max = 0

    model.eval()
    with torch.no_grad():
        for conv in conv_layers:
            if model_number == 20: # RN18 3T
                feature_map = model(data, device, stop_at=conv)
            elif model_number == 23: # GN 3T
                aux1, aux2, feature_map = model(data, device, None, False)
            elif model_number == 21: # RN18 3D
                feature_map = model(data, stop_at=conv)
            elif model_number == 25: # GN 3D
                aux1, aux2, feature_map = model(data, None, False)

        layer_max = np.array(feature_map.data.cpu()).max()
        model_max = model_max + layer_max

    return model_max


def find_best_videos(dataset, model):
    proj_var = init(dataset, model)
    # model num 21, 20, 25, 23
    # dataset jester, ucf101

    my_model = setup.get_model(proj_var)
    device = setup.get_device(proj_var)

    if dataset == 'jester':
        the_iterator = DL.get_jester_iter(None, proj_var)
    elif dataset == 'ucf101':
        the_iterator = DL.get_ucf101_iter(None, proj_var)
    else:
        print("Dataset name '%s' not recognized" % dataset)
        return

    correct_pred = dict()
    wrong_pred = dict()

    for i, data_and_labels in tqdm(enumerate(the_iterator)):
        prediction = None

        data = data_and_labels[0]['data']
        labels = data_and_labels[0]['labels']

        og_data, data, labels = prepare_data(data, labels, dataset)

        my_model.eval()
        if proj_var.model_number == 20:
            prediction = my_model(data, proj_var.device)
        elif proj_var.model_number == 23:
            aux1, aux2, prediction = my_model(data, proj_var.device, None, False)

        my_model.zero_grad()
        prediction = np.array(prediction[0].data.cpu()).argmax()
        labels = int(labels.data.cpu())

        value_max_activation = get_max_activation(my_model, model[1], data, device)

        if prediction == labels:
            correct_pred[i] = value_max_activation
        else:
            wrong_pred[i] = value_max_activation

    correct_pred = sort_dict(correct_pred)
    wrong_pred = sort_dict(wrong_pred)

    if dataset == 'jester':
        names = os.path.join(PP.jester_location, 'filelist_test_xai.txt')
    elif dataset == 'ucf101':
        names = PP.ucf101_168_224_xai
    else:
        print('dataset name is wrong')
        return

    correct_pred = map_to_names(correct_pred, names, dataset)
    wrong_pred = map_to_names(wrong_pred, names, dataset)

    filename_correct = 'high_act_vids-correct_pred-%s-exp_%d_mod_%d_ep_%d.txt' % (dataset, model[0], model[1], model[2])
    filename_wrong = 'high_act_vids-wrong_pred-%s-exp_%d_mod_%d_ep_%d.txt' % (dataset, model[0], model[1], model[2])

    with open(filename_correct, 'a') as my_file:
        for k, v in correct_pred:
            line = '%s %f\n' % (k, v)
            # my_file.write(line)

    with open(filename_wrong, 'a') as my_file:
        for k, v in wrong_pred:
            line = '%s %f\n' % (k, v)
            # my_file.write(line)
