import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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
from relative_baseline.omg_emotion.utils import opt_mkdir, opt_makedirs
from relative_baseline.omg_emotion.xai_tools.layer_visualization import create_next_frame, normalize

from relative_baseline.omg_emotion.xai_tools.misc_functions import preprocess_image, recreate_image, save_clip
from relative_baseline.omg_emotion.models import deconv_3DTTN, deconv_3D

from relative_baseline.omg_emotion import visualization as VZ


def init1(dataset, model):
    proj_var = pv(debug_mode=True)
    proj_var.dataset = dataset
    proj_var.load_model = model
    proj_var.model_number = model[1]
    proj_var.xai_only_mode = True
    proj_var.use_dali = True
    proj_var.batch_size_val_test = 1
    proj_var.dali_workers = 32
    proj_var.device = 1
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
            nam = [os.path.join(PP.ucf101_168_224_xai, fol, nam[i]) for i in range(len(nam))]
            names.extend(nam)

    # assert len(the_dict) == len(names)

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
                feature_map = model(data, device, stop_at=conv, aux=False)
            elif model_number == 21: # RN18 3D
                feature_map = model(data, stop_at=conv)
            elif model_number == 25: # GN 3D
                feature_map = model(data, stop_at=conv, aux=False)

            layer_max = np.array(feature_map.data.cpu()).max()
            model_max = model_max + layer_max

    return model_max


def find_best_videos(dataset, model):
    proj_var = init1(dataset, model)
    # model num 21, 20, 25, 23
    # dataset jester, ucf101

    my_model = setup.get_model(proj_var)
    device = setup.get_device(proj_var)
    my_model.cuda(device)

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
        # if i > 2:
        #     break
        prediction = None

        data = data_and_labels[0]['data']
        labels = data_and_labels[0]['labels']

        og_data, data, labels = prepare_data(data, labels, dataset)

        my_model.eval()
        if proj_var.model_number == 20:
            prediction = my_model(data, proj_var.device)
        elif proj_var.model_number == 23:
            aux1, aux2, prediction = my_model(data, proj_var.device, None, False)
        elif proj_var.model_number == 21:
            prediction = my_model(data)
        elif proj_var.model_number == 25:
            aux1, aux2, prediction = my_model(data, None, False)

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
    
    filename_correct = os.path.join(PP.xai_metadata, filename_correct)
    filename_wrong = os.path.join(PP.xai_metadata, filename_wrong)
    
    with open(filename_correct, 'a') as my_file:
        for k in correct_pred:
            line = '%s %f\n' % (k, correct_pred[k])
            my_file.write(line)

    with open(filename_wrong, 'a') as my_file:
        for k in wrong_pred:
            line = '%s %f\n' % (k, wrong_pred[k])
            my_file.write(line)


# find_best_videos('jester', [31, 20, 8, 0])
# find_best_videos('ucf101', [1001, 20, 45, 0])
# find_best_videos('jester', [26, 21, 45, 0])
# find_best_videos('ucf101', [1000, 21, 40, 0])
# find_best_videos('jester', [28, 25, 25, 0])
# find_best_videos('ucf101', [1002, 25, 54, 0])
# find_best_videos('jester', [30, 23, 28, 0])
# find_best_videos('ucf101', [1003, 23, 12, 0])

def save_as_plot(scales, rotations, xs, ys, model, conv, ch, dataset):
    x_axis = np.arange(len(scales)+1)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    # plt.setp(ax3, adjustable='box', aspect='equal')


    if dataset == 'jester':
        h, w = 150, 224
    elif dataset == 'ucf101':
        h, w = 168, 224
    
    new_scales = [1.]
    new_rotations = [0.]
    new_xs = [0.]
    new_ys = [0.]

    for i in range(len(scales)):
        new_scales.append(scales[i] * new_scales[-1])
        new_rotations.append(rotations[i] + new_rotations[-1])
        new_xs.append(xs[i]*w + new_xs[-1])
        new_ys.append(ys[i]*h + new_ys[-1])

    # FIX: yticks and xticks
    # FIX: labelsize
    # FIX: spacing
    # FIX: overall fig shape
    ax1.plot(x_axis, new_scales, 'o-', linewidth=1, markersize=5)
    ax1.set_ylabel('size ratio')
    ax1.set_xlabel('time')
    ax1.set_title('cumulative scale')
    # ax1.set_aspect('equal', 'box')
    # ax1.set(adjustable='box')
    ax1.grid(True)
    ax1.axis('square')

    ax2.plot(x_axis, new_rotations, 'o-', linewidth=1, markersize=5)
    ax2.set_ylabel('degrees')
    ax2.set_xlabel('time')
    ax2.set_title('cumulative rotation')
    # ax2.set_aspect('equal', 'box')
    # ax2.set(adjustable='box')
    ax2.grid(True)
    ax2.axis('square')

    txt = ['t'+str(i) for i in range(len(new_scales))]
    ax3.plot(new_xs, new_ys, 'o-', linewidth=1, markersize=5)
    ax3.set_title('X and Y location in pixels')
    ax3.set_ylabel('y')
    ax3.set_xlabel('x')
    # ax3.set_aspect('equal', 'box')
    # ax3.set(adjustable='box')
    ax3.grid(True)
    ax3.axis('square')
    for i, j in enumerate(txt):
        ax3.annotate(j, (new_xs[i], new_ys[i]))

    if model[1] == 21:
        m = '3D-ResNet18'
    elif model[1] == 20:
        m = '3T-ResNet18'
    elif model[1] == 25:
        m = '3D-GoogLeNet'
    elif model[1] == 23:
        m = '3T-GoogLeNet'

    fig.suptitle('%s on %s, layer %d channel %d' % (m, dataset, conv, ch + 1))

    # fig.tight_layout()

    p1 = 'exp_%d_mod_%d_ep_%d' % (model[0], model[1], model[2])
    p2 = 'layer_%d_channel_%d.jpg' % (conv, ch+1)
    save_location = os.path.join(PP.srxy_plots, p1, p2)

    intermediary_path = os.path.join(PP.srxy_plots, p1)
    opt_mkdir(intermediary_path)

    plt.savefig(save_location)


def plot_all_srxy(dataset, model):
    proj_var = init1(dataset, model)
    my_model = setup.get_model(proj_var)
    # device = setup.get_device(proj_var)
    # my_model.cuda(device)

    if model[1] in [21, 20]: # resnet18
        conv_layers = [i+1 for i in range(20) if (i+1) not in [6, 11, 16]]
    else: # googlenet
        conv_layers = [1, 3, 6, 8, 12, 14, 18, 20, 24, 26, 31, 33, 37, 39, 43, 45, 50, 52, 56, 58]

    for ind in tqdm(conv_layers):
        end = getattr(my_model, 'conv%d' % ind)
        end = end.weight.shape[0]

        for ch in range(0, end):
            _conv_name = 'conv%d' % ind
            transformations = getattr(getattr(my_model, _conv_name), 'scale')
            num_transformations = transformations.shape[0]
            transformations = getattr(my_model, _conv_name)

            scales = []
            rotations = []
            xs = []
            ys = []

            for trafo in range(num_transformations):
                s = getattr(transformations, 'scale')[trafo, ch]
                r = getattr(transformations, 'rotate')[trafo, ch]
                x = getattr(transformations, 'translate_x')[trafo, ch]
                y = getattr(transformations, 'translate_y')[trafo, ch]
                
                # translate parameters to interpretable things
                # scale -> 1/s
                # rotation -> degrees counterclockwise
                # x, y -> half of size image

                s = 1 / float(s)
                r = -1 * float(r)
                x = -0.5 * float(x)
                y = 0.5 * float(y)
                
                scales.append(s)
                rotations.append(r)
                xs.append(x)
                ys.append(y)

            save_as_plot(scales, rotations, xs, ys, model, ind, ch, dataset)

# +---------+------------+--------------+
# |         |   Jester   |    UCF101    |
# +---------+------------+--------------+
# | RN18 3D | 26, 21, 45 | 1000, 21, 40 |
# +---------+------------+--------------+
# | RN18 3T |  31, 20, 8 | 1001, 20, 45 |
# +---------+------------+--------------+
# | GN 3D   | 28, 25, 25 | 1002, 25, 54 |
# +---------+------------+--------------+
# | GN 3T   | 30, 23, 28 | 1003, 23, 12 |
# +---------+------------+--------------+

# plot_all_srxy('jester', [31, 20, 8, 0])

def visualize_all_first_layer_filters(dataset, model):
    proj_var = init1(dataset, model)
    my_model = setup.get_model(proj_var)
    proj_var.device = None
    device = setup.get_device(proj_var)

    p1 = 'exp_%d_mod_%d_ep_%d' % (model[0], model[1], model[2])
    intermediary_path = os.path.join(PP.filters_conv1, p1)
    opt_mkdir(intermediary_path)

    num_channels = getattr(my_model, 'conv1')
    num_channels = num_channels.weight.shape[0]

    if model[1] in [21, 25]: # 3D
        w = np.array(my_model.conv1.weight.data)
    else: # 3T
        w = my_model.conv1.first_weight.data
        s = my_model.conv1.scale.data
        r = my_model.conv1.rotate.data
        x = my_model.conv1.translate_x.data
        y = my_model.conv1.translate_y.data


    for ch in tqdm(range(num_channels)):
        # if ch > 2:
        #     break
        channel_path = os.path.join(intermediary_path, 'channel_%d' % (ch + 1))
        opt_mkdir(channel_path)

        if model[1] in [20, 23]: # 3T
            k0 = w[ch, :, 0,]
            num_transformations = s.shape[0]
            w3d = np.zeros(shape=(w.shape[1], w.shape[2] + num_transformations, w.shape[3], w.shape[4]), dtype=np.float32)
            w3d[:, 0] = np.array(k0.clone().data)
            _s, _r, _x, _y = s[0, ch], r[0, ch], x[0, ch], y[0, ch]
            for i in range(num_transformations):
                # apply them on the final image
                if i == 0:
                    next_k = create_next_frame(_s, _r, _x, _y, k0, device)
                    w3d[:, i+1] = np.array(next_k.clone().data)
                else:
                    _s = _s * s[i, ch]
                    _r = _r + r[i, ch]
                    _x = _x + x[i, ch]
                    _y = _y + y[i, ch]
                    next_k = create_next_frame(_s, _r, _x, _y, k0, device)
                    w3d[:, i+1] = np.array(next_k.clone().data)
            w_chan = w3d.copy()
        else:
            w_chan = w[ch].copy()

        w_chan = w_chan.transpose(1, 2, 3, 0)
        num_slices = w_chan.shape[0]

        for slc in range(num_slices):
            w_slice = w_chan[slc]
            w_slice = normalize(w_slice.transpose(2, 0, 1))
            w_slice = np.array(w_slice.transpose(1, 2, 0), dtype=np.uint8)

            img = Image.fromarray(w_slice, mode='RGB')
            name = 'slice_%d.jpg' % (slc+1)
            save_path = os.path.join(channel_path, name)
            img.save(save_path)


# +---------+------------+--------------+
# |         |   Jester   |    UCF101    |
# +---------+------------+--------------+
# | RN18 3D | 26, 21, 45 | 1000, 21, 40 |
# +---------+------------+--------------+
# | RN18 3T |  31, 20, 8 | 1001, 20, 45 |
# +---------+------------+--------------+
# | GN 3D   | 28, 25, 25 | 1002, 25, 54 |
# +---------+------------+--------------+
# | GN 3T   | 30, 23, 28 | 1003, 23, 12 |
# +---------+------------+--------------+

# visualize_all_first_layer_filters('jester', [31, 20, 8, 0])
# visualize_all_first_layer_filters('ucf101', [1001, 20, 45, 0])
# visualize_all_first_layer_filters('jester', [26, 21, 45, 0])
# visualize_all_first_layer_filters('ucf101', [1000, 21, 40, 0])
# visualize_all_first_layer_filters('jester', [28, 25, 25, 0])
# visualize_all_first_layer_filters('ucf101', [1002, 25, 54, 0])
# visualize_all_first_layer_filters('jester', [30, 23, 28, 0])
# visualize_all_first_layer_filters('ucf101', [1003, 23, 12, 0])