from matplotlib.colors import LogNorm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec, transforms
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import subprocess
import time
from datetime import datetime
import skvideo.io as skvid


import torch
from torch.optim import AdamW

from relative_baseline.omg_emotion.settings import ProjectVariable as pv
from relative_baseline.omg_emotion import setup
import relative_baseline.omg_emotion.data_loading as DL
import relative_baseline.omg_emotion.project_paths as PP
from relative_baseline.omg_emotion.utils import opt_mkdir, opt_makedirs
from relative_baseline.omg_emotion.xai_tools.layer_visualization import create_next_frame, normalize
import relative_baseline.omg_emotion.xai_tools.feature_visualization as FV
import relative_baseline.omg_emotion.utils as U

from relative_baseline.omg_emotion.xai_tools.misc_functions import preprocess_image, recreate_image, save_clip
from relative_baseline.omg_emotion.models import deconv_3DTTN, deconv_3D

from relative_baseline.omg_emotion import visualization as VZ

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def wait_for_gpu(wait, device_num=None, threshold=100):

    if wait:
        go = False
        while not go:
            gpu_available = get_gpu_memory_map()
            if gpu_available[device_num] < threshold:
                go = True
            else:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print('%s Waiting for gpu %d...' % (current_time, device_num))
                time.sleep(10)
    else:
        return


def init1(dataset, model):
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
    data[:, 0] = (data[:, 0] - 0.485) / 0.229
    data[:, 1] = (data[:, 1] - 0.456) / 0.224
    data[:, 2] = (data[:, 2] - 0.406) / 0.225
    
    if labels is not None:
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


def find_best_videos(dataset, model, device):
    print('\n running function find_best_videos for %s\n' % (str(model)))

    proj_var = init1(dataset, model)
    proj_var.device = device
    # model num 21, 20, 25, 23
    # dataset jester, ucf101

    my_model = setup.get_model(proj_var)
    device = setup.get_device(proj_var)
    wait_for_gpu(wait=True, device_num=proj_var.device, threshold=9000)
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
        names = os.path.join(PP.jester_location, 'filelist_test_xai_150_224.txt')
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


# find_best_videos('jester', [31, 20, 8, 0], device=0)
# find_best_videos('ucf101', [1001, 20, 45, 0], device=0)
# find_best_videos('jester', [26, 21, 45, 0], device=0)
# find_best_videos('ucf101', [1000, 21, 40, 0], device=0)
# find_best_videos('jester', [28, 25, 25, 0], device=0)
# find_best_videos('ucf101', [1002, 25, 54, 0], device=1)
# find_best_videos('jester', [30, 23, 28, 0], device=1)
# find_best_videos('ucf101', [1003, 23, 12, 0], device=0)

# find_best_videos('jester', [36, 20, 13, 0], device=0)
# find_best_videos('ucf101', [1008, 20, 11, 0], device=0)
# find_best_videos('jester', [33, 23, 33, 0], device=1)
# find_best_videos('ucf101', [1005, 23, 28, 0], device=1)


def save_as_plot(scales, rotations, xs, ys, model, conv, ch, dataset):
    x_axis = np.arange(len(scales)+1)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(9, 3))
    # plt.setp(ax3, adjustable='box', aspect='equal')
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

    fontsize_label = 9
    fontsize_title = 11
    fontsize_suptitle = 12
    fontsize_ticks = 7
    fontsize_anno = 10
    markersize = 4
    linewidth = 1.1
    linecolor = 'b'

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

    ax1 = plt.subplot(gs[0])
    ax1.plot(x_axis, new_scales, 'o-', linewidth=linewidth, markersize=markersize, color=linecolor)
    # ax1.axis('square')
    ax1.set_ylabel('size ratio', fontsize=fontsize_label)
    eps = (max(new_scales) - min(new_scales)) / 10
    plt.ylim(min(new_scales)-eps, max(new_scales)+eps)
    # ax1.set_ylim([min(new_scales), max(new_scales)])
    ax1.set_xlabel('time', fontsize=fontsize_label)
    ax1.xaxis.set_ticks(x_axis)
    # ax1.yaxis.set_ticks()
    # ax1.set_xticks(x_axis, tuple([str(i) for i in x_axis]))
    ax1.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    # ax1.set_title('cumulative scale', fontsize=fontsize_title)
    # ax1.set_aspect('equal')
    # ax1.set(adjustable='box')
    ax1.grid(True)


    ax2 = plt.subplot(gs[1])
    ax2.plot(x_axis, new_rotations, 'o-', linewidth=linewidth, markersize=markersize, color=linecolor)
    # ax2.axis('square')
    ax2.set_ylabel('degrees', fontsize=fontsize_label)
    eps = (max(new_rotations) - min(new_rotations)) / 10
    plt.ylim(min(new_rotations)-eps, max(new_rotations)+eps)
    # ax2.set_ylim([min(new_rotations), max(new_rotations)])
    ax2.set_xlabel('time', fontsize=fontsize_label)
    # ax2.set_xticks(x_axis, tuple([str(i) for i in x_axis]))
    ax2.xaxis.set_ticks(x_axis)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    # ax2.set_title('cumulative rotation', fontsize=fontsize_title)
    # ax2.set_aspect('equal', 'box')
    # ax2.set(adjustable='box')
    ax2.grid(True)

    # txt = ['t'+str(i) for i in range(len(new_scales))]
    txt = [str(i) for i in range(len(new_scales))]
    ax3 = plt.subplot(gs[2])
    ax3.plot(new_xs, new_ys, 'o-', linewidth=linewidth, markersize=markersize, color=linecolor)
    # ax3.set_title('x and y location in pixels', fontsize=fontsize_title)
    ax3.set_ylabel('y', fontsize=fontsize_label)
    ax3.set_xlabel('x', fontsize=fontsize_label)
    # eps_x = (max(new_xs) - min(new_xs)) / 10
    # eps_y = (max(new_ys) - min(new_ys)) / 10
    # plt.xlim(min(new_xs)-eps_x, max(new_xs)+eps_x)
    # plt.ylim(min(new_ys)-eps_y, max(new_ys)+eps_y)

    ax3.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    # ax3.set_aspect('equal', 'box')
    # ax3.set(adjustable='box')
    ax3.grid(True)
    # ax3.axis('square')
    eps = 0
    for i, j in enumerate(txt):
        ax3.annotate(j, xy=(new_xs[i]-eps, new_ys[i]-eps), xytext=(new_xs[i], new_ys[i]), fontsize=fontsize_anno,
                     horizontalalignment='left', verticalalignment='bottom')

    # ax3.quiver(new_xs[:-1], new_ys[:-1], np.array(new_xs[1:])-np.array(new_xs[:-1]),
    #            np.array(new_ys[1:])-np.array(new_ys[:-1]), scale_units='xy', angles='xy', scale=2,
    #            headwidth=8, headlength=7, color='b', linestyle='None')

    if model[1] == 21:
        m = '3D-ResNet18'
    elif model[1] == 20:
        m = '3T-ResNet18'
    elif model[1] == 25:
        m = '3D-GoogLeNet'
    elif model[1] == 23:
        m = '3T-GoogLeNet'

    # fig.suptitle('%s on %s, layer %d channel %d\n' % (m, dataset, conv, ch + 1), fontsize=fontsize_suptitle)

    fig.tight_layout()
    # plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=True)

    p1 = 'exp_%d_mod_%d_ep_%d' % (model[0], model[1], model[2])
    p2 = 'layer_%d_channel_%d.jpg' % (conv, ch)
    save_location = os.path.join(PP.srxy_plots, p1, p2)

    intermediary_path = os.path.join(PP.srxy_plots, p1)
    opt_makedirs(intermediary_path)

    plt.savefig(save_location)


def plot_all_srxy(dataset, model, convlayer=None, channel=None):
    proj_var = init1(dataset, model)
    my_model = setup.get_model(proj_var)
    # device = setup.get_device(proj_var)
    # my_model.cuda(device)

    if convlayer is None:
        if model[1] in [21, 20]: # resnet18
            conv_layers = [i+1 for i in range(20) if (i+1) not in [6, 11, 16]]
        elif model[1] in [26]:
            conv_layers = [60]
        else: # googlenet
            conv_layers = [1, 3, 6, 8, 12, 14, 18, 20, 24, 26, 31, 33, 37, 39, 43, 45, 50, 52, 56, 58]
    else:
        conv_layers = [convlayer]

    for ind in tqdm(conv_layers):

        if channel is None:
            start = 0
            end = getattr(my_model, 'conv%d' % ind)
            end = end.weight.shape[0]
        else:
            start = channel
            end = start + 1

        for ch in range(start, end):
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
# +---------+------------+--------------+
# |         |   Jester   |    UCF101    |
# +---------+------------+--------------+
# | RN18 3T | 36, 20, 13 | 1008, 20, 11 |
# +---------+------------+--------------+
# | GN 3T   | 33, 23, 33 | 1005, 23, 28 |
# +---------+------------+--------------+

# >>> channel count from 1 <<<

# plot_all_srxy('jester', [31, 20, 8, 0])
# plot_all_srxy('jester', [31, 20, 8, 0], convlayer=7, channel=0)
# plot_all_srxy('jester', [31, 20, 8, 0], convlayer=1, channel=2)
# plot_all_srxy('jester', [30, 23, 28, 0], convlayer=12, channel=57)
# plot_all_srxy('jester', [30, 23, 28, 0], convlayer=12, channel=141)
# plot_all_srxy('jester', [30, 23, 28, 0], convlayer=31, channel=141)
# plot_all_srxy('jester', [30, 23, 28, 0], convlayer=50, channel=105)
# plot_all_srxy('jester', [30, 23, 28, 0], convlayer=1, channel=None)
# plot_all_srxy('jester', [37, 26, 5, 0])
# plot_all_srxy('jester', [38, 26, 31, 0])
plot_all_srxy('jester', [39, 26, 0, 0])



def make_scale_rot_plot(scales, rotations, model, mode, layer):
    scale_xmin, scale_xmax = 0.5, 2.0
    scale_ymin, scale_ymax = 0, 10e3
    rot_xmin, rot_xmax = -0.3, 0.3
    rot_ymin, rot_ymax = 0, 10e3

    fontsize_title = 11
    fontsize_ticks = 9
    bins = 50
    linestyle = '--'
    gridcolor = 'lightslategray'
    # barcolor = 'royalblue'
    barcolor = 'darkblue'

    fig1 = plt.figure(figsize=(8, 4))
    gs1 = fig1.add_gridspec(1, 2, width_ratios=[1, 1])
    # gs.update(wspace=0.025, hspace=0.05)

    f1_ax1 = fig1.add_subplot(gs1[0, 0])
    # f1_ax1.grid(True)
    f1_ax1.set_axisbelow(True)
    f1_ax1.grid(b=True, which='major', color=gridcolor, linestyle=linestyle)
    f1_ax1.hist(scales, bins=bins, color=barcolor)
    f1_ax1.set_title('scale', fontsize=fontsize_title)
    f1_ax1.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    plt.yscale('log')
    f1_ax1.set_xlim([scale_xmin, scale_xmax])
    f1_ax1.set_ylim([scale_ymin, scale_ymax])


    f1_ax2 = fig1.add_subplot(gs1[0, 1])
    f1_ax2.set_axisbelow(True)
    f1_ax2.grid(b=True, which='major', color=gridcolor, linestyle=linestyle)
    # f1_ax2.grid(True)
    f1_ax2.hist(rotations, bins=bins, color=barcolor)
    f1_ax2.set_title('rotation', fontsize=fontsize_title)
    f1_ax2.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    plt.yscale('log')
    f1_ax2.set_xlim([rot_xmin, rot_xmax])
    f1_ax2.set_ylim([rot_ymin, rot_ymax])


    fig1.tight_layout()

    p1 = 'exp_%d_mod_%d_ep_%d' % (model[0], model[1], model[2])

    if mode == 'model':
        p2 = 'model_distribution_SR.jpg'
    elif mode == 'layer':
        assert layer is not None
        p2 = 'layer_%d_distribution_SR.jpg' % layer

    save_location = os.path.join(PP.distributions, p1, p2)

    intermediary_path = os.path.join(PP.distributions, p1)
    opt_makedirs(intermediary_path)

    plt.savefig(save_location)


def make_xy_plot(xs, ys, model, mode, layer):
    fontsize_title = 11
    fontsize_ticks = 9
    linestyle = '--'
    gridcolor = 'lightslategray'
    # barcolor = 'royalblue'
    barcolor = 'darkblue'

    xmin, xmax = -0.1, 0.3
    ymin, ymax = -0.2, 0.1

    bins = 50

    fig2 = plt.figure(figsize=(4, 4))
    gs2 = fig2.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=0.025, hspace=0.025)

    f2_ax1 = fig2.add_subplot(gs2[1, 0])
    f2_ax1.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    plt.set_cmap('magma')
    plt.hist2d(xs, ys, bins=bins, norm=LogNorm())
    f2_ax1.set_axisbelow(True)
    f2_ax1.grid(b=True, which='major', color=gridcolor, linestyle=linestyle)
    f2_ax1.set_xlim([xmin, xmax])
    f2_ax1.set_ylim([ymin, ymax])


    f2_ax2 = fig2.add_subplot(gs2[0, 0])
    f2_ax2.set_axisbelow(True)
    f2_ax2.grid(b=True, which='major', color=gridcolor, linestyle=linestyle)
    f2_ax2.hist(xs, bins=bins, color=barcolor)
    f2_ax2.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    f2_ax2.set_xticklabels([])
    plt.yscale('log')
    f2_ax2.set_xlim([xmin, xmax])
    # f2_ax2.set_ylim([ymin, ymax])


    f2_ax3 = fig2.add_subplot(gs2[1, 1])
    # f2_ax3.grid(True)
    f2_ax3.set_axisbelow(True)
    f2_ax3.hist(ys, bins=bins, orientation='horizontal', color=barcolor)

    f2_ax3.grid(b=True, which='major', color=gridcolor, linestyle=linestyle)
    f2_ax3.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
    # f2_ax3.get_yaxis().set_visible(False)
    f2_ax3.set_yticklabels([])
    plt.xscale('log')
    # f2_ax3.set_xlim([xmin, xmax])
    f2_ax3.set_ylim([ymin, ymax])

    plt.colorbar()

    fig2.tight_layout()

    p1 = 'exp_%d_mod_%d_ep_%d' % (model[0], model[1], model[2])

    if mode == 'model':
        p2 = 'model_distribution_XY.jpg'
    elif mode == 'layer':
        assert layer is not None
        p2 = 'layer_%d_distribution_XY.jpg' % layer

    save_location = os.path.join(PP.distributions, p1, p2)

    intermediary_path = os.path.join(PP.distributions, p1)
    opt_makedirs(intermediary_path)

    plt.savefig(save_location)


def make_distribution_plots(scales, rotations, xs, ys, model, mode, layer=None):

    p1 = 'exp_%d_mod_%d_ep_%d' % (model[0], model[1], model[2])

    intermediary_path = os.path.join(PP.distributions, p1)
    opt_makedirs(intermediary_path)

    make_scale_rot_plot(scales, rotations, model, mode, layer)
    make_xy_plot(xs, ys, model, mode, layer)

    _scales = np.abs(1-np.abs(scales))
    print('\n%f,%f' % (float(np.mean(_scales)), float(np.std(_scales))))
    print('%f,%f' % (float(np.mean(np.abs(rotations))), float(np.std(np.abs(rotations)))))
    print('%f,%f' % (float(np.mean(np.abs(xs))), float(np.std(np.abs(xs)))))
    print('%f,%f\n' % (float(np.mean(np.abs(ys))), float(np.std(np.abs(ys)))))



def distribution_plots(dataset, model, mode='model', convlayer=None):
    assert mode in ['model', 'layer']

    print('\nRunning function: distribution_plots for model %s\n' % (str(model)))
    proj_var = init1(dataset, model)
    my_model = setup.get_model(proj_var)
    proj_var.device = None
    device = setup.get_device(proj_var)

    p1 = 'exp_%d_mod_%d_ep_%d' % (model[0], model[1], model[2])
    intermediary_path = os.path.join(PP.distributions, p1)
    opt_makedirs(intermediary_path)

    if mode == 'model':
        if model[1] in [21, 20]: # resnet18
            conv_layers = [i+1 for i in range(20) if (i+1) not in [6, 11, 16]]
        else: # googlenet
            conv_layers = [1, 3, 6, 8, 12, 14, 18, 20, 24, 26, 31, 33, 37, 39, 43, 45, 50, 52, 56, 58]


        scales = []
        rotations = []
        xs = []
        ys = []

        for ind in conv_layers:
            start = 0
            end = getattr(my_model, 'conv%d' % ind)
            end = end.weight.shape[0]

            for ch in range(start, end):
                _conv_name = 'conv%d' % ind
                transformations = getattr(getattr(my_model, _conv_name), 'scale')
                num_transformations = transformations.shape[0]
                transformations = getattr(my_model, _conv_name)

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

        make_distribution_plots(scales, rotations, xs, ys, model, mode, layer=None)

    elif mode == 'layer':

        if convlayer is None:
            if model[1] in [21, 20]: # resnet18
                conv_layers = [i+1 for i in range(20) if (i+1) not in [6, 11, 16]]
            else: # googlenet
                conv_layers = [1, 3, 6, 8, 12, 14, 18, 20, 24, 26, 31, 33, 37, 39, 43, 45, 50, 52, 56, 58]
        else:
            conv_layers = []
            conv_layers.extend(convlayer)


        for ind in tqdm(conv_layers):
            start = 0
            end = getattr(my_model, 'conv%d' % ind)
            end = end.weight.shape[0]

            scales = []
            rotations = []
            xs = []
            ys = []

            for ch in range(start, end):
                _conv_name = 'conv%d' % ind
                transformations = getattr(getattr(my_model, _conv_name), 'scale')
                num_transformations = transformations.shape[0]
                transformations = getattr(my_model, _conv_name)

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

            make_distribution_plots(scales, rotations, xs, ys, model, mode, layer=ind)

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

# +---------+------------+--------------+
# |         |   Jester   |    UCF101    |
# +---------+------------+--------------+
# | RN18 3T | 36, 20, 13 | 1008, 20, 11 |
# +---------+------------+--------------+
# | GN 3T   | 33, 23, 33 | 1005, 23, 28 |
# +---------+------------+--------------+

# 3T jester
# distribution_plots('jester', [30, 23, 28, 0], mode='model', convlayer=None)
# distribution_plots('jester', [31, 20, 8, 0], mode='model', convlayer=None)
# distribution_plots('jester', [36, 20, 13, 0], mode='model', convlayer=None)
# distribution_plots('jester', [33, 23, 33, 0], mode='model', convlayer=None)
# #
# # # 3T ucf101
# distribution_plots('ucf101', [1001, 20, 45, 0], mode='model', convlayer=None)
# distribution_plots('ucf101', [1003, 23, 12, 0], mode='model', convlayer=None)
# distribution_plots('ucf101', [1008, 20, 11, 0], mode='model', convlayer=None)
# distribution_plots('ucf101', [1005, 23, 28, 0], mode='model', convlayer=None)

# scratch
# distribution_plots('jester', [36, 20, 13, 0], mode='model', convlayer=None)
# distribution_plots('ucf101', [1008, 20, 11, 0], mode='model', convlayer=None)
# distribution_plots('jester', [33, 23, 33, 0], mode='model', convlayer=None)
# distribution_plots('ucf101', [1005, 23, 28, 0], mode='model', convlayer=None)


# distribution_plots('ucf101', [1001, 20, 45, 0], mode='model', convlayer=None)
# distribution_plots('jester', [30, 23, 28, 0], mode='model', convlayer=None)
# distribution_plots('ucf101', [1003, 23, 12, 0], mode='model', convlayer=None)
# distribution_plots('jester', [36, 20, 13, 0], mode='model', convlayer=None)
# distribution_plots('ucf101', [1008, 20, 11, 0], mode='model', convlayer=None)
# distribution_plots('jester', [33, 23, 33, 0], mode='model', convlayer=None)
# distribution_plots('ucf101', [1005, 23, 28, 0], mode='model', convlayer=None)


def visualize_all_first_layer_filters(dataset, model):
    print('\nrunning function: visualize_all_first_layer_filters for model %s\n' % (str(model)))
    proj_var = init1(dataset, model)
    my_model = setup.get_model(proj_var)
    proj_var.device = None
    device = setup.get_device(proj_var)

    p1 = 'exp_%d_mod_%d_ep_%d' % (model[0], model[1], model[2])
    intermediary_path = os.path.join(PP.filters_conv1, p1)
    opt_makedirs(intermediary_path)

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


# visualize_all_first_layer_filters('jester', [31, 20, 8, 0])
# visualize_all_first_layer_filters('ucf101', [1001, 20, 45, 0])
# visualize_all_first_layer_filters('jester', [26, 21, 45, 0])
# visualize_all_first_layer_filters('ucf101', [1000, 21, 40, 0])
# visualize_all_first_layer_filters('jester', [28, 25, 25, 0])
# visualize_all_first_layer_filters('ucf101', [1002, 25, 54, 0])
# visualize_all_first_layer_filters('jester', [30, 23, 28, 0])
# visualize_all_first_layer_filters('ucf101', [1003, 23, 12, 0])

# visualize_all_first_layer_filters('jester', [36, 20, 13, 0])
# visualize_all_first_layer_filters('ucf101', [1008, 20, 11, 0])
# visualize_all_first_layer_filters('jester', [33, 23, 33, 0])
# visualize_all_first_layer_filters('ucf101', [1005, 23, 28, 0])


def remove_imagenet_mean_std(data):
    data = data * 1.
    data = data / 255
    data[:, 0] = (data[:, 0] - 0.485) / 0.229
    data[:, 1] = (data[:, 1] - 0.456) / 0.224
    data[:, 2] = (data[:, 2] - 0.406) / 0.225
    return data


def add_imagenet_mean_std(data):
    data[0] = data[0] * 0.229 + 0.485
    data[1] = data[1] * 0.224 + 0.456
    data[2] = data[2] * 0.225 + 0.406
    data = data * 255
    return data


def init_random(h, w, seed, device, mode):
    np.random.seed(seed)
        
    if mode == 'image':
        random_img = np.random.randint(low=0, high=255, size=(1, 3, 1, h, w))
        random_img = remove_imagenet_mean_std(random_img)
        random_img = torch.Tensor(random_img)
        random_img = random_img.cuda(device)
        random_img = FV.rgb_to_lucid_colorspace(random_img[:,:,0], device)  # torch.Size([1, 3, 150, 224])
        random_img = FV.rgb_to_fft(h, w, random_img, device)  # torch.Size([1, 3, 150, 113, 2])
        random_img = torch.nn.Parameter(random_img)  # torch.Size([1, 3, 150, 113, 2])
        random_img.requires_grad = True
        return random_img
        
    elif mode == 'volume':
        random_vol = np.random.randint(low=0, high=255, size=(1, 3, 30, h, w))
        random_vol = remove_imagenet_mean_std(random_vol)
        random_vol = torch.Tensor(random_vol)
        random_vol = random_vol.cuda(device)  # torch.Size([1, 3, 30, 150, 224])
        random_vol = FV.rgb_to_lucid_colorspace(random_vol, device)  # torch.Size([1, 3, 30, 150, 224])
        random_vol = FV.rgb_to_fft(h, w, random_vol, device)  # torch.Size([1, 3, 30, 150, 113, 2])
        volume = torch.nn.ParameterList()
        for _f in range(30):
            volume.append(torch.nn.Parameter(random_vol[:, :, _f]))
        return volume
    

def preprocess(the_input, h, w, mode, device):
    
    if mode == 'image':
        img = FV.fft_to_rgb(h, w, the_input, device)  # torch.Size([1, 3, 150, 224])
        img = FV.lucid_colorspace_to_rgb(img, device)  # torch.Size([1, 3, 150, 224])
        img = torch.sigmoid(img)
        img = FV.normalize(img, device)
        img = FV.lucid_transforms(img, device)
        img = img.unsqueeze(2)  # torch.Size([1, 3, 1, 150, 224])
        random_video = img.repeat(1, 1, 30, 1, 1)  #torch.Size([1, 3, 30, 150, 224])
    elif mode == 'volume':
        random_video = torch.Tensor([])
        random_video = random_video.cuda(device)

        for _f in range(30):
            vid = FV.fft_to_rgb(h, w, the_input[_f], device)
            vid = FV.lucid_colorspace_to_rgb(vid, device)  # torch.Size([1, 3, 150, 224])
            vid = torch.sigmoid(vid)
            vid = FV.normalize(vid, device)
            vid = FV.lucid_transforms(vid, device)
            random_video = torch.cat((random_video, vid.unsqueeze(2)), 2)

    return random_video


def postprocess(h, w, the_input, mode, device):
    if mode == 'image':
        img = FV.fft_to_rgb(h, w, the_input.clone(), device)
        img = FV.lucid_colorspace_to_rgb(img, device)
        img = torch.sigmoid(img)
        img = np.array(img.data.cpu())
        img = img[0]
        return img
    elif mode == 'volume':
        _shape = the_input[0].shape  # 1, 3, 150, 224
        vid = np.zeros(shape=(3, 30, h, w))

        for _f in range(30):
            v = FV.fft_to_rgb(h, w, the_input[_f].clone(), device)  # torch.Size([1, 3, 150, 224])
            v = FV.lucid_colorspace_to_rgb(v, device)
            v = torch.sigmoid(v)
            v = np.array(v.data.cpu())
            v = v[0]  # (3, 150, 224)
            vid[:, _f] = v

        return vid


def save_output(output, mode, p2, ch, me):
    if mode == 'image':
        img = add_imagenet_mean_std(output)
        img = normalize(img)
        img = np.array(img.transpose(1, 2, 0), dtype=np.uint8)
        img = Image.fromarray(img, mode='RGB')
        name = 'chan_%d_step_%d.jpg' % (ch, me)
        path = os.path.join(p2, name)
        img.save(path)

    elif mode == 'volume':
        p3 = os.path.join(p2, 'frames_chan_%d_step_%d' % (ch, me))
        opt_mkdir(p3)
        for _f in range(30):
            img = add_imagenet_mean_std(output[:, _f])
            img = normalize(img)
            img = np.array(img.transpose(1, 2, 0), dtype=np.uint8)  # (150, 224, 3)
            img = Image.fromarray(img, mode='RGB')
            name = 'frame_%d.jpg' % (_f)
            path = os.path.join(p3, name)
            img.save(path)


def activation_maximization_single_channels(dataset, model, begin=0, num_channels=1, seed=6, steps=500, mode='image', gpunum=0, layer_begin=None, single_layer=False):
    assert mode in ['image', 'volume']
    print('\nMODEL %s\n' % (str(model)))
    proj_var = init1(dataset, model)
    my_model = setup.get_model(proj_var)
    proj_var.device = gpunum
    wait_for_gpu(wait=True, device_num=proj_var.device, threshold=9000)
    device = setup.get_device(proj_var)
    my_model.cuda(device)


    p1 = 'exp_%d_mod_%d_ep_%d' % (model[0], model[1], model[2])
    intermediary_path = os.path.join(PP.act_max, p1)
    opt_makedirs(intermediary_path)

    if model[1] in [21, 20]: # resnet18
        conv_layers = [i+1 for i in range(20) if (i+1) not in [6, 11, 16]]
    elif model[1] in [26]: # googlenet with special last layer
        conv_layers = [60]
    else: # googlenet
        conv_layers = [1, 3, 6, 8, 12, 14, 18, 20, 24, 26, 31, 33, 37, 39, 43, 45, 50, 52, 56, 58]

    if layer_begin is not None:
        ind = conv_layers.index(layer_begin)
        conv_layers = conv_layers[ind:]
        if single_layer:
            conv_layers = [conv_layers[0]]


    for ind in conv_layers:
        print('conv layer %d' % ind)
        if num_channels is None:
            end = getattr(my_model, 'conv%d' % ind)
            end = end.weight.shape[0]
        else:
            end = num_channels

        p2 = os.path.join(intermediary_path, 'conv_%d' % ind)
        opt_mkdir(p2)
        
        if dataset == 'jester':
            h = 150
            w = 224
        elif dataset == 'ucf101':
            h = 168
            w = 224

        for ch in range(begin, end):
            
            the_input = init_random(h, w, seed, device, mode)
            if mode == 'image':
                optimizer = AdamW([the_input], lr=0.004, weight_decay=0.1)
            else:
                optimizer = AdamW(the_input, lr=0.004, weight_decay=0.1)

            for me in tqdm(range(steps)):
                
                optimizer.zero_grad()
                
                random_input = preprocess(the_input, h, w, mode, device)

                my_model.eval()
                if proj_var.model_number == 20:
                    prediction = my_model(random_input, proj_var.device, stop_at=ind)
                elif proj_var.model_number in [23, 26]:
                    prediction = my_model(random_input, proj_var.device, ind, False)
                elif proj_var.model_number == 21:
                    prediction = my_model(random_input, stop_at=ind)
                elif proj_var.model_number == 25:
                    prediction = my_model(random_input, ind, False)

                loss = -1 * torch.mean(prediction[0, ch])
                # loss = -1 * torch.mean(prediction[0, ch]**2)
                loss.backward()
                optimizer.step()
                my_model.zero_grad()

                liist = [steps-1]
                if me in liist:
                    output = postprocess(h, w, the_input, mode, device)  # (3, 30, 150, 224)
                    save_output(output, mode, p2, ch, me)

# activation_maximization_single_channels('jester', [28, 25, 25, 0], begin=11, num_channels=12, seed=111, steps=500, mode='volume', gpunum=0, layer_begin=12, single_layer=True)
# activation_maximization_single_channels('jester', [28, 25, 25, 0], begin=140, num_channels=141, seed=111, steps=500, mode='volume', gpunum=0, layer_begin=31, single_layer=True)
# activation_maximization_single_channels('jester', [28, 25, 25, 0], begin=186, num_channels=187, seed=111, steps=500, mode='volume', gpunum=0, layer_begin=50, single_layer=True)

# activation_maximization_single_channels('jester', [30, 23, 28, 0], begin=140, num_channels=141, seed=666, steps=500, mode='image', gpunum=0, layer_begin=12, single_layer=True)
# activation_maximization_single_channels('jester', [30, 23, 28, 0], begin=140, num_channels=141, seed=666, steps=500, mode='image', gpunum=0, layer_begin=31, single_layer=True)
# activation_maximization_single_channels('jester', [30, 23, 28, 0], begin=104, num_channels=105, seed=666, steps=500, mode='image', gpunum=0, layer_begin=50, single_layer=True)

# activation_maximization_single_channels('jester', [37, 26, 5, 0], seed=42, num_channels=None, steps=700, mode='image', gpunum=2)
# activation_maximization_single_channels('jester', [38, 26, 1, 0], seed=42, num_channels=None, steps=700, mode='image', gpunum=2)

# RN18 3T
# DONE activation_maximization_single_channels('jester', [31, 20, 8, 0], begin=1, num_channels=5, mode='image', gpunum=0, seed=66)
# DONE activation_maximization_single_channels('ucf101', [1001, 20, 45, 0], begin=0, num_channels=5, mode='image', gpunum=1, layer_begin=12, seed=6)
# GN 3T
# DONE activation_maximization_single_channels('jester', [30, 23, 28, 0], begin=3, num_channels=5, mode='image', gpunum=0, seed=666, layer_begin=58)
# DONE activation_maximization_single_channels('ucf101', [1003, 23, 12, 0], begin=2, num_channels=5, mode='image', gpunum=0, seed=6666, layer_begin=58)
#
# # RN18 3D
# DONE activation_maximization_single_channels('jester', [26, 21, 45, 0], begin=0, num_channels=5, mode='volume', gpunum=1, seed=1)
# DONE activation_maximization_single_channels('ucf101', [1000, 21, 40, 0], begin=1, num_channels=5, mode='volume', gpunum=0, seed=11, layer_begin=20)
# # GN 3D
# DONE activation_maximization_single_channels('jester', [28, 25, 25, 0], begin=0, num_channels=5, mode='volume', gpunum=1, seed=111, layer_begin=50)
# DONE activation_maximization_single_channels('ucf101', [1002, 25, 54, 0], begin=0, num_channels=5, mode='volume', gpunum=1, seed=1111, layer_begin=26)
#
# # [[ scratch ]]
# # RN18 3T
# DONE activation_maximization_single_channels('jester', [36, 20, 13, 0], begin=0, num_channels=1, mode='image', gpunum=0, seed=66666)
# DONE activation_maximization_single_channels('ucf101', [1008, 20, 11, 0], begin=0, num_channels=5, mode='image', gpunum=1, seed=666666, layer_begin=5)
# # GN 3T
# DONE activation_maximization_single_channels('jester', [33, 23, 33, 0], begin=2, num_channels=5, mode='image', gpunum=1, seed=6666666, layer_begin=58)
# DONE activation_maximization_single_channels('ucf101', [1005, 23, 28, 0], begin=2, num_channels=5, mode='image', gpunum=0, seed=66666666, layer_begin=58)


# combines the results for the relevant categories to produce a ranking of videos
# Note: has not been debugged
def combine_results(list_videos, dataset):
        vid_sets = []
        list_vid_dicts = []
        list_videos.sort()
        for _vid in list_videos:
            path = os.path.join(PP.xai_metadata, _vid)
            vids = np.genfromtxt(path, dtype=str, delimiter=' ')
            vid_dict = {vids[i, 0]: i for i in range(len(vids))}
            list_vid_dicts.append(vid_dict)
            vids = vids[:, 0]
            vids = set(vids)
            vid_sets.append(vids)

        present_in_all_sets = vid_sets[0]
        for i in range(1, len(vid_sets)):
            present_in_all_sets = present_in_all_sets.intersection(vid_sets[i])

        print('number of videos present in all lists: %d' % len(present_in_all_sets))

        if len(present_in_all_sets) == 0:
            print("don't know what to do now...")
        else:
            if dataset == 'jester':
                labels_path = os.path.join(PP.jester_location, 'filelist_test_xai_150_224.txt')
                labels = np.genfromtxt(labels_path, dtype=str, delimiter=' ')
                labels_dict = {labels[i, 0]: labels[i, 1] for i in range(len(labels))}

                vid_score = {}
                score = np.zeros(len(present_in_all_sets))

                # map video to score
                for i, _vid in enumerate(present_in_all_sets):
                    for ind in range(len(list_vid_dicts)):
                        score[i] = score[i] + list_vid_dicts[ind][_vid]

                    vid_score[_vid] = score[i]

                # sort on lowest score -> lower the sore, the higher on the list
                vid_score = {k: v for k, v in sorted(vid_score.items(), reverse=False, key=lambda item: item[1])}

                # create ordered mapping from video to scores and labels
                vid_score_class = {k: (vid_score[k], labels_dict[k]) for k in vid_score.keys()}

                return vid_score_class


def find_top_xai_videos(dataset, prediction_type, model=None, combine=False):
    video_files = os.listdir(PP.xai_metadata)

    if combine:
        which_file = 'high_act_vids-%s_pred-%s' % (prediction_type, dataset)
        chosen_files = [i for i in video_files if which_file in i]
        top_videos = combine_results(chosen_files, dataset)
    else:
        assert model is not None
        # high_act_vids-wrong_pred-jester-exp_36_mod_20_ep_13.txt
        which_file = 'high_act_vids-%s_pred-%s-exp_%d_mod_%d_ep_%d.txt' % (prediction_type, dataset, model[0], model[1],
                                                                           model[2])
        top_videos = np.genfromtxt(os.path.join(PP.xai_metadata, which_file), dtype=str, delimiter=' ')[:, 0]

    return top_videos


# has not been made for the combined mode
def load_videos(list_paths, combine=False):
    data = None
    
    if not combine:
        dataset = list_paths[0].split('/')[3]
        if dataset == 'jester':
            h, w = 150, 224
        elif dataset == 'ucf101':
            h, w = 168, 224

        data = np.zeros(shape=(len(list_paths), 3, 30, h, w), dtype=np.float32)

        for i, _path in enumerate(list_paths):
            videodata = skvid.vread(_path)
            videodata = videodata.transpose(3, 0, 1, 2)
            data[i] = videodata
            
    return data
    

def save_image(nparray, path):
    if nparray.dtype != np.uint8:
        nparray = nparray.astype(np.uint8)

    nparray = nparray.transpose(1, 2, 0)
    nparray = Image.fromarray(nparray, mode='RGB')
    nparray.save(path)


def grad_x_frame(frame_gradient, most_notable_frame, og_datapoint):
    frame_gradient = torch.nn.functional.relu(frame_gradient)
    frame_gradient = np.array(frame_gradient)
    frame_gradient = normalize(frame_gradient, z=1.) * 3
    final = frame_gradient * og_datapoint[:, most_notable_frame]
    # hard light blending mode from GIMP: https://docs.gimp.org/en/gimp-concepts-layer-modes.html
    # M = frame_gradient, I = og_datapoint
    def E1(M, I):
        return 254 - ((254 - 2 * (M - 128)) * (254-I)) / 255

    def E2(M, I):
        return (2 * M * I) / 255

    for i in range(3):
        final[i] = np.where(final[i] > 128,
                            E1(final[i], og_datapoint[i, most_notable_frame]),
                            E2(final[i], og_datapoint[i, most_notable_frame]))

    return final


def save_gradients(dataset, model, mode, prediction_type, begin=0, num_channels=1, num_videos=5, gpunum=0):
    assert mode in ['image', 'volume']
    assert prediction_type in ['correct', 'wrong']

    print('\nRunning function save_gradients for MODEL %s\n' % (str(model)))
    proj_var = init1(dataset, model)
    my_model = setup.get_model(proj_var)
    proj_var.device = gpunum
    wait_for_gpu(wait=True, device_num=proj_var.device, threshold=9000)
    device = setup.get_device(proj_var)
    my_model.cuda(device)

    p1 = 'exp_%d_mod_%d_ep_%d' % (model[0], model[1], model[2])
    intermediary_path = os.path.join(PP.gradient, p1)
    opt_makedirs(intermediary_path)

    top_videos = find_top_xai_videos(dataset, prediction_type, model, combine=False)
    top_videos = top_videos[:num_videos]

    data = load_videos(top_videos)

    if model[1] in [21, 20]: # resnet18
        conv_layers = [7, 12, 17]
    else:  # googlenet
        conv_layers = [12, 31, 50]


    for ind in tqdm(conv_layers):
        
        for vid in range(num_videos):

            p2 = os.path.join(intermediary_path, 'vid_%d' % vid, 'conv_%d' % ind)
            opt_makedirs(p2)
            datapoint_1 = data[vid].copy()
            og_datapoint = data[vid].copy()

            if mode == 'volume':
                vid_path = os.path.join(intermediary_path, 'vid_%d' % vid, 'video')
                opt_mkdir(vid_path)
                for _f in range(30):
                    path = os.path.join(vid_path, 'og_frame_%d.jpg' % _f)
                    save_image(og_datapoint[:, _f], path)

            channels = []
            
            for ch in range(begin, num_channels):

                datapoint = torch.Tensor(datapoint_1.copy()).unsqueeze(0).cuda(device)
                datapoint = remove_imagenet_mean_std(datapoint)
                datapoint = torch.nn.Parameter(datapoint, requires_grad=True)
                
                # get feature map
                if proj_var.model_number == 20:
                    feature_map = my_model(datapoint, proj_var.device, stop_at=ind)
                elif proj_var.model_number == 23:
                    feature_map = my_model(datapoint, proj_var.device, ind, False)
                elif proj_var.model_number == 21:
                    feature_map = my_model(datapoint, stop_at=ind)
                elif proj_var.model_number == 25:
                    feature_map = my_model(datapoint, ind, False)
                    
                _, chan, d, h, w = feature_map.shape
                
                # find the channel with the highest activation
                feature_map_arr = np.array(feature_map[0].clone().data.cpu())
                highest_value = 0
                ind_0, ind_1, ind_2, ind_3 = 0, 0, 0, 0

                for k in range(chan):
                    if k not in channels:
                        # for l in range(d):
                        max_index = feature_map_arr[k].argmax()
                        indices = np.unravel_index(max_index, (d, h, w))
                        if feature_map_arr[k, indices[0], indices[1], indices[2]] > highest_value:
                            highest_value = feature_map_arr[k, indices[0], indices[1], indices[2]]
                            ind_0 = k
                            ind_1, ind_2, ind_3 = indices

                print('\n Conv%d: %d Highest unit value of %f found at channel %d\n' % (ind, ch, highest_value, ind_0))
                channels.append(ind_0)
                
                p3 = os.path.join(p2, 'rank_%d_channel_%d' % (ch, ind_0))
                opt_mkdir(p3)

                # calculate d(max(feature_map)) / d(data)
                feature_map[0, ind_0, ind_1, ind_2, ind_3].backward()
                image_grad = datapoint.grad

                # if in image mode, find the frame with the greatest gradient
                if mode == 'image':
                    copy_image_grad = image_grad[0].clone()
                    copy_image_grad = 255 * copy_image_grad
                    copy_image_grad = torch.nn.functional.relu(copy_image_grad)
                    copy_image_grad = np.array(copy_image_grad.data.cpu(), dtype=np.uint8)

                    copy_image_grad = copy_image_grad.transpose(1, 0, 2, 3)
                    frames, og_channels, h, w = copy_image_grad.shape

                    highest_value = 0
                    ind_1 = 0
                    for l in range(frames):
                        max_index = copy_image_grad[l].argmax()
                        indices = np.unravel_index(max_index, (og_channels, h, w))
                        if copy_image_grad[l, indices[0], indices[1], indices[2]] > highest_value:
                            highest_value = copy_image_grad[l, indices[0], indices[1], indices[2]]
                            ind_1 = l
                    most_notable_frame = ind_1

                    if og_datapoint.dtype == np.uint8:
                        og_datapoint = og_datapoint.astype(np.float32)

                    frame_gradient = image_grad[0, :, most_notable_frame].data.cpu()
                    final = grad_x_frame(frame_gradient, most_notable_frame, og_datapoint)

                    # save the image and the original
                    path = os.path.join(p3, 'grad_x_frame_%d.jpg' % most_notable_frame)
                    save_image(final, path)

                    path = os.path.join(p3, 'og_frame_%d.jpg' % most_notable_frame)
                    save_image(og_datapoint[:, most_notable_frame], path)

                elif mode == 'volume':
                    if og_datapoint.dtype == np.uint8:
                        og_datapoint = og_datapoint.astype(np.float32)

                    og_channels, frames, h, w = image_grad[0].shape
                    for _f in range(frames):
                        frame_gradient = image_grad[0, :, _f].data.cpu()
                        final = grad_x_frame(frame_gradient, _f, og_datapoint)

                        # save the image and the original
                        path = os.path.join(p3, 'grad_x_frame_%d.jpg' % _f)
                        save_image(final, path)

                        # path = os.path.join(p3, 'og_frame_%d.jpg' % _f)
                        # save_image(og_datapoint[:, _f], path)

                my_model.zero_grad()
    print('THE END')

# save_gradients('jester', [28, 25, 25, 0], mode='volume', prediction_type='correct', num_videos=4, num_channels=5, gpunum=0)
# save_gradients('jester', [30, 23, 28, 0], mode='image', prediction_type='correct', num_videos=4, num_channels=5, gpunum=0)
# save_gradients('jester', [33, 23, 33, 0], mode='image', prediction_type='correct', num_videos=4, num_channels=5, gpunum=0)
# save_gradients('ucf101', [1002, 25, 54, 0], mode='volume', prediction_type='correct', num_videos=4, num_channels=5, gpunum=0)
# save_gradients('ucf101', [1003, 23, 12, 0], mode='image', prediction_type='correct', num_videos=4, num_channels=5, gpunum=0)
# save_gradients('ucf101', [1005, 23, 28, 0], mode='image', prediction_type='correct', num_videos=4, num_channels=5, gpunum=0)


# 3D convs
# save_gradients('jester', [26, 21, 45, 0], mode='volume', prediction_type='correct', num_videos=5, num_channels=20)
# save_gradients('jester', [28, 25, 25, 0], mode='volume', prediction_type='correct', num_videos=5, num_channels=20)
# save_gradients('ucf101', [1000, 21, 40, 0], mode='volume', prediction_type='correct', num_videos=5, num_channels=20, gpunum=1)
# save_gradients('ucf101', [1002, 25, 54, 0], mode='volume', prediction_type='correct', num_videos=5, num_channels=20, gpunum=1)

# 3T convs
# save_gradients('jester', [31, 20, 8, 0], mode='image', prediction_type='correct', num_videos=10, num_channels=5, gpunum=1)
# save_gradients('jester', [30, 23, 28, 0], mode='image', prediction_type='correct', num_videos=10, num_channels=5, gpunum=1)
#
# save_gradients('ucf101', [1001, 20, 45, 0], mode='image', prediction_type='correct', num_videos=10, num_channels=5, gpunum=1)
# save_gradients('ucf101', [1003, 23, 12, 0], mode='image', prediction_type='correct', num_videos=10, num_channels=5, gpunum=1)
#
# save_gradients('jester', [36, 20, 13, 0], mode='image', prediction_type='correct', num_videos=10, num_channels=5, gpunum=0)
# save_gradients('jester', [33, 23, 33, 0], mode='image', prediction_type='correct', num_videos=10, num_channels=5, gpunum=0)
#
# save_gradients('ucf101', [1008, 20, 11, 0], mode='image', prediction_type='correct', num_videos=10, num_channels=5, gpunum=1)
# save_gradients('ucf101', [1005, 23, 28, 0], mode='image', prediction_type='correct', num_videos=10, num_channels=5, gpunum=1)

# save_gradients(dataset, model, mode, prediction_type, begin=0, num_channels=1, num_videos=5, gpunum=0)



def quick_load_model(dataset, model, gpunum):
    proj_var = init1(dataset, model)
    proj_var.model_number = 26
    my_model = setup.get_model(proj_var)

    proj_var.device = gpunum
    wait_for_gpu(wait=True, device_num=proj_var.device, threshold=9000)
    device = setup.get_device(proj_var)
    my_model.cuda(device)
    video = ['/fast/gabras/jester/data_150_224_avi/48467.avi']
    data = load_videos(video)
    datapoint = torch.Tensor(data).cuda(device)
    aux1, aux2, prediction = my_model(datapoint, proj_var.device, None, False)


# quick_load_model('jester', [30, 23, 28, 0], 2)
