import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import relative_baseline.omg_emotion.project_paths as PP

import os
from textwrap import wrap
import re
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torchviz import make_dot
import cv2 as cv
import PIL.Image as Image
from tqdm import tqdm


# def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
def plot_confusion_matrix(confusion_matrix, dataset):

    if dataset == 'omg_emotion':
        labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    elif dataset in ['mnist', 'mov_mnist']:
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif dataset == 'kth_actions':
        labels = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    elif dataset == 'dhg':
        labels = ['grab', 'tap', 'expand', 'pinch', 'rotation CW', 'rotation CCW', 'swipe R', 'swipe L',
                  'swipe U', 'swipe D', 'swipe X', 'swipe Y', 'swipe +', 'shake']
    elif dataset == 'jester':
        # labels = ["Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up", "Pushing Hand Away",
        #           "Pulling Hand In", "Sliding Two Fingers Left", "Sliding Two Fingers Right",
        #           "Sliding Two Fingers Down", "Sliding Two Fingers Up", "Pushing Two Fingers Away",
        #           "Pulling Two Fingers In", "Rolling Hand Forward", "Rolling Hand Backward", "Turning Hand Clockwise",
        #           "Turning Hand Counterclockwise", "Zooming In With Full Hand", "Zooming Out With Full Hand",
        #           "Zooming In With Two Fingers", "Zooming Out With Two Fingers", "Thumb Up", "Thumb Down",
        #           "Shaking Hand", "Stop Sign", "Drumming Fingers", "No gesture", "Doing other things"]
        labels = [str(i) for i in range(27)]
    else:
        labels = []

    cm = confusion_matrix

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = plt.Figure(figsize=(6, 6), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=10, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=16, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    # summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return fig

# HERE
def plot_srxy(optional_srxy, which_layer, which_channel):
    datapoints = optional_srxy[which_layer]
    datapoints = datapoints[which_channel]  # should be 2D now
    
    # jester
    image_width = 336
    image_height = 224

    # x = np.arange(1, 5)
    x = len(datapoints[0, 0])

    fig, axs = plt.subplots(3, 1, constrained_layout=True)

    scale_data = []
    for i in range(len(datapoints[:, 0])):
        if i == 0:
            scale_data.append(datapoints[i, 0])
        else:
            scale_data.append(datapoints[i, 0] * scale_data[i - 1])

    # axs[0].plot(x, datapoints[:, 0])
    axs[0].plot(x, scale_data)
    axs[0].set_ylabel('size ratio')
    axs[0].set_title('scale')
    axs[0].grid(True)

    # TODO: how to do polar coordinates
    rotate_data = []
    for i in range(len(datapoints[:, 1])):
        if i == 0:
            rotate_data.append()

    axs[1].plot(x, datapoints[:, 1])
    axs[1].set_ylabel('degrees')
    axs[1].set_title('rotation')
    axs[1].grid(True)

    
    x_location_data = []
    y_location_data = []
    
    for i in range(len(datapoints[:, 0])):
        if i == 0:
            x_location_data.append(0)
            y_location_data.append(0)
        else:
            x_location_data.append(x_location_data[i - 1] + datapoints[i, 2] * image_width)
            y_location_data.append(y_location_data[i - 1] + datapoints[i, 3] * image_height)

    axs[2].plot(x_location_data, y_location_data)
    axs[2].set_title('X and Y location in pixels')

    # axs[2].plot(x, datapoints[:, 2])
    axs[2].set_ylabel('y')
    # axs[2].set_title('move x')
    axs[2].grid(True)
    # axs[3].plot(x, datapoints[:, 3])
    # axs[3].set_ylabel('image portion')
    # axs[3].set_title('move y')
    # axs[3].grid(True)
    axs[0].set_xlabel('x')
    # TODO: time info per point

    # fig.tight_layout()
    fig.suptitle('layer %d channel %d' % (which_layer + 1, which_channel + 1))
    # plt.savefig('this_is_test.jpg')

    return fig


def visualize_network(model, file_name, save_location):
    x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
    out = model(x)
    dot = make_dot(out)
    # dot.save('resnet18_emotion.dot', '/huge/gabras/AffectNet/misc')
    dot.save(file_name, save_location)


def load_og_pacman():
    pacman_path = '/home/gabras/deployed/relative_baseline/omg_emotion/images/pacman.jpg'
    img = cv.imread(pacman_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def make_pacman_frame(pacman_img, matrix):
    # pacman_path = '/home/gabras/deployed/relative_baseline/omg_emotion/images/pacman.jpg'
    # pacman_img = cv.imread(pacman_path)

    # pacman_img = cv.cvtColor(pacman_img, cv.COLOR_RGB2BGR)
    rows, cols = pacman_img.shape

    pacman_trnsfrm = cv.warpAffine(pacman_img, matrix, (cols, rows))

    # pacman_trnsfrm = cv.cvtColor(pacman_trnsfrm, cv.COLOR_BGR2RGB)

    return pacman_trnsfrm

    # img = cv2.imread('messi5.jpg',0)
    # rows,cols = img.shape
    #
    # M = np.float32([[1,0,100],[0,1,50]])
    # dst = cv2.warpAffine(img,M,(cols,rows))

def save_kernels(kernel_vis, og_data, info, all_notable_frames):
    # setting up correct save paths
    if type(info[1]) == list:
        if len(info[1]) == 4:
            folder_name = 'experiment_%d_model_%d_run_%d_epoch_%d' % (info[1][0], info[1][1], info[1][2], info[1][3])
        else:
            print('len info is not 4')
            folder_name = None
    elif type(info[1]) == bool:
        folder_name = 'model_%d_pretrain_%d' % (info[0], int(info[1]))
    else:
        print('unknown behavior, what is list?')
        folder_name = None

    folder_path = os.path.join(PP.our_method, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # saving the images
    if info[0] == 20:
        conv_layers = [i + 1 for i in range(19) if (i + 1) not in [6, 11, 16]]
        conv_layers = [conv_layers[0]]
    elif info[0] == 23:
        conv_layers = [1, 3, 6, 8, 12, 14, 18, 20, 24, 26, 31, 33, 37, 39, 43, 45, 50, 52, 56, 58]
    else:
        conv_layers = None
        print('model number is unknown')

    assert len(conv_layers) == len(kernel_vis)

    for i, layer in tqdm(enumerate(conv_layers)):
        layer_path = os.path.join(folder_path, 'conv_%d' % layer)


        for channel in range(len(kernel_vis[i])):
            channel_path = os.path.join(layer_path, 'channel_%d' % channel)
            if not os.path.exists(channel_path):
                os.makedirs(channel_path)

            # save the notable frame for this channel
            notable_frame = all_notable_frames[i][channel]
            notable_path = os.path.join(channel_path, 'og_frame.jpg')
            frame = og_data[0, :, notable_frame]
            frame = frame.permute(1, 2, 0)
            frame = np.array(frame.data.cpu())
            img = Image.fromarray(frame, mode='RGB')
            img.save(notable_path)

            for kernel_slice in range(len(kernel_vis[i][channel])):
                frame = kernel_vis[i][channel][kernel_slice]
                name = 'k_%d.jpg' % kernel_slice
                slice_path = os.path.join(channel_path, name)

                frame = frame[0].transpose(1, 2, 0)
                img = Image.fromarray(frame, mode='RGB')

                # print(slice_path)

                img.save(slice_path)
