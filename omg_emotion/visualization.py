import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from textwrap import wrap
import re
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torchviz import make_dot

l = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
def plot_confusion_matrix(confusion_matrix, labels=l):

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
    c = ax.set_xticklabels(classes, fontsize=16, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=16, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=16, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)
    # summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return fig


def visualize_network(model, file_name, save_location):
    x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
    out = model(x)
    dot = make_dot(out)
    # dot.save('resnet18_emotion.dot', '/huge/gabras/AffectNet/misc')
    dot.save(file_name, save_location)
