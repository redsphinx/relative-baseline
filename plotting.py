import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import deepimpression2.constants as C


def plot_labels_for_subject(which, subject, story, window_size=None, order=None, label_type='discrete'):
    assert label_type in ['discrete', 'smooth']
    name = 'Subject_%d_Story_%d' % (subject, story)
    path = '/scratch/users/gabras/data/omg_empathy/%s' % which

    if label_type == 'smooth':
        full_name = os.path.join(path, 'AnnotationsSmooth_%d_%d' % (window_size, order), name + '.csv')
        all_labels = np.genfromtxt(full_name, dtype=np.float32, skip_header=True)[:1000]
    else:
        full_name = os.path.join(path, 'Annotations', name + '.csv')
        all_labels = np.genfromtxt(full_name, dtype=np.float32, skip_header=True)[:1000]

    save_path = '/scratch/users/gabras/data/omg_empathy/saving_data/label_plots'

    fig = plt.figure(figsize=(16, 6), dpi=120)
    x = range(len(all_labels))
    plt.plot(x, all_labels, 'g')
    plt.title('%s' % name)

    if label_type == 'smooth':
        plt.savefig(os.path.join(save_path, '%s_smooth_%d_%d.png' % (name, window_size, order)))
    else:
        plt.savefig(os.path.join(save_path, '%s.png' % name))

    del fig


for i in range(1, 11):
    plot_labels_for_subject('Validation', i, 1)
    plot_labels_for_subject('Validation', i, 1, window_size=75, order=2, label_type='smooth')


def plot_loss(which, model, experiment, loss_name):
    assert which in ['train', 'val', 'test']
    path = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/%s/epochs/model_%d_experiment_%d.txt' % (which, model, experiment)
    save_path = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/%s/loss_plots/model_%d_experiment_%d.png' % (which, model, experiment)

    loss = np.genfromtxt(path, delimiter=',')[:, 1]

    fig = plt.figure()
    x = range(len(loss))
    plt.plot(x, loss, 'b')
    plt.title('%s %s model %d experiment %d' % (which, loss_name, model, experiment))
    plt.xlabel('epoch')

    plt.ylabel('%s' % loss_name)

    plt.savefig(save_path)


# plot_loss('val', 1, 12, 'MSE loss')
# plot_loss('train', 1, 12, 'MSE loss')
