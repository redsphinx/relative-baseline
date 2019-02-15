import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_labels_for_subject(which, subject, story):
    name = 'Subject_%d_Story_%d' % (subject, story)
    path = '/scratch/users/gabras/data/omg_empathy/%s' % which

    full_name = os.path.join(path, 'Annotations', name + '.csv')
    all_labels = np.genfromtxt(full_name, dtype=np.float32, skip_header=True)

    save_path = '/scratch/users/gabras/data/omg_empathy/saving_data/label_plots'


    fig = plt.figure()
    x = range(len(all_labels))
    plt.plot(x, all_labels, 'g')
    plt.title('%s' % name)

    plt.savefig(os.path.join(save_path, '%s.png' % name))


# for i in range(1, 11):
#     plot_labels_for_subject('Validation', i, 1)
