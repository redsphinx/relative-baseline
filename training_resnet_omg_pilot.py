import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import chainer
import numpy as np
# from deepimpression2.model_53 import Deepimpression
from .model_1 import Siamese
from . import data_loading as L
# import plotting

import deepimpression2.constants as C
from chainer.functions import sigmoid_cross_entropy, mean_absolute_error, softmax_cross_entropy, mean_squared_error
from chainer.optimizers import Adam
import h5py as h5
import deepimpression2.paths as P
# import deepimpression2.chalearn20.data_utils as D
import deepimpression2.chalearn30.data_utils as D
import time
from chainer.backends.cuda import to_gpu, to_cpu
import deepimpression2.util as U
import os
import cupy as cp
from chainer.functions import expand_dims
from random import shuffle
from tqdm import tqdm


my_model = Siamese()

my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8, weight_decay_rate=0.0001)
# my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
my_optimizer.setup(my_model)

if C.ON_GPU:
    my_model = my_model.to_gpu(device=C.DEVICE)

print('Initializing')
print('model initialized with %d parameters' % my_model.count_params())

epochs = 100
batches = 32
# batches = 16
frame_matrix, valid_story_idx_all = L.make_frame_matrix()

train_total_steps = 50
# train_total_steps = 2

val_total_steps = 5

# TODO: decide on principled approach to steps
# train_total_steps = int(160 / batches)  # we have 10 * 4**2 possible pairs of id-stories in training using same person
# train_total_steps = int(1600 / batches)  # we have 40**2 possible pairs of id-stories in training using random people
# train_total_steps = int(100 / batches)  # for debugging
# val_total_steps = int(100 / batches)  # we have 10**2 possible pairs of id-stories in validation


def run(which, model, optimizer, epoch, validation_mode='sequential', model_num=None, experiment_number=None):
    _loss_steps = []
    assert (which in ['train', 'test', 'val'])
    val_idx = None
    steps = None
    if which == 'train':
        val_idx = valid_story_idx_all[0]
        steps = train_total_steps
    elif which == 'val':
        val_idx = valid_story_idx_all[1]
        steps = val_total_steps
    elif which == 'test':
        raise NotImplemented

    # print('%s, steps: %d' % (which, steps))

    if not which == 'val' and validation_mode == 'sequential':
        for s in tqdm(range(steps)):
            # data_left, data_right, labels = L.load_data(which, frame_matrix, val_idx, batches) # deprecated
            data_left, data_right, labels = L.load_data_relative(which, frame_matrix, val_idx, batches,
                                                                 label_mode='difference')

            # labels, data = L.dummy_load_data()  # for debugging purposes only

            if C.ON_GPU:
                data_left = to_gpu(data_left, device=C.DEVICE)
                data_right = to_gpu(data_right, device=C.DEVICE)
                labels = to_gpu(labels, device=C.DEVICE)

            with cp.cuda.Device(C.DEVICE):
                if which == 'train':
                    config = True
                else:
                    config = False

                with chainer.using_config('train', config):
                    if which == 'train':
                        model.cleargrads()
                    prediction = model(data_left, data_right)

                    loss = mean_squared_error(prediction, labels)
                    _loss_steps.append(float(loss.data))

                    if which == 'train':
                        loss.backward()
                        optimizer.update()

        # save model
        plots_folder = 'model_%d_experiment_%d' % (model_num, experiment_number)
        save_location = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
        model_folder = os.path.join(save_location, plots_folder)
        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        name = os.path.join(model_folder, 'epoch_%d' % e)
        chainer.serializers.save_npz(name, my_model)

    else:
        for subject in range(10):
            _loss_steps_subject = []
            previous_prediction = np.array([0.], dtype=np.float32)
            all_predictions = []

            name = 'Subject_%d_Story_1' % (subject+1)
            path = '/scratch/users/gabras/data/omg_empathy/Validation'
            subject_folder = os.path.join(path, 'jpg_participant_662_542', name)
            all_frames = os.listdir(subject_folder)

            full_name = os.path.join(path, 'Annotations', name + '.csv')
            all_labels = np.genfromtxt(full_name, dtype=np.float32, skip_header=True)

            # num_frames = len(all_frames)
            num_frames = 1000

            for f in tqdm(range(num_frames)):
                if f == 0:
                    with cp.cuda.Device(C.DEVICE):

                        with chainer.using_config('train', False):
                            prediction = np.array([0.0], dtype=np.float32)  # baseline
                            labels = np.array([all_labels[f]])

                            loss = mean_squared_error(prediction, labels)
                            _loss_steps_subject.append(float(loss.data))
                else:
                    data_left, data_right = L.get_left_right_consecutively(which, name, f)
                    data_left = np.expand_dims(data_left, 0)
                    data_right = np.expand_dims(data_right, 0)
                    labels = np.array([all_labels[f]])

                    if C.ON_GPU:
                        previous_prediction = to_gpu(previous_prediction, device=C.DEVICE)
                        data_left = to_gpu(data_left, device=C.DEVICE)
                        data_right = to_gpu(data_right, device=C.DEVICE)
                        labels = to_gpu(labels, device=C.DEVICE)

                    with cp.cuda.Device(C.DEVICE):

                        with chainer.using_config('train', False):
                            prediction = model(data_left, data_right)

                            prediction = previous_prediction + prediction

                            loss = mean_squared_error(prediction.data[0], labels)
                            _loss_steps_subject.append(float(loss.data))

                            prediction = float(prediction.data)

                previous_prediction = to_cpu(previous_prediction)

                previous_prediction[0] = float(prediction)
                all_predictions.append(prediction)

            print('loss %s: %f' % (name, float(np.mean(_loss_steps_subject))))
            _loss_steps.append(np.mean(_loss_steps_subject))

            # save graph
            p = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/val/epochs'
            plots_folder = 'model_%d_experiment_%d' % (model_num, experiment_number)
            plot_path = os.path.join(p, plots_folder)
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)

            fig = plt.figure()
            x = range(num_frames)
            plt.plot(x, all_labels[:num_frames], 'g')
            plt.plot(x, all_predictions, 'b')
            plt.savefig(os.path.join(plot_path, '%s_epoch_%d_.png' % (name, epoch)))

    return _loss_steps


print('Enter training loop with validation')
for e in range(0, epochs):
    exp_number = 3
    mod_num = 1
    # ----------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------
    loss_train = run(which='train', model=my_model, optimizer=my_optimizer, model_num=mod_num, experiment_number=exp_number, epoch=e)
    L.update_logs(which='train', loss=float(np.mean(loss_train)), epoch=e, model_num=mod_num, experiment_number=exp_number)
    # L.make_epoch_plot(which)
    # ----------------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------------
    # loss_val = run(which='val', model=my_model, optimizer=my_optimizer, model_num=mod_num, experiment_number=exp_number, epoch=e, validation_mode='sequential')
    # L.update_logs(which='val', loss=float(np.mean(loss_val)), epoch=e, model_num=mod_num, experiment_number=exp_number)
    #
    # print('epoch %d, train_loss: %f, val_loss: %f' % (e, float(np.mean(loss_train)), float(np.mean(loss_val))))
    # ----------------------------------------------------------------------------
    # test
    # ----------------------------------------------------------------------------
    # times = 1
    # for i in range(1):
    #     if times == 1:
    #         ordered = True
    #         save_all_results = True
    #     else:
    #         ordered = False
    #         save_all_results = False
    #
    #     run(which='test', steps=test_steps, which_labels=test_labels, frames=id_frames,
    #         model=my_model, optimizer=my_optimizer, pred_diff=pred_diff_test,
    #         loss_saving=test_loss, which_data=test_on, ordered=ordered, save_all_results=save_all_results,
    #         twostream=True)

    # save model
    # if ((e + 1) % 10) == 0:
    # save_location = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
    # name = os.path.join(save_location, 'epoch_%d_0' % e)
    # chainer.serializers.save_npz(name, my_model)
