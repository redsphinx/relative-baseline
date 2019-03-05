import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import chainer
import numpy as np
# from deepimpression2.model_53 import Deepimpression
# from model_1 import Siamese
from model_0 import Siamese
import data_loading as L
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
import utils as U
import validation_only as V


load_model = True

my_model = Siamese()

if load_model:
    m_num = 0
    e_num = 15
    ep = 0
    models_path = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
    p = os.path.join(models_path, 'model_%d_experiment_%d' % (m_num, e_num), 'epoch_%d' % ep)
    chainer.serializers.load_npz(p, my_model)
else:
    ep = 0

my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8, weight_decay_rate=0.0001)
# my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
my_optimizer.setup(my_model)

if C.ON_GPU:
    my_model = my_model.to_gpu(device=C.DEVICE)

print('Initializing')
print('model initialized with %d parameters' % my_model.count_params())

# --------------------------------------------------------------------------------------------
DEBUG = True
# --------------------------------------------------------------------------------------------
if DEBUG:
    batches = 32
    train_total_steps = 50
    epochs = 200
else:
    batches = 32
    train_total_steps = 1600 // batches
    epochs = 100

frame_matrix, valid_story_idx_all = L.make_frame_matrix()

val_total_steps = 50

# TODO: decide on principled approach to steps
# train_total_steps = int(160 / batches)  # we have 10 * 4**2 possible pairs of id-stories in training using same person
# train_total_steps = int(1600 / batches)  # we have 40**2 possible pairs of id-stories in training using random people
# train_total_steps = int(100 / batches)  # for debugging
# val_total_steps = int(100 / batches)  # we have 10**2 possible pairs of id-stories in validation


# train_change_points = U.get_all_change_points('Training', valid_story_idx_all) # (10, 4, 2, ?) list that holds lists


def run(which, model, optimizer, epoch, training_mode='change_points', validation_mode='sequential', model_num=None,
        experiment_number=None, label_mode='difference'):
    _loss_steps = []
    assert (which in ['train', 'test', 'val'])
    assert validation_mode in ['sequential', 'random']
    assert training_mode in ['far', 'close', 'both', 'change_points']  # far = frames are >1 apart, close = frames are 1 apart

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

    if which == 'train' or (which == 'val' and validation_mode == 'random'):
        for s in tqdm(range(steps)):
            if which != 'train':
                if training_mode == 'close':
                    data_left, data_right, labels = L.load_data_relative(which, frame_matrix, val_idx, batches,
                                                                         label_mode=label_mode, data_mix='close',
                                                                         step=s)
                elif training_mode == 'change_points':
                    data_left, data_right, labels = L.load_data_relative(which, frame_matrix, val_idx, batches,
                                                                         label_mode=label_mode, data_mix='change_points',
                                                                         step=s)
            else:
                data_left, data_right, labels = L.load_data_relative(which, frame_matrix, val_idx, batches,
                                                                     label_mode=label_mode, data_mix=training_mode, step=s)

            # data_left, data_right, labels = L.load_data_change_points(s, train_change_points)

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

        if which == 'train' and not DEBUG:
            # save model
            plots_folder = 'model_%d_experiment_%d' % (model_num, experiment_number)
            save_location = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
            model_folder = os.path.join(save_location, plots_folder)
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
            name = os.path.join(model_folder, 'epoch_%d' % e)
            chainer.serializers.save_npz(name, my_model)

        _all_ccc = None

    elif which == 'val' and validation_mode == 'sequential':
        _all_ccc = []
        for subject in range(10):
            _loss_steps_subject = []
            previous_prediction = np.array([0.], dtype=np.float32)
            all_predictions = []

            name = 'Subject_%d_Story_1' % (subject + 1)
            path = '/scratch/users/gabras/data/omg_empathy/Validation'
            subject_folder = os.path.join(path, 'jpg_participant_662_542', name)
            all_frames = os.listdir(subject_folder)

            full_name = os.path.join(path, 'Annotations', name + '.csv')
            all_labels = np.genfromtxt(full_name, dtype=np.float32, skip_header=True)

            # num_frames = len(all_frames)
            if not DEBUG:
                num_frames = 1000
            else:
                num_frames = 50

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


                            if label_mode != 'value':
                                prediction = previous_prediction + prediction

                            loss = mean_squared_error(prediction.data[0], labels)
                            _loss_steps_subject.append(float(loss.data))
                            prediction = float(prediction.data)

                previous_prediction = to_cpu(previous_prediction)

                previous_prediction[0] = float(prediction)
                all_predictions.append(prediction)

            ccc_subject = V.calculate_ccc(all_predictions, all_labels[:num_frames])
            _all_ccc.append(ccc_subject)

            print('loss %s: %f, ccc: %f' % (name, float(np.mean(_loss_steps_subject)), ccc_subject))
            _loss_steps.append(np.mean(_loss_steps_subject))

            if not DEBUG:
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

    return _loss_steps, _all_ccc


print('Enter training loop with validation')
for e in range(ep+1, epochs):
    exp_number = 15
    mod_num = 0
    # ----------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------
    loss_train, _ = run(which='train', model=my_model, optimizer=my_optimizer, model_num=mod_num,
                     experiment_number=exp_number, epoch=e, label_mode='value')
    if not DEBUG:
        L.update_logs(which='train', loss=float(np.mean(loss_train)), epoch=e, model_num=mod_num,
                      experiment_number=exp_number)
    # ----------------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------------
    loss_val, ccc = run(which='val', model=my_model, optimizer=my_optimizer, model_num=mod_num, experiment_number=exp_number,
                   epoch=e, validation_mode='sequential', label_mode='value')
    # loss_val = run(which='val', model=my_model, optimizer=my_optimizer, model_num=mod_num, experiment_number=exp_number,
    #                epoch=e, validation_mode='random', training_mode='change_points', label_mode='value')
    if not DEBUG:
        L.update_logs(which='val', loss=float(np.mean(loss_val)), epoch=e, model_num=mod_num, experiment_number=exp_number)

    # print('epoch %d, train_loss: %f, val_loss: %f' % (e, float(np.mean(loss_train)), float(np.mean(loss_val))))
    print('epoch %d, train_loss: %f, val_loss: %f, ccc: %f' % (e, float(np.mean(loss_train)), float(np.mean(loss_val)), float(np.mean(ccc))))
