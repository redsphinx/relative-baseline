import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import chainer
import numpy as np
# from deepimpression2.model_53 import Deepimpression
from .model_4 import TernaryClassifier
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
import utils as U
import validation_only as V


load_model = True

my_model = TernaryClassifier()

if load_model:
    m_num = 4
    e_num = 14
    ep = 68
    models_path = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
    p = os.path.join(models_path, 'model_%d_experiment_%d' % (m_num, e_num), 'epoch_%d' % ep)
    chainer.serializers.load_npz(p, my_model)
else:
    ep = -1

my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8, weight_decay_rate=0.0001)
# my_optimizer = Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
my_optimizer.setup(my_model)

if C.ON_GPU:
    my_model = my_model.to_gpu(device=C.DEVICE)

print('Initializing')
print('model initialized with %d parameters' % my_model.count_params())

# --------------------------------------------------------------------------------------------
DEBUG = False
# --------------------------------------------------------------------------------------------
if DEBUG:
    batches = 16
    train_total_steps = 2
    epochs = 10
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


def run(which, model, optimizer, epoch, training_mode='close', validation_mode='sequential', model_num=None,
        experiment_number=None, time_gap=1000, label_type='smooth'):

    _loss_steps = []  # or for holding accuracy in the validation step
    assert (which in ['train', 'test', 'val'])
    assert validation_mode in ['sequential', 'random']
    assert training_mode in ['far', 'close', 'both', 'change_points']  # far = frames are >1 apart, close = frames are 1 apart
    # assert label_output in ['single', 'double']  # single= 1 output, double= 2 outputs (class+reg)

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

    if which == 'train':
        _all_mad = []

        for s in tqdm(range(steps)):
            data_left, data_right, labels = L.load_data_relative(which, frame_matrix, val_idx, batches,
                                                                 label_mode='stepwise',
                                                                 data_mix=training_mode,
                                                                 step=s, time_gap=time_gap,
                                                                 label_type=label_type)

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

                    loss = softmax_cross_entropy(prediction, labels)

                    prediction = chainer.functions.softmax(prediction)
                    prediction = U.threshold_ternary_train(to_cpu(prediction.data))
                    prediction = np.array(prediction, dtype=np.float32)
                    prediction = to_gpu(prediction, device=C.DEVICE)

                    mad = U.get_mad(prediction, labels)

                    _all_mad.append(mad)

                    _loss_steps.append(float(loss.data))

                    if which == 'train':
                        loss.backward()
                        optimizer.update()

        _all_mad = np.mean(_all_mad)
        _loss_steps = np.mean(_loss_steps)
        print('classification loss: %f, mean_directional_accuracy: %f' % (_loss_steps, _all_mad))

        # save model
        if not DEBUG:
            plots_folder = 'model_%d_experiment_%d' % (model_num, experiment_number)
            save_location = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
            model_folder = os.path.join(save_location, plots_folder)
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)
            name = os.path.join(model_folder, 'epoch_%d' % e)
            chainer.serializers.save_npz(name, my_model)

        _all_ccc = None
        _all_pearson = None

    else:
        _all_ccc = []
        _all_pearson = []
        _all_mad = []

        for subject in range(10):

            mean_directional_accuracy = 0
            all_predictions = []

            name = 'Subject_%d_Story_1' % (subject + 1)
            path = '/scratch/users/gabras/data/omg_empathy/Validation'
            subject_folder = os.path.join(path, 'jpg_participant_662_542', name)
            all_frames = os.listdir(subject_folder)

            if label_type == 'discrete':
                full_name = os.path.join(path, 'Annotations', name + '.csv')
            elif label_type == 'smooth':
                full_name = os.path.join(path, C.SMOOTH_ANNOTATIONS_PATH, name + '.csv')
            all_labels = np.genfromtxt(full_name, dtype=np.float32, skip_header=True)
            all_labels = U.to_ternary(all_labels, time_gap=time_gap)
            all_labels = np.array(all_labels)

            if DEBUG:
                num_frames = 10
            else:
                # num_frames = len(all_frames)
                num_frames = 1000

            if time_gap > 1:
                _b = C.OMG_EMPATHY_FRAME_RATE
                _e = _b + num_frames
            else:
                _b = 0
                _e = num_frames

            # for f in tqdm(range(_b, _e)):
            for f in range(_b, _e):
                if f == 0:
                    all_predictions.append(1)
                    mean_directional_accuracy += 1
                else:
                    data_left, data_right = L.get_left_right_consecutively(which, name, f, time_gap=time_gap)
                    data_left = np.expand_dims(data_left, 0)
                    data_right = np.expand_dims(data_right, 0)

                    labels = all_labels[f]

                    if C.ON_GPU:
                        # previous_prediction = to_gpu(previous_prediction, device=C.DEVICE)
                        data_left = to_gpu(data_left, device=C.DEVICE)
                        data_right = to_gpu(data_right, device=C.DEVICE)
                        labels = to_gpu(labels, device=C.DEVICE)

                    with cp.cuda.Device(C.DEVICE):

                        with chainer.using_config('train', False):
                            prediction = model(data_left, data_right)
                            prediction = chainer.functions.softmax(prediction)
                            prediction = U.threshold_ternary(to_cpu(prediction.data))
                            prediction = np.array(prediction)
                            prediction = to_gpu(prediction, device=C.DEVICE)

                            all_predictions.append(int(prediction))

                            if prediction == labels:
                                mean_directional_accuracy += 1

            mad_subject = mean_directional_accuracy/num_frames
            _all_mad.append(mad_subject)

            # ccc and pearson
            ccc_subject = V.calculate_ccc(all_predictions, all_labels[_b:_e])
            _all_ccc.append(ccc_subject)
            pearson_subject, p_vals = V.calculate_pearson(all_predictions, all_labels[_b:_e])
            _all_pearson.append(pearson_subject)

            print('mean_directional_accuracy %s: %f, ccc: %f, pearson: %f' % (name, mad_subject, ccc_subject, pearson_subject))

            # if mean_directional_accuracy/num_frames == 1.0:
            #     print(all_labels[_b:_e])
            #     print(all_predictions)
            _loss_steps = None

            # save graph
            if not DEBUG:
                p = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/val/epochs'
                plots_folder = 'model_%d_experiment_%d' % (model_num, experiment_number)
                plot_path = os.path.join(p, plots_folder)
                if not os.path.exists(plot_path):
                    os.mkdir(plot_path)

                fig = plt.figure()
                x = range(num_frames)
                plt.plot(x, all_labels[_b:_e], 'g')
                plt.plot(x, all_predictions, 'b')
                plt.savefig(os.path.join(plot_path, '%s_epoch_%d_.png' % (name, epoch)))
                del fig

    return _loss_steps, _all_mad, _all_ccc, _all_pearson


print('Enter training loop with validation')
for e in range(ep+1, epochs):
    exp_number = 14
    mod_num = 4
    # ----------------------------------------------------------------------------
    # training
    # --------------------------------------------------------------------------loss=float(np.mean(loss_val))--
    loss_train, accuracy_train, _, _ = run(which='train', model=my_model, optimizer=my_optimizer, model_num=mod_num,
                     experiment_number=exp_number, epoch=e)
    if not DEBUG:
        L.update_logs(which='train', loss=loss_train, epoch=e, model_num=mod_num,
                      experiment_number=exp_number, accuracy=accuracy_train)
    # ----------------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------------
    _, accuracy_val, ccc, pearson = run(which='val', model=my_model, optimizer=my_optimizer, model_num=mod_num,
                                        experiment_number=exp_number, epoch=e, validation_mode='sequential')
    if not DEBUG:
        L.update_logs(which='val', epoch=e, model_num=mod_num,
                      experiment_number=exp_number, ccc=float(np.mean(ccc)), pearson=float(np.mean(pearson))
                      , accuracy=float(np.mean(accuracy_val)))

        print('epoch %d, train_loss: %f, train_accuracy: %f, val_accuracy: %f, ccc: %f, pearson: %f'
              % (e, float(loss_train), accuracy_train, float(np.mean(accuracy_val)), float(np.mean(ccc)),float(np.mean(pearson))))
