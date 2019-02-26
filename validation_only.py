import chainer
import os
import numpy as np
from chainer.functions import mean_squared_error
from tqdm import tqdm
import cupy as cp
import deepimpression2.constants as C
from chainer.backends.cuda import to_gpu, to_cpu
from model_1 import Siamese
from model_2 import Resnet
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import data_loading as L
from scipy.stats import pearsonr
import math
from scipy import stats


# TODO:debug
# predicts the difference between two random pairs
def predict_valence_random_relative(which, steps, batches, data_mix, experiment_number, epoch):
    start_seed = 42
    assert data_mix in ['far', 'close', 'both']  # far = frames are >1 apart, close = frames are 1 apart

    _loss_steps = []

    model_number = 1
    model = Siamese()
    models_path = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
    p = os.path.join(models_path, 'model_%d_experiment_%d' % (model_number, experiment_number), 'epoch_%d' % epoch)
    chainer.serializers.load_npz(p, model)
    if C.ON_GPU:
        model = model.to_gpu(device=C.DEVICE)

    frame_matrix, valid_story_idx_all = L.make_frame_matrix()

    val_idx = None
    if which == 'train':
        val_idx = valid_story_idx_all[0]
    elif which == 'val':
        val_idx = valid_story_idx_all[1]
    elif which == 'test':
        raise NotImplemented

    for s in tqdm(range(steps)):
        data_left, data_right, labels = L.load_data_relative(which, frame_matrix, val_idx, batches, start_seed+s,
                                                             label_mode='difference', data_mix=data_mix)

        if C.ON_GPU:
            data_left = to_gpu(data_left, device=C.DEVICE)
            data_right = to_gpu(data_right, device=C.DEVICE)
            labels = to_gpu(labels, device=C.DEVICE)

        with cp.cuda.Device(C.DEVICE):

            with chainer.using_config('train', False):
                prediction = model(data_left, data_right)

                loss = mean_squared_error(prediction, labels)
                _loss_steps.append(float(loss.data))

    print('%s mean loss: %f' % (which, float(np.mean(_loss_steps))))


# predict_valence_random_relative('val', 10, 16, 'far', 3, 9)


def calculate_ccc(predictions, labels):
    true_mean = np.mean(labels)
    pred_mean = np.mean(predictions)

    rho, _ = pearsonr(predictions, labels)

    if math.isnan(rho):
        rho = 0

    std_predictions = np.std(predictions)

    std_gt = np.std(labels)

    ccc = 2 * rho * std_gt * std_predictions / (
            std_predictions ** 2 + std_gt ** 2 +
            (pred_mean - true_mean) ** 2)

    return ccc


def calculate_pearson(predictions, labels):
    corr_mat, p_vals = stats.pearsonr(predictions, labels)
    return corr_mat, p_vals


# predicts difference two consecutive pairs for first 1000 frame pairs
def predict_valence_sequential_relative(which, experiment_number, epoch):
    use_ccc = False
    use_pearson = True
    _loss_steps = []
    _all_ccc = []
    _all_pearson = []

    model_number = 1
    model = Siamese()
    models_path = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
    p = os.path.join(models_path, 'model_%d_experiment_%d' % (model_number, experiment_number), 'epoch_%d' % epoch)
    chainer.serializers.load_npz(p, model)
    if C.ON_GPU:
        model = model.to_gpu(device=C.DEVICE)

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
        num_frames = 1000
        # num_frames = 10

        for f in range(num_frames):
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

        if use_ccc:
            ccc_subject = calculate_ccc(all_predictions, all_labels[:num_frames])
            _all_ccc.append(ccc_subject)
            print('%s, loss: %f, ccc: %f' % (name, float(np.mean(_loss_steps_subject)), ccc_subject))


        elif use_pearson:
            pearson_subject, p_vals = calculate_pearson(all_predictions, all_labels[:num_frames])
            _all_pearson.append(pearson_subject)
            print('%s, loss: %f, pearson: %f' % (name, float(np.mean(_loss_steps_subject)), pearson_subject))

        _loss_steps.append(np.mean(_loss_steps_subject))

        # save graph
        save_graph = False
        if save_graph:
            p = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/val/epochs'
            plots_folder = 'model_%d_experiment_%d_VO' % (model_number, experiment_number)
            plot_path = os.path.join(p, plots_folder)
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)

            fig = plt.figure()
            x = range(num_frames)
            plt.plot(x, all_labels[:num_frames], 'g')
            plt.plot(x, all_predictions, 'b')
            plt.savefig(os.path.join(plot_path, '%s_epoch_%d_.png' % (name, epoch)))

    if use_ccc:
        print('model_%d_experiment_%d_epoch_%d, val_loss: %f, CCC: %f' % (model_number, experiment_number, epoch,
                                                                          float(np.mean(_loss_steps)),
                                                                          float(np.mean(_all_ccc))))
    elif use_pearson:
        print('model_%d_experiment_%d_epoch_%d, val_loss: %f, pearson: %f' % (model_number, experiment_number, epoch,
                                                                          float(np.mean(_loss_steps)),
                                                                          float(np.mean(_all_pearson))))


predict_valence_sequential_relative('val', 7, 99)


# predicts valence value first 1000 frames
def predict_valence_sequential_single(which, experiment_number, epoch):
    use_ccc = False
    use_pearson = True
    _loss_steps = []
    _all_ccc = []
    _all_pearson = []

    model_number = 2
    model = Resnet()
    models_path = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
    p = os.path.join(models_path, 'model_%d_experiment_%d' % (model_number, experiment_number), 'epoch_%d' % epoch)
    chainer.serializers.load_npz(p, model)
    if C.ON_GPU:
        model = model.to_gpu(device=C.DEVICE)

    for subject in range(10):
        _loss_steps_subject = []
        all_predictions = []

        name = 'Subject_%d_Story_1' % (subject + 1)
        path = '/scratch/users/gabras/data/omg_empathy/Validation'
        subject_folder = os.path.join(path, 'jpg_participant_662_542', name)
        all_frames = os.listdir(subject_folder)

        full_name = os.path.join(path, 'Annotations', name + '.csv')
        all_labels = np.genfromtxt(full_name, dtype=np.float32, skip_header=True)

        # num_frames = len(all_frames)
        num_frames = 1000

        for f in range(num_frames):
            data = L.get_single_consecutively(which, name, f)
            data = np.expand_dims(data, 0)
            labels = np.array([all_labels[f]])

            if C.ON_GPU:
                data = to_gpu(data, device=C.DEVICE)
                labels = to_gpu(labels, device=C.DEVICE)

            with cp.cuda.Device(C.DEVICE):

                with chainer.using_config('train', False):
                    prediction = model(data)

                    loss = mean_squared_error(prediction.data[0], labels)
                    _loss_steps_subject.append(float(loss.data))

            all_predictions.append(float(prediction.data))

        if use_ccc:
            ccc_subject = calculate_ccc(all_predictions, all_labels[:num_frames])
            _all_ccc.append(ccc_subject)
            print('%s, loss: %f, ccc: %f' % (name, float(np.mean(_loss_steps_subject)), ccc_subject))


        elif use_pearson:
            pearson_subject, p_vals = calculate_pearson(all_predictions, all_labels[:num_frames])
            _all_pearson.append(pearson_subject)
            print('%s, loss: %f, pearson: %f' % (name, float(np.mean(_loss_steps_subject)), pearson_subject))

        _loss_steps.append(np.mean(_loss_steps_subject))

        # save graph
        save_graph = False
        if save_graph:
            p = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/val/epochs'
            plots_folder = 'model_%d_experiment_%d_VO' % (model_number, experiment_number)
            plot_path = os.path.join(p, plots_folder)
            if not os.path.exists(plot_path):
                os.mkdir(plot_path)

            fig = plt.figure()
            x = range(num_frames)
            plt.plot(x, all_labels[:num_frames], 'g')
            plt.plot(x, all_predictions, 'b')
            plt.savefig(os.path.join(plot_path, '%s_epoch_%d_.png' % (name, epoch)))


    if use_ccc:
        print('model_%d_experiment_%d_epoch_%d, val_loss: %f, CCC: %f' % (model_number, experiment_number, epoch,
                                                                          float(np.mean(_loss_steps)),
                                                                          float(np.mean(_all_ccc))))
    elif use_pearson:
        print('model_%d_experiment_%d_epoch_%d, val_loss: %f, pearson: %f' % (model_number, experiment_number, epoch,
                                                                          float(np.mean(_loss_steps)),
                                                                          float(np.mean(_all_pearson))))


predict_valence_sequential_single('val', 6, 99)
