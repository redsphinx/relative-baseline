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
from model_3 import Triplet
from model_4 import TernaryClassifier
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import data_loading as L
from scipy.stats import pearsonr
import math
from scipy import stats
import utils as U


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
def predict_valence_sequential_relative(which, experiment_number, epoch, time_gap=1, model_number=1,
                                        label_type='discrete'):
    use_ccc = True
    use_pearson = True
    _loss_steps = []
    _all_ccc = []
    _all_pearson = []
    _all_mad = []

    if model_number == 1:
        model = Siamese()
    elif model_number == 3:
        model = Triplet()
    elif model_number == 4:
        model = TernaryClassifier()
    models_path = '/scratch/users/gabras/data/omg_empathy/saving_data/models'
    p = os.path.join(models_path, 'model_%d_experiment_%d' % (model_number, experiment_number), 'epoch_%d' % epoch)
    chainer.serializers.load_npz(p, model)
    if C.ON_GPU:
        model = model.to_gpu(device=C.DEVICE)

    for subject in range(10):
        mean_directional_accuracy = 0
        _loss_steps_subject = []
        previous_prediction = np.array([0.], dtype=np.float32)
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

        if model_number == 4:
            all_labels = np.genfromtxt(full_name, dtype=np.float32, skip_header=True)
            all_labels = U.to_ternary(all_labels, time_gap=time_gap)
            all_labels = np.array(all_labels)

        # num_frames = len(all_frames)
        # num_frames = 1000
        num_frames = 50

        if time_gap > 1:
            _b = C.OMG_EMPATHY_FRAME_RATE
            _e = _b + num_frames
        else:
            _b = 0
            _e = num_frames

        # for f in tqdm(range(_b, _e)):
        for f in range(_b, _e):
            if f == 0:
                with cp.cuda.Device(C.DEVICE):

                    with chainer.using_config('train', False):
                        prediction = np.array([0.0], dtype=np.float32)  # baseline
                        labels = np.array([all_labels[f]])

                        loss = mean_squared_error(prediction, labels)
                        _loss_steps_subject.append(float(loss.data))
            else:
                data_left, data_right = L.get_left_right_consecutively(which, name, f, time_gap=time_gap)
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
                        if model_number == 3:
                            pred_1, pred_2 = model(data_left, data_right)

                            pred_1 = chainer.functions.sigmoid(pred_1)

                            pred_1 = U.threshold_all(to_cpu(pred_1.data))

                            prediction = to_gpu(pred_1, device=C.DEVICE) * pred_2

                            prediction = previous_prediction + prediction

                            loss = mean_squared_error(prediction.data[0], labels)
                            _loss_steps_subject.append(float(loss.data))

                            prediction = float(prediction.data)
                        else:
                            prediction = model(data_left, data_right)

                            if model_number == 4:
                                prediction = chainer.functions.softmax(prediction)
                                prediction = U.threshold_ternary(to_cpu(prediction.data))
                                prediction = np.array(prediction)
                                prediction = to_gpu(prediction, device=C.DEVICE)

                                all_predictions.append(int(prediction))

                                if prediction == labels:
                                    mean_directional_accuracy += 1
                            else:

                                prediction = previous_prediction + prediction

                                loss = mean_squared_error(prediction.data[0], labels)
                                _loss_steps_subject.append(float(loss.data))

                                prediction = float(prediction.data)

            if model_number == 4:
                pass
                # mad_subject = mean_directional_accuracy / num_frames
                # _all_mad.append(mad_subject)
            else:
                previous_prediction = to_cpu(previous_prediction)

                previous_prediction[0] = float(prediction)
                all_predictions.append(prediction)

        if model_number == 4:
            ccc_subject = calculate_ccc(all_predictions, all_labels[_b:_e])
            _all_ccc.append(ccc_subject)
            pearson_subject, p_vals = calculate_pearson(all_predictions, all_labels[_b:_e])
            _all_pearson.append(pearson_subject)

            mad_subject = mean_directional_accuracy / num_frames
            _all_mad.append(mad_subject)
            print('mean_directional_accuracy %s: %f, ccc: %f, pearson: %f' % (name, mad_subject, ccc_subject, pearson_subject))

        else:
            if use_ccc:
                ccc_subject = calculate_ccc(all_predictions, all_labels[_b:_e])
                _all_ccc.append(ccc_subject)
                print('%s, loss: %f, ccc: %f' % (name, float(np.mean(_loss_steps_subject)), ccc_subject))

            if use_pearson:
                pearson_subject, p_vals = calculate_pearson(all_predictions, all_labels[_b:_e])
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

    if model_number == 4:
        print('model_%d_experiment_%d_epoch_%d, val_mad: %f, ccc: %f, pearson: %f' % (model_number, experiment_number, epoch,
                                                                              float(np.mean(_all_mad)),
                                                                              float(np.mean(_all_ccc)),
                                                                              float(np.mean(_all_pearson))))
    else:
        if use_ccc:
            print('model_%d_experiment_%d_epoch_%d, val_loss: %f, CCC: %f' % (model_number, experiment_number, epoch,
                                                                              float(np.mean(_loss_steps)),
                                                                              float(np.mean(_all_ccc))))
        if use_pearson:
            print('model_%d_experiment_%d_epoch_%d, val_loss: %f, pearson: %f' % (model_number, experiment_number, epoch,
                                                                              float(np.mean(_loss_steps)),
                                                                              float(np.mean(_all_pearson))))



# for i in range(100):
#     predict_valence_sequential_relative('val', 9, 99, time_gap=1, model_number=3)
# predict_valence_sequential_relative('val', 14, 14, time_gap=1000, model_number=4, label_type='smooth')


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


# predict_valence_sequential_single('val', 6, 99)


def make_pred_vs_y_plot(experiment_number, epoch, which='val', time_gap=1, model_number=1, label_type='discrete'):
    use_ccc = True
    use_pearson = True
    _loss_steps = []
    _all_ccc = []
    _all_pearson = []

    if model_number == 1:
        model = Siamese()
    elif model_number == 3:
        model = Triplet()
    elif model_number == 4:
        model = TernaryClassifier()

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

        if label_type == 'discrete':
            full_name = os.path.join(path, 'Annotations', name + '.csv')
        elif label_type == 'smooth':
            full_name = os.path.join(path, C.SMOOTH_ANNOTATIONS_PATH, name + '.csv')
        all_labels = np.genfromtxt(full_name, dtype=np.float32, skip_header=True)

        # num_frames = len(all_frames)
        num_frames = 50
        # num_frames = 10

        if time_gap > 1:
            _b = C.OMG_EMPATHY_FRAME_RATE
            _e = _b + num_frames
        else:
            _b = 0
            _e = num_frames

        # for f in tqdm(range(_b, _e)):
        for f in range(_b, _e):
            if f == 0:
                with cp.cuda.Device(C.DEVICE):

                    with chainer.using_config('train', False):
                        prediction = np.array([0.0], dtype=np.float32)  # baseline
                        labels = np.array([all_labels[f]])

                        loss = mean_squared_error(prediction, labels)
                        _loss_steps_subject.append(float(loss.data))
            else:
                data_left, data_right = L.get_left_right_consecutively(which, name, f, time_gap=time_gap)
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
                        if model_number == 3:
                            pred_1, pred_2 = model(data_left, data_right)

                            pred_1 = chainer.functions.sigmoid(pred_1)

                            pred_1 = U.threshold_all(to_cpu(pred_1.data))

                            prediction = to_gpu(pred_1, device=C.DEVICE) * pred_2

                            prediction = previous_prediction + prediction

                            loss = mean_squared_error(prediction.data[0], labels)
                            _loss_steps_subject.append(float(loss.data))

                            prediction = float(prediction.data)
                        else:
                            prediction = model(data_left, data_right)

                            prediction = previous_prediction + prediction

                            loss = mean_squared_error(prediction.data[0], labels)
                            _loss_steps_subject.append(float(loss.data))

                            prediction = float(prediction.data)

            previous_prediction = to_cpu(previous_prediction)

            previous_prediction[0] = float(prediction)
            all_predictions.append(prediction)

        # if use_ccc:
        #     ccc_subject = calculate_ccc(all_predictions, all_labels[_b:_e])
        #     _all_ccc.append(ccc_subject)
        #     print('%s, loss: %f, ccc: %f' % (name, float(np.mean(_loss_steps_subject)), ccc_subject))
        #
        # if use_pearson:
        #     pearson_subject, p_vals = calculate_pearson(all_predictions, all_labels[_b:_e])
        #     _all_pearson.append(pearson_subject)
        #     print('%s, loss: %f, pearson: %f' % (name, float(np.mean(_loss_steps_subject)), pearson_subject))
        #
        # _loss_steps.append(np.mean(_loss_steps_subject))

        # save graph
        save_graph = True
        if save_graph:
            labs = all_labels[_b:_e]
            diff_arr = labs - all_predictions

            diff_matrix = np.zeros((num_frames, num_frames))

            for i in range(num_frames):
                diff_matrix[i, i] = diff_arr[i]

            p = '/scratch/users/gabras/data/omg_empathy/saving_data/logs/val/pred_vs_y'
            plots_folder = 'model_%d_experiment_%d' % (model_number, experiment_number)
            plot_path = os.path.join(p, plots_folder)

            if not os.path.exists(plot_path):
                os.mkdir(plot_path)

            fig = plt.figure()

            # assert len(labs) == len(all_predictions)
            # plt.scatter(labs, all_predictions)

            fig, ax = plt.subplots()
            im = ax.imshow(diff_matrix)

            # We want to show all ticks...
            ax.set_xticks(np.arange(len(diff_arr)))
            ax.set_yticks(np.arange(len(diff_arr)))
            # ... and label them with the respective list entries
            # ax.set_xticklabels(farmers)
            # ax.set_yticklabels(vegetables)

            # Rotate the tick labels and set their alignment.
            # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            #          rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            # for i in range(len(diff_arr)):
            #     for j in range(len(diff_arr)):
            #         text = ax.text(j, i, str(diff_matrix[i, j])[0:7], ha="center", va="center", color="w")

            ax.set_title('difference: labels - predictions')
            fig.tight_layout()

            plt.savefig(os.path.join(plot_path, '%s_epoch_%d_.png' % (name, epoch)))

    # if use_ccc:
    #     print('model_%d_experiment_%d_epoch_%d, val_loss: %f, CCC: %f' % (model_number, experiment_number, epoch,
    #                                                                       float(np.mean(_loss_steps)),
    #                                                                       float(np.mean(_all_ccc))))
    # if use_pearson:
    #     print('model_%d_experiment_%d_epoch_%d, val_loss: %f, pearson: %f' % (model_number, experiment_number, epoch,
    #                                                                           float(np.mean(_loss_steps)),
    #                                                                           float(np.mean(_all_pearson))))


# make_pred_vs_y_plot(experiment_number=7, epoch=99, time_gap=1, model_number=1, label_type='discrete')
