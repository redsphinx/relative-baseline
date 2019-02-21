import os
import skvideo.io
from PIL import Image
from tqdm import tqdm
import utils as U
import numpy as np
import shutil


def sort_downloaded_videos_in_splits(which):
    assert which in ['Training', 'Validation', 'Test']

    path = '/scratch/users/gabras/data/omg_emotion'
    dst_path = os.path.join(path, which)
    annotation_base = '/home/gabras/deployed/relative-baseline/OMGEmotionChallenge'

    if which == 'Training':
        _p = 'omg_TrainVideos.csv'
    elif which == 'Validation':
        _p = 'omg_ValidationVideos.csv'
    else:
        _p = 'omg_TestVideos_WithLabels.csv'

    path_annotation = os.path.join(annotation_base, _p)

    all_annotations = np.genfromtxt(path_annotation, delimiter=',', dtype=str, skip_header=True)

    video_names = all_annotations[:, 3]

    for v in tqdm(video_names):
        src = os.path.join(path, v)
        dst = os.path.join(dst_path, v)

        if os.path.exists(src):
            shutil.move(src, dst)


# sort_downloaded_videos_in_splits('Test')

def small_fix(which):
    path = '/scratch/users/gabras/data/omg_emotion/%s' % which
    vs = os.listdir(path)

    for v in vs:
        if v == 'Videos' or v == 'Annotations':
            pass
        else:
            s = os.path.join(path, v)
            d = os.path.join(path, 'Videos', v)
            if os.path.exists(s):
                shutil.move(s, d)


# small_fix('Validation')
# small_fix('Test')


def check_for_overlap_in_splits():
    training_names = set(os.listdir('/scratch/users/gabras/data/omg_emotion/Training/Videos'))
    validation_names = set(os.listdir('/scratch/users/gabras/data/omg_emotion/Validation/Videos'))
    test_names = set(os.listdir('/scratch/users/gabras/data/omg_emotion/Test/Videos'))

    print('validations in training: %d' % (len(training_names.intersection(validation_names))))
    print('test in training: %d' % (len(training_names.intersection(test_names))))
    print('validations in test: %d' % (len(test_names.intersection(validation_names))))

    print(len(training_names), len(validation_names), len(test_names))

    # output:
    # validations in training: 0
    # test in training: 0
    # validations in test: 0
    # 231 60 204


# check_for_overlap_in_splits()


def make_annotations(which):
    annotation_base = '/home/gabras/deployed/relative-baseline/OMGEmotionChallenge'

    if which == 'Training':
        _p = 'omg_TrainVideos.csv'
    elif which == 'Validation':
        _p = 'omg_ValidationVideos.csv'
    else:
        _p = 'omg_TestVideos_WithLabels.csv'

    path_annotation = os.path.join(annotation_base, _p)

    all_annotations = np.genfromtxt(path_annotation, delimiter=',', dtype=str, skip_header=True)

    video_path = '/scratch/users/gabras/data/omg_emotion/%s/Videos' % which
    videos = os.listdir(video_path)

    video_names = all_annotations[:, 3]
    utterances = all_annotations[:, 4]
    arousals = all_annotations[:, 5]
    valences = all_annotations[:, 6]
    emotion = all_annotations[:, 7]

    for i in range(len(video_names)):
        if video_names[i] in videos:
            utters = os.listdir(os.path.join(video_path, video_names[i]))

            if utterances[i] in utters:
                l = '%s,%s,%s,%s,%s\n' % (video_names[i], utterances[i], arousals[i], valences[i], emotion[i])

                save_path = '/scratch/users/gabras/data/omg_emotion/%s/Annotations/annotations.csv' % which
                with open(save_path, 'a') as my_file:
                    my_file.write(l)


# make_annotations('Test')

# TODO: convert mp4s to jpgs
# TODO: extract bbox around person