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


def get_list_all_videos(path):
    # base_path = '/scratch/users/gabras/data/omg_emotion'
    # path = os.path.join(base_path, which, 'Videos')

    folders = os.listdir(path)

    all_videos_names = []
    all_videos_full_path = []

    for i, f in enumerate(folders):
        folder_path = os.path.join(path, f)
        videos = os.listdir(folder_path)

        all_videos_names.extend(videos)
        videos = [os.path.join(folder_path, v) for v in videos]
        all_videos_full_path.extend(videos)

    return all_videos_names, all_videos_full_path


def remove_empty_folders(which):
    base_path = '/scratch/users/gabras/data/omg_emotion'
    video_path = os.path.join(base_path, which, 'Videos')

    all_folders = os.listdir(video_path)
    cnt = 0

    for f in all_folders:
        p = os.path.join(video_path, f)
        mp4s = os.listdir(p)
        if len(mp4s) == 0:
            print('removing %s' % p)
            # shutil.rmtree(p)
            cnt += 1

    print('%d empty folders' % cnt)


def convert_to_jpgs(which, body_part, extension='jpg', num_frames=0, dims=None):
    base_path = '/scratch/users/gabras/data/omg_emotion'

    assert which in ['Training', 'Validation', 'Test']  # which part of the which
    assert body_part in ['full_body_background', 'face']  # how to extract part of body
    assert extension in ['jpg', 'png']  # what extension to save frames in
    assert type(num_frames) is int
    if dims is not None:  # what final dimensions to save frame in, if None > same as original
        assert type(dims) is tuple

    if dims is None:
        # h x w
        if body_part == 'full_body_background':
            dims = (1280, 720)

    video_path = os.path.join(base_path, which, 'Videos')
    save_location = os.path.join(base_path, which, '%s_%s_%d_%d' % (extension, body_part, dims[0], dims[1]))

    if not os.path.exists(save_location):
        os.mkdir(save_location)

    # continue where we left off with extraction
    video_names, video_paths = get_list_all_videos(video_path)
    done_names, done_paths = get_list_all_videos(save_location)

    indices_overlap = [video_names.index(i) for i in done_names]
    indices_no_overlap = list(set(range(0, len(video_names))) - set(indices_overlap))

    video_paths = [video_paths[i] for i in indices_no_overlap]

    for mp4 in tqdm(video_paths):
        video_folder_name = mp4.split('/')[-2]
        mp4_name = mp4.split('/')[-1].split('.')[0]
        dst = os.path.join(save_location, video_folder_name, mp4_name)

        if not os.path.exists(dst):
            os.makedirs(dst)

        print('extracting %s ...' % mp4)
        mp4_frames = skvideo.io.vread(mp4, num_frames=num_frames)

        for i in range(len(mp4_frames)):
            frame = mp4_frames[i] # h x w x c
            name_img = os.path.join(dst, '%d.%s' % (i, extension))
            frame_img = Image.fromarray(frame)
            frame_img.save(name_img, mode='RGB')


convert_to_jpgs(which='Test', body_part='full_body_background')
convert_to_jpgs(which='Training', body_part='full_body_background')


# create new annotations file without empty folder lines
def create_new_labels():
    pass
