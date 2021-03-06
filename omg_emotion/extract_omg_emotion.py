import os
import skvideo.io
from PIL import Image
from tqdm import tqdm
import relative_baseline.utils as U
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


# convert_to_jpgs(which='Validation', body_part='full_body_background')


# list utterances that were not downloaded
# also creates new labels
def list_missing_utterances(which, save_present=False):
    assert which in ['Validation', 'Training', 'Test']

    base_path = '/scratch/users/gabras/data/omg_emotion'
    _, full_path = get_list_all_videos(os.path.join(base_path, which, 'Videos'))
    full_path = [full_path[i].split('Videos/')[-1] for i in range(len(full_path))]

    annotation_path = os.path.join(base_path, which, 'Annotations', 'annotations.csv')
    annotations = np.genfromtxt(annotation_path, delimiter=',', dtype=str)

    edited_annotations = os.path.join(base_path, which, 'Annotations', 'edited_annotations.csv')

    cnt = 0
    missing_list = os.path.join(base_path, which, 'MissingAnnotations.csv')

    for i in range(len(annotations)):
        name = '%s/%s' % (annotations[i][0], annotations[i][1])

        line = ''
        for j in range(len(annotations[i])):
            line = line + annotations[i][j] + ','
        line = line[0:-1]
        line += '\n'

        if name not in full_path:
            cnt += 1
            f = missing_list
        else:
            f = edited_annotations

        if save_present:
            with open(f, 'a') as my_file:
                my_file.write(line)

    print('%d utterances missing out of %d' % (cnt, len(annotations)))


# list_missing_utterances('Validation')


def distribution_classes_over_splits(which):
    assert which in ['Validation', 'Training', 'Test']

    base_path = '/scratch/users/gabras/data/omg_emotion/%s/Annotations/annotations.csv' % which

    all_annotations = np.genfromtxt(base_path, delimiter=',', dtype=str)

    classes = all_annotations[:, -1]

    classes = np.array([int(classes[i]) for i in range(len(classes))])

    unique, counts = np.unique(classes, return_counts=True)

    class_names = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    d = dict(zip(unique, counts))

    print(which, len(classes))
    cnt = 0
    for i in range(7):
        print(class_names[i], d[i])
        cnt += d[i]
    print(cnt)

    # Training 1955
    # anger 262
    # disgust 96
    # fear 54
    # happy 503
    # neutral 682
    # sad 339
    # surprise 19

    # Validation 481
    # anger 51
    # disgust 34
    # fear 17
    # happy 156
    # neutral 141
    # sad 75
    # surprise 7

    # Test 1989
    # anger 329
    # disgust 135
    # fear 50
    # happy 550
    # neutral 678
    # sad 231
    # surprise 16

#
# for i in ['Test']:
#     distribution_classes_over_splits(i)


def resize_images():
    from relative_baseline.omg_emotion import data_loading as DL
    from relative_baseline.omg_emotion.settings import ProjectVariable
    from relative_baseline.omg_emotion import project_paths as PP

    project_variable = ProjectVariable()

    # splits = ['Training', 'Validation', 'Test']
    splits = ['Test']

    for i in splits:
        cnt = 0
        cnt_d = 0
        labels = DL.load_labels(i, project_variable)
        datapoints = len(labels[0])
        for j in range(datapoints):
            aff = True
            utterance_path = os.path.join(PP.data_path,
                                          i,
                                          PP.omg_emotion_jpg,
                                          labels[0][j][0],
                                          labels[0][j][1].split('.')[0])

            frames = os.listdir(utterance_path)

            for k in range(len(frames)):
                jpg_path = os.path.join(utterance_path, '%d.jpg' % k)
                jpg_as_arr = Image.open(jpg_path)
                if jpg_as_arr.width != 1280 or jpg_as_arr.height != 720:
                    jpg_as_arr = jpg_as_arr.resize((1280, 720))
                    jpg_as_arr.save(jpg_path)
                    cnt += 1
                    if aff:
                        cnt_d += 1
                        print('%d' % j)
                        aff = False

        print('resized in %s, frames: %d, datapoints: %d ' % (i, cnt, cnt_d))


# resize_images()
