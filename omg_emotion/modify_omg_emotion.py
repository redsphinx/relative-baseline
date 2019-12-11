import os
import numpy as np
from relative_baseline.omg_emotion import project_paths as PP
import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import dlib # TODO: install this
import cv2
from PIL import Image


def check_for_data_overlap():

    path = PP.data_path
    which = ['Training', 'Test', 'Validation']

    # train, val, test
    uniques = [[], [], []]

    for w in which:
        annotation_path = os.path.join(path, w, 'Annotations/annotations.csv')

        all_annotations = np.genfromtxt(annotation_path, delimiter=',', dtype=str)

        if w == 'Training':
            num = 0
        elif w == 'Test':
            num = 1
        elif w == 'Validation':
            num = 2
        else:
            print('invalid which')

        for i in range(len(all_annotations)):
            tmp_1 = str(all_annotations[i][0])
            tmp_2 = str(all_annotations[i][1])
            tmp = tmp_1 + tmp_2
            uniques[num].append(tmp)

        uniques[num] = set(uniques[num])

    print('overlap train with test: ')
    overlap = uniques[0].intersection(uniques[2])
    if len(overlap) > 0:
        print('OVERLAP!!! :(')
        print(overlap)
    else:
        print('no overlap')

    print('overlap train with val: ')
    overlap = uniques[0].intersection(uniques[1])
    if len(overlap) > 0:
        print('OVERLAP!!! :(')
        print(overlap)
    else:
        print('no overlap')

    print('overlap val with test: ')
    overlap = uniques[1].intersection(uniques[2])
    if len(overlap) > 0:
        print('OVERLAP!!! :(')
        print(overlap)
    else:
        print('no overlap')


def find_largest_face(face_rectangles):
    number_rectangles = len(face_rectangles)

    if number_rectangles == 0:
        return None
    elif number_rectangles == 1:
        return face_rectangles[0]
    else:
        largest = 0
        which_rectangle = None
        for i in range(number_rectangles):
            r = face_rectangles[i]
            # it's a square so only one side needs to be checked
            width = r.right() - r.left()
            if width > largest:
                largest = width
                which_rectangle = i
        # print('rectangle %d is largest with a side of %d' % (which_rectangle, largest))
        return face_rectangles[which_rectangle]


def crop_face(path, side):

    frame = np.array(Image.open(path))
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rectangles = detector(gray, 2)
    if len(face_rectangles) == 0:
        print('no face detected in the generated image')
        return None
        # return xp.zeros((image.shape), dtype=xp.uint8)
    largest_face_rectangle = find_largest_face(face_rectangles)

    # TODO: crop face using the rectangle, make sure it's 96x96
    new_frame = None

    # if no face detected, copy face from previous frame
    if new_frame is None:
        print('no face detected')
        return None
    else:
        new_frame = np.array(new_frame, dtype='uint8')
        return new_frame


# use methods from chalearn
def crop_all_faces_in(which, b, e):
    side = 96
    total_dp = {'Training': 1955, 'Validation': 481, 'Test': 1989}
    assert(which in ['Training', 'Validation', 'Test'])
    assert(e < total_dp[which] and b > -1)

    og_data_path = os.path.join(PP.data_path, which, PP.omg_emotion_jpg)
    face_data_path = os.path.join(PP.data_path, which, PP.omg_emotion_jpg_face)

    # get list of all names
    label_path = os.path.join(PP.data_path, which, 'easy_labels.txt')
    all_labels = np.genfromtxt(label_path, delimiter=',', dtype=str)[b:e]
    data_names = all_labels[:, 0:2]

    for i in range(len(data_names)):
        og_utterance_path = os.path.join(og_data_path, data_names[i][0], data_names[i][1])

        face_utterance_path = os.path.join(face_data_path, data_names[i][0], data_names[i][1])
        if not os.path.exists(face_utterance_path):
            os.makedirs(face_utterance_path)

        frames = os.listdir(og_utterance_path)
        for f in range(len(frames)):
            pic_path = os.path.join(og_utterance_path, frames[f])
            cropped_face = crop_face(pic_path, side) # assume it's array
            # convert to image
            # save image
            save_path = os.path.join(face_utterance_path, frames[f])


def make_easy_labels_from_annotations():
    which = ['Training', 'Test', 'Validation']

    for w in which:
        print(w)

        og_annotation_path = os.path.join(PP.data_path, w, 'Annotations', 'annotations.csv')
        og_annotations = np.genfromtxt(og_annotation_path, delimiter=',', dtype=str)

        new_labels_path = os.path.join(PP.data_path, w, 'easy_labels.txt')

        data_path = os.path.join(PP.data_path, w, 'jpg_full_body_background_1280_720')
        video_names_l1 = os.listdir(data_path)
        video_names_l1.sort()

        for f1 in tqdm.tqdm(video_names_l1):
            f1_path = os.path.join(data_path, f1)
            video_names_l2 = os.listdir(f1_path)
            video_names_l2.sort()
            for f2 in video_names_l2:
                num_frames = len(os.listdir(os.path.join(f1_path, f2)))
                with open(new_labels_path, 'a') as my_file:
                    for i in range(len(og_annotations)):
                        if og_annotations[i][0] == f1 and og_annotations[i][1] == f2 + '.mp4':
                            line = '%s,%s,%d,%s,%s,%s\n' % (f1, f2, num_frames,
                                                            og_annotations[i][2],
                                                            og_annotations[i][3],
                                                            og_annotations[i][4])

                            my_file.write(line)


def get_num_frames(plot_histogram=False):
    which = ['Training', 'Test', 'Validation']

    for w in which:

        new_labels_path = os.path.join(PP.data_path, w, 'easy_labels.txt')
        all_labels = np.genfromtxt(new_labels_path, delimiter=',', dtype=str)
        frames = all_labels[:, 2]
        frames = [int(i) for i in frames]
        frames = np.array(frames)
        print('.\n%s\n'
              'max: %d      min: %d     avg: %f     median: %d' % (w, frames.max(), frames.min(), frames.mean(),
                                                                   np.median(frames)))
        limit = 60
        count_less_than_limit = 0
        short_clips = []
        for i in frames:
            if i < limit:
                count_less_than_limit += 1
                short_clips.append(i)
        print('%d out of %d clips have less than %d frames' % (count_less_than_limit, len(frames), limit))
        print('on average these have %f frames' % np.median(short_clips))


        if plot_histogram:
            fig = plt.figure()
            save_path = '/home/gabras/deployed/relative_baseline/omg_emotion/images/omg_emotion_frames'
            save_path = os.path.join(save_path, '%s_frames_distribution.jpg' % w)
            print(save_path)
            plt.hist(frames, bins=20)
            plt.title('%s frames distribution' % w)
            plt.savefig(save_path)


