import os
import numpy as np
from relative_baseline.omg_emotion import project_paths as PP
import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import dlib
import cv2
from PIL import Image
from PIL.ImageStat import Stat


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


def get_face_bb(path):

    frame = np.array(Image.open(path))
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_rectangles = detector(gray, 2) # top-left(x, y), bot-right(x, y)
    if len(face_rectangles) == 0:
        # print('no face detected in the generated image')
        return None
        # return xp.zeros((image.shape), dtype=xp.uint8)
    largest_face_rectangle = find_largest_face(face_rectangles)
    return largest_face_rectangle


def get_fail_point(which, b, e):
    print('%s, b=%d, e=%d' % (which, b, e))

    side = 96
    total_dp = {'Training': 1955, 'Validation': 481, 'Test': 1989}
    assert (which in ['Training', 'Validation', 'Test'])
    # assert(e < total_dp[which] and b > -1)

    og_data_path = os.path.join(PP.data_path, which, PP.omg_emotion_jpg)
    face_data_path = os.path.join(PP.data_path, which, PP.omg_emotion_jpg_face)

    # get list of all names
    label_path = os.path.join(PP.data_path, which, 'easy_labels.txt')
    all_labels = np.genfromtxt(label_path, delimiter=',', dtype=str)[b:e]
    data_names = all_labels[:, 0:2]

    for i in range(len(data_names)):
        face_utterance_path = os.path.join(face_data_path, data_names[i][0], data_names[i][1])

        # this is the check
        if len(os.listdir(face_utterance_path)) == 0:
            print('!!! fail point is %d\n' % i)
            return i

    print('no fail point\n')

get_fail_point('Training', 0, 2000)
get_fail_point('Validation', 0, 500)
get_fail_point('Test', 0, 2000)

# cropping first 100 frames
def crop_all_faces_in(which, b, e):
    print('%s, b=%d, e=%d\n' % (which, b, e))

    side = 96
    total_dp = {'Training': 1955, 'Validation': 481, 'Test': 1989}
    assert(which in ['Training', 'Validation', 'Test'])
    # assert(e < total_dp[which] and b > -1)

    og_data_path = os.path.join(PP.data_path, which, PP.omg_emotion_jpg)
    face_data_path = os.path.join(PP.data_path, which, PP.omg_emotion_jpg_face)

    # get list of all names
    label_path = os.path.join(PP.data_path, which, 'easy_labels.txt')
    all_labels = np.genfromtxt(label_path, delimiter=',', dtype=str)[b:e]
    data_names = all_labels[:, 0:2]

    for i in range(len(data_names)):

        og_utterance_path = os.path.join(og_data_path, data_names[i][0], data_names[i][1])

        print(og_utterance_path, i, len(data_names))

        face_utterance_path = os.path.join(face_data_path, data_names[i][0], data_names[i][1])
        if not os.path.exists(face_utterance_path):
            os.makedirs(face_utterance_path)

        frames = os.listdir(og_utterance_path)
        frame_names_int = [int(n.split('.')[0]) for n in frames]
        frame_names_int, frames = zip(*sorted(zip(frame_names_int, frames)))

        largest_rectangle = None # (w, h)
        all_mid_points = [] # (x, y)

        frames = frames[:100]

        for f in range(len(frames)):

            pic_path = os.path.join(og_utterance_path, frames[f])
            face_bb = get_face_bb(pic_path)


            if f == 0:
                largest_rectangle = [0, 0]
            else:
                if face_bb is not None:
                    if face_bb.width() > largest_rectangle[0]:
                        largest_rectangle[0] = face_bb.width()
                    if face_bb.height() > largest_rectangle[1]:
                        largest_rectangle[1] = face_bb.height()

            if face_bb is None and f == 0:
                mid_point = [640, 360]

            elif face_bb is None and f != 0:
                mid_point = all_mid_points[-1]

            elif face_bb is not None:
                mid_point = face_bb.center()
                mid_point = [mid_point.x, mid_point.y]
            else:
                mid_point = None

            assert (mid_point is not None)
            all_mid_points.append(mid_point)

        # add 20 pixel border
        largest_rectangle = list(np.array(largest_rectangle) + np.array([20, 20]))
        # make sure it's a square
        largest_side = largest_rectangle[0] if largest_rectangle[0] > largest_rectangle[1] else largest_rectangle[1]
        all_mid_points = np.array(all_mid_points)
        avg_mid_point = [int(np.mean(all_mid_points[:,0])), int(np.mean(all_mid_points[:,1]))]

        for f in range(len(frames)):
            pic_path = os.path.join(og_utterance_path, frames[f])
            frame = np.array(Image.open(pic_path))

            # top = all_mid_points[f][1] - largest_side // 2
            # bot = all_mid_points[f][1] + largest_side // 2
            # left = all_mid_points[f][0] - largest_side // 2
            # right = all_mid_points[f][0] + largest_side // 2

            top =  avg_mid_point[1] - largest_side // 2
            bot =  avg_mid_point[1] + largest_side // 2
            left = avg_mid_point[0] - largest_side // 2
            right =avg_mid_point[0] + largest_side // 2

            if top < 0:
                top = 0
            if bot > frame.shape[0]:
                bot = frame.shape[0]
            if left < 0:
                left = 0
            if right > frame.shape[1]:
                right = frame.shape[1]

            cropped_face = frame[top:bot, left:right]
            # cropped_face = frame[left:right, top:bot]
            cropped_face = Image.fromarray(cropped_face, mode='RGB')

            # resize
            cropped_face_w = cropped_face.width
            cropped_face_h = cropped_face.height

            factor = max(cropped_face_w / side, cropped_face_h / side)
            new_w = int(cropped_face_w / factor)
            assert(new_w <= side)
            new_h = int(cropped_face_h / factor)
            assert (new_h <= side)

            cropped_face = cropped_face.resize((new_w, new_h))

            m1 = np.array(cropped_face)
            mean_pixel = m1.mean(axis=0).mean(axis=0)

            background = Image.new('RGB', (96, 96), (int(mean_pixel[0]), int(mean_pixel[1]), int(mean_pixel[2])))

            offset = [0, 0]

            if new_w != side:
                offset[0] = (side - new_w) // 2
            if new_h != side:
                offset[1] = (side - new_h) // 2

            background.paste(cropped_face, offset)

            save_path = os.path.join(face_utterance_path, frames[f])
            background.save(save_path)


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




# crop_all_faces_in('Validation', 200+13, 300)

# crop_all_faces_in('Training', 0+36, 200)
# crop_all_faces_in('Training', 200+19, 400)
# crop_all_faces_in('Training', 400+114, 600)
# crop_all_faces_in('Training', 600+54, 800)
# crop_all_faces_in('Training', 800+168, 1000)
# crop_all_faces_in('Training', 1000+20, 1200)
# crop_all_faces_in('Training', 1400+124, 1600)

# crop_all_faces_in('Test', 0+62, 200)
# crop_all_faces_in('Test', 400+83, 600)
# crop_all_faces_in('Test', 600+122, 800)
# crop_all_faces_in('Test', 800+31, 1000)
# crop_all_faces_in('Test', 1000+95, 1200)
# crop_all_faces_in('Test', 1200+29, 1400)
# crop_all_faces_in('Test', 1400+107, 1600)
# crop_all_faces_in('Test', 1600+111, 1800)
# crop_all_faces_in('Test', 1800+172, 2000)
