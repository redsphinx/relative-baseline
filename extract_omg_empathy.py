import os
import skvideo.io
from PIL import Image
from tqdm import tqdm
import utils as U
import numpy as np
from scipy.signal import savgol_filter
from deepimpression2 import constants as C


def video_to_frames(which, path, extract='participant', body_part='full_body_background', extension='jpg', num_frames=None, dims=None):
    assert which in ['Training', 'Validation', 'Test']  # which part of the which
    assert extract in ['participant', 'storyteller', 'all']  # which person to extract
    assert body_part in ['full_body_background', 'full_body_closeup', 'face']  # how to extract part of body
    assert extension in ['jpg', 'png']  # what extension to save frames in
    if num_frames is not None:  # number of frames to extract, if None > extract all frames
        assert type(num_frames) is int
    if dims is not None:  # what final dimensions to save frame in, if None > same as original
        assert type(dims) is tuple

    if body_part == 'full_body_closeup':
        avg_bbox = U.get_avg_body_bbox()
        if avg_bbox[3] > 720:
            avg_bbox[3] = 720
        # avg_bbox = (132, 184, 132+652, 184+535)

    cut = None  # indicates the width to cut

    if extract == 'participant':
        cut = (1280, 2560)  # x1 to x2
    elif extract == 'storyteller':
        cut = (0, 1280)
    elif extract == 'all':
        cut = (0, 2560)


    # if body_part == 'full_body_background':
    #     if extract == 'participant':
    #         cut = (1280, 2560)  # x1 to x2
    #     elif extract == 'storyteller':
    #         cut = (0, 1280)
    #     elif extract == 'all':
    #         cut = (0, 2560)
    # elif body_part == 'full_body_closeup':
    #     if extract == 'participant':
    #         cut = avg_bbox[0], avg_bbox[2]
    # elif body_part == 'face':
    #     raise NotImplemented

    if dims is None:
        # h x w
        if body_part == 'full_body_background':
            dims = (cut[1]-cut[0], 720)
        elif body_part == 'full_body_closeup':
            # dims = None
            dims = (avg_bbox[2]-avg_bbox[0], avg_bbox[3]-avg_bbox[1])

    p = path + '/' + which + '/Videos'
    save_location = path + '/' + which + '/' + extension + '_' + extract + '_' + str(dims[0]) + '_' + str(dims[1])

    if not os.path.exists(save_location):
        os.mkdir(save_location)

    # for mp4 in path
    all_videos = os.listdir(p)

    done_videos = os.listdir(save_location)

    all_videos = list(set(all_videos) - set(done_videos))

    for mp4 in all_videos:
        # create jpg folder
        jpg_folder = save_location + '/' + mp4.split('.mp4')[0]

        if not os.path.exists(jpg_folder):
            os.mkdir(jpg_folder)
            print('%s created' % jpg_folder)

            # read in video, shape: (8675, 720, 2560, 3)
            if num_frames is None:
                num_frames = 0

            print('extracting %s ...' % mp4)
            mp4_arr = skvideo.io.vread(p + '/' + mp4, num_frames=num_frames)

            # for each number of frames, in video
            num_frames = mp4_arr.shape[0]

            # _num = 100
            # for i in tqdm(range(_num)):
            for i in tqdm(range(num_frames)):
                frame = mp4_arr[i]
                name_img = jpg_folder + '/' + str(i) + '.' + extension

                # save frame as extension in size dims
                frame_img = Image.fromarray(frame)
                # frame_img.size = (2560, 720)

                # crop image
                frame_img = frame_img.crop((cut[0], 0, cut[1], 720))  # left, upper, right, and lower

                # crop further
                if body_part == 'full_body_closeup':
                    frame_img = frame_img.crop((avg_bbox[0], avg_bbox[1], avg_bbox[2], avg_bbox[3]))  # left, upper, right, and lower

                frame_img = frame_img.resize((dims[0], dims[1]))
                frame_img.save(name_img, mode='RGB')

            num_frames = None


path_to_data = '/scratch/users/gabras/data/omg_empathy'

# video_to_frames(which='Validation', path=path_to_data, dims=(640, 360))
# video_to_frames(which='Training', path=path_to_data, dims=(640, 360))

# video_to_frames(which='Validation', path=path_to_data)
# video_to_frames(which='Training', path=path_to_data)

# video_to_frames(which='Validation', path=path_to_data, body_part='full_body_closeup')
# video_to_frames(which='Training', path=path_to_data, body_part='full_body_closeup')
# video_to_frames(which='Test', path=path_to_data, body_part='full_body_closeup')


def smooth_labels(data=None, window_size=C.OMG_EMPATHY_FRAME_RATE, order=3):
    if (window_size % 2) == 0:
        window_size = window_size - 1

    if data is None:
        data = ['Training', 'Validation', 'Test']
    else:
        if type(data) == str:
            data = [data]

    for i in data:
        print('smoothing %s' % i)
        src = '/scratch/users/gabras/data/omg_empathy/%s/Annotations' % i
        dst = '/scratch/users/gabras/data/omg_empathy/%s/AnnotationsSmooth_%d_%d' % (i, window_size, order)

        if not os.path.exists(dst):
            os.mkdir(dst)

        subjects = os.listdir(src)

        for name in subjects:
            csv_path_src = os.path.join(src, name)
            labels_src = np.genfromtxt(csv_path_src, skip_header=True, dtype=float)

            labels_smooth = savgol_filter(labels_src, window_size, order)  # window size, polynomial order 3

            csv_path_smooth = os.path.join(dst, name)

            with open(csv_path_smooth, 'w') as my_file:
                for i in range(len(labels_smooth)):
                    line = '%f\n' % labels_smooth[i]
                    my_file.write(line)


smooth_labels(window_size=3*C.OMG_EMPATHY_FRAME_RATE, order=3)
