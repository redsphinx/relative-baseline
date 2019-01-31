import os
import skvideo.io
from PIL import Image

# paths
path_to_data = '/scratch/users/gabras/data/omg_empathy'

path_test = path_to_data + '/Test'
path_train = path_to_data + '/Training'
path_val = path_to_data + '/Validation'

path_test_videos = path_test + '/Videos'
path_train_videos = path_train + '/Videos'
path_val_videos = path_val + '/Videos'

path_test_labels = path_test + '/Annotations'
path_train_labels = path_train + '/Annotations'
path_val_labels = path_val + '/Annotations'


def video_to_frames(dataset, extract='participant', path=path_to_data, format='jpg', num_frames=None, dims=None):
    assert dataset in ['Training', 'Validation', 'Test']  # which part of the dataset
    assert extract in ['participant', 'storyteller', 'all']  # which person to extract
    assert format in ['jpg', 'png']  # what format to save frames in
    if num_frames is not None:  # number of frames to extract, if None > extract all frames
        assert num_frames is int
    if dims is not None:  # what final dimensions to save frame in, if None > same as original
        assert dims is tuple

    p = path + '/' + dataset + '/Videos'
    save_location = path + '/' + dataset + '/' + format + '_' + extract

    if not os.path.exists(save_location):
        os.mkdir(save_location)

    if extract == 'participant':
        cut = (1280, 2560)
    elif extract == 'storyteller':
        cut = (0, 1280)
    elif extract == 'all':
        cut = (0, 2560)
    else:
        cut = None

    if dims is None:
        # TODO: h x w ?
        dims = (cut[1]-cut[0], 720)


    # for mp4 in path
    all_videos = os.listdir(p)

    for mp4 in all_videos:
        # create jpg folder
        jpg_folder = save_location + '/' + mp4.split('.mp4')[0]
        if not os.path.exists(jpg_folder):
            os.mkdir(jpg_folder)

        # read in video
        mp4_arr = skvideo.io.vread(path_to_data + '/' + mp4)

        # for each number of frames, in video
        if num_frames is None:
            num_frames = mp4_arr.shape()[0]

        for i in range(num_frames):
            frame = mp4_arr[i]
            name_img = jpg_folder + '/' +str(i) + '.' + format

            # save frame as format in size dims
            frame_img = Image.fromarray(frame)
            frame_img = frame_img.crop((cut[0], 0, cut[1], 720)) # left, upper, right, and lower
            frame_img = frame_img.resize((dims[0], dims[1]))
            # TODO: to uint8
            frame_img.save(name_img)

