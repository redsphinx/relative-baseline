import os
import skvideo.io
from PIL import Image
from tqdm import tqdm


def video_to_frames(dataset, path, extract='participant', format='jpg', num_frames=None, dims=None):
    assert dataset in ['Training', 'Validation', 'Test']  # which part of the dataset
    assert extract in ['participant', 'storyteller', 'all']  # which person to extract
    assert format in ['jpg', 'png']  # what format to save frames in
    if num_frames is not None:  # number of frames to extract, if None > extract all frames
        assert type(num_frames) is int
    if dims is not None:  # what final dimensions to save frame in, if None > same as original
        assert type(dims) is tuple

    p = path + '/' + dataset + '/Videos'
    save_location = path + '/' + dataset + '/' + format + '_' + extract + '_' + str(dims[0]) + '_' + str(dims[1])

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
        # h x w
        dims = (cut[1]-cut[0], 720)

    # for mp4 in path
    all_videos = os.listdir(p)

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

            for i in tqdm(range(num_frames)):
                frame = mp4_arr[i]
                name_img = jpg_folder + '/' +str(i) + '.' + format

                # save frame as format in size dims
                frame_img = Image.fromarray(frame)
                # frame_img.size = (2560, 720)
                frame_img = frame_img.crop((cut[0], 0, cut[1], 720)) # left, upper, right, and lower
                frame_img = frame_img.resize((dims[0], dims[1]))
                frame_img.save(name_img, mode='RGB')

            num_frames = None


path_to_data = '/scratch/users/gabras/data/omg_empathy'

# video_to_frames(dataset='Validation', path=path_to_data, dims=(640, 360))
video_to_frames(dataset='Training', path=path_to_data, dims=(640, 360))

