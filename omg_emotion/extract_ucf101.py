import ffmpeg
import numpy as np
import skvideo.io as skvidio
import math
import os
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import shutil
import random
import subprocess
from tqdm import tqdm

from relative_baseline.omg_emotion.utils import opt_mkdir


PATH_UCF101 = '/fast/gabras/ucf101/og_data'
NEW_PATH_UCF101 = '/fast/gabras/ucf101/data_168_224'
opt_mkdir(NEW_PATH_UCF101)


def resize_frame(image, h, w, c):
    # resize to height of h
    or_w, or_h = image.size
    new_w = int(h * or_w / or_h)
    image = image.resize((new_w, h), resample=Image.BICUBIC) # w, h

    if new_w > w:
        delta_w = (new_w - w) // 2
        delta_w_2 = w + delta_w
        image = image.crop((delta_w, 0, delta_w_2, h))  # l, u, r, d
    elif new_w < w:
        delta_w = (w - new_w) // 2
        image = np.array(image)
        pixel_mean = np.mean(np.mean(image, axis=0), axis=0)
        pixel_mean = np.array(pixel_mean, dtype=int)
        canvas = np.ones(shape=(h, w, c), dtype=np.uint8)
        canvas = canvas * pixel_mean
        # paste it
        canvas[:, delta_w:new_w+delta_w, :] = image
        image = canvas

    image = np.array(image, dtype=np.uint8)
    assert image.shape == (h, w, c)

    return image


def select_frames(num_frames, frames):
    if num_frames < frames:
        missing_frames = frames - num_frames
        dupl_1 = [0] * (missing_frames // 2)
        dupl_2 = [num_frames-1] * (missing_frames - missing_frames // 2)
        dupl_mid = [n for n in range(num_frames)]
        frames_to_copy = dupl_1 + dupl_mid + dupl_2
        assert len(frames_to_copy) == frames

    elif num_frames > frames:

        if num_frames - frames >= num_frames / 2:
            frames_to_keep = [n for n in range(0, num_frames, int(num_frames / frames))]
            frames_to_remove = [a for a in range(num_frames) if a not in frames_to_keep]

        else:
            frames_to_remove = [n for n in range(0, num_frames, int(math.ceil(num_frames / (num_frames - frames))))]

        leftover = num_frames - len(frames_to_remove)

        if leftover < frames:
            random_indices = random.sample(frames_to_remove, k=(frames - leftover))
            for n in random_indices:
                frames_to_remove.remove(n)

            assert num_frames - len(frames_to_remove) == frames

        elif leftover > frames:
            to_add = leftover - frames

            if to_add == 1:
                try:
                    item = frames_to_keep.pop()
                    frames_to_remove.append(item)
                except UnboundLocalError:
                    frames_to_remove.append(frames_to_remove[-1]-1)
            else:
                selection_list = [i for i in range(num_frames)]
                tmp = []
                ind = 0
                while len(tmp) != num_frames:
                    tmp.append(selection_list.pop(ind))
                    if ind == 0:
                        ind = -1
                    else:
                        ind = 0

                for i in range(len(tmp)):
                    if tmp[i] not in frames_to_remove:
                        selection_list.append(tmp[i])

                for a_t in range(to_add):
                    frames_to_remove.append(selection_list[a_t])

            frames_to_remove.sort()


        frames_to_copy = [n for n in range(num_frames)]
        for n in frames_to_remove:
            frames_to_copy.remove(n)

        assert len(frames_to_copy) == frames

    else:
        frames_to_copy = [n for n in range(num_frames)]

    return frames_to_copy


def vidwrite(fn, images, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width,channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()


def standardize_clips(b, e, h=168, w=224, frames=30):
    print(b, e)
    all_classes = os.listdir(PATH_UCF101)
    all_classes.sort()

    all_classes = all_classes[b:e]

    for cl in tqdm(all_classes):
        print('\n'
              'extracting class %s...' % cl)

        og_class_path = os.path.join(PATH_UCF101, cl)
        new_class_path = os.path.join(NEW_PATH_UCF101, cl)
        opt_mkdir(new_class_path)

        all_video_clips = os.listdir(og_class_path)
        all_video_clips.sort()

        for vid in all_video_clips:

            og_vid_path = os.path.join(og_class_path, vid)
            new_vid_path = os.path.join(new_class_path, vid)

            og_vid = skvidio.vread(og_vid_path)

            vid_shape = og_vid.shape
            new_vid = np.zeros(shape=(frames, h, w, 3), dtype=np.uint8)
            frames_to_copy = select_frames(vid_shape[0], frames)

            for i, fr in enumerate(frames_to_copy):
                og_frame = Image.fromarray(og_vid[fr], mode='RGB')
                new_frame = resize_frame(og_frame, h, w, 3)
                new_frame = np.array(new_frame, dtype=np.uint8)
                new_vid[i] = new_frame

            # convert to avi and save
            vidwrite(new_vid_path, new_vid)


# standardize_clips(0, 5)
# standardize_clips(5, 10)
# standardize_clips(10, 20)
# standardize_clips(20, 30)
# standardize_clips(30, 40)
# standardize_clips(40, 50)

# standardize_clips(50, 60)
# standardize_clips(60, 70)
# standardize_clips(70, 80)
# standardize_clips(80, 90)
# standardize_clips(90, 110)

