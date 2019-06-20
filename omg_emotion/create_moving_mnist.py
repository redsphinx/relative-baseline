import numpy as np
from PIL import Image
import PIL

from relative_baseline.omg_emotion import data_loading as D
from relative_baseline.omg_emotion.settings import ProjectVariable


FRAMES = 30
DTYPE = np.uint8
RESAMPLE = Image.BILINEAR
SIDE = 28


def move_horizontal(image, frames=FRAMES):
    img_x, img_y = SIDE, SIDE
    video = np.zeros((frames, img_x, img_y), dtype=DTYPE)
    right_first = np.random.randint(2)

    velocity_1 = (img_x // 2) // (frames // 2)
    velocity_2 = round(img_x / (frames // 2))

    if right_first: # 1 == move right
        last_loc = 0
        for i in range(frames // 2):
            video[i, :, i*velocity_1:] = image[:, 0:img_x-i*velocity_1]
            last_loc = i*velocity_1
        place = 0
        for i in range(frames // 2, frames):
            loc_v = last_loc - place * velocity_2
            loc_i = img_x - last_loc + place * velocity_2
            if loc_v < 0:
                loc_i = img_x + loc_v
                video[i, :, 0:img_x + loc_v] = image[:, 0:loc_i]
            else:
                video[i, :, loc_v:] = image[:, 0:loc_i]
            place += 1
    else:
        last_loc = 0
        for i in range(frames // 2):
            video[i, :, 0:img_x - i * velocity_1] = image[:, i * velocity_1:]
            last_loc = img_x - i * velocity_1
        place = 0
        place_2 = 0
        for i in range(frames // 2, frames):
            loc_v = last_loc + (place + 1) * velocity_2
            loc_i = img_x - loc_v
            if loc_i < 0:
                video[i, :, (place_2 + 1) * velocity_2:] = image[:, 0:img_x - (place_2 + 1) * velocity_2]
                place_2 += 1
            else:
                video[i, :, 0:loc_v] = image[:, loc_i:]
                place += 1

    return video


def move_vertical(image, frames=FRAMES):
    image = np.rot90(image)
    video = move_horizontal(image, frames)
    video = np.rot90(video, 3, (1, 2))
    return video


def scale(image, frames=FRAMES):
    img_x, img_y = SIDE, SIDE
    video = np.zeros((frames, img_x, img_y), dtype=DTYPE)

    image_pil = Image.fromarray(image)
    place = 0

    for i in range(frames // 2):
        side = img_x - i * 2

        if side < 0:
            side = 0

        image_resized = image_pil.resize((side, side), RESAMPLE)
        canvas = Image.new(image_pil.mode, size=image_pil.size)
        top_left = (i, i)
        canvas.paste(image_resized, top_left)

        video[i] = np.array(canvas, dtype=DTYPE)
        place = i

    for i in range(1, frames // 2 + 1):
        side = i * 2
        image_resized = image_pil.resize((side, side), RESAMPLE)
        canvas = Image.new(image_pil.mode, size=image_pil.size)

        if side > SIDE:
            offset = (side - img_x) // 2
            box = (offset, offset, img_x-offset, img_x-offset)
            image_resized = image_resized.crop(box)
            top_left = (0, 0)
            canvas.paste(image_resized, top_left)
            video[place] = np.array(canvas, dtype=DTYPE)
        else:
            top_left = (img_x // 2 - i, img_x // 2 - i)
            canvas.paste(image_resized, top_left)
            video[place] = np.array(canvas, dtype=DTYPE)

        place += 1

    return video


def rotate(direction, image, frames=FRAMES):
    # direction = -1 is clockwise, 1 is counterclockwise

    img_x, img_y = SIDE, SIDE
    video = np.zeros((frames, img_x, img_y), dtype=DTYPE)
    image_pil = Image.fromarray(image)
    delta_angle = direction * (360 // frames)

    for i in range(frames):
        image_rot = image_pil.rotate(i * delta_angle, resample=RESAMPLE)
        video[i] = np.array(image_rot, dtype=DTYPE)

    return video

def create_moving_mnist():
    '''
    Method to create the moving MNIST dataset
    Moving MNIST is MNIST but each digit class moves in a specific way, for at least 10 frames
    Affine transformations: rotate, scale, translate x y
    0	moves only horizontally
    1	moves only vertically
    2	scales down and then up
    3	rotates clockwise
    4	rotates counter clockwise
    5	moves in circle
    6	scale up while rotating clockwise
    7	moves horizontally while rotating counter clockwise
    8	rotate clockwise and then counter clockwise
    9	random movements
    '''

    # get original mnist

