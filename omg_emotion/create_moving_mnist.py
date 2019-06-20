import os
import numpy as np
from PIL import Image

from relative_baseline.omg_emotion import data_loading as D
from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import project_paths as PP


FRAMES = 30
DTYPE = np.uint8
RESAMPLE = Image.BILINEAR
SIDE = 28
MODE = 'L'

PV = ProjectVariable()
PV.dataset = 'mnist'
PV.train = False
PV.val = False
PV.test = True


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
            if i == 28:
                # TODO: properly fix this; make the image move left again
                video[i] = video[i-2]
            elif i == 29:
                video[i] = video[i - 3]
            else:
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

        if side <= 0:
            side = 1

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


def scale_up_rotate_clockwise(image, frames=FRAMES):
    img_x, img_y = SIDE, SIDE
    video = np.zeros((frames, img_x, img_y), dtype=DTYPE)
    image_pil = Image.fromarray(image)
    delta_angle = -1 * (360 // frames) # -1 for clockwise direction

    for i in range(frames):
        side = i * 2
        if side == 0:
            side = 1

        image_resized = image_pil.resize((side, side), RESAMPLE)
        canvas = Image.new(image_pil.mode, size=image_pil.size)

        if side > SIDE:
            offset = (side - img_x) // 2
            box = (offset, offset, img_x - offset, img_x - offset)
            image_resized = image_resized.crop(box)
            top_left = (0, 0)
            canvas.paste(image_resized, top_left)
        else:
            top_left = (img_x // 2 - i, img_x // 2 - i)
            canvas.paste(image_resized, top_left)

        canvas = canvas.rotate(i * delta_angle, resample=RESAMPLE)
        video[i] = np.array(canvas, dtype=DTYPE)

    return video


def move_horizontal_rotate_counter(image, frames=FRAMES):
    img_x, img_y = SIDE, SIDE
    video = np.zeros((frames, img_x, img_y), dtype=DTYPE)

    # rotate
    image_pil = Image.fromarray(image)
    delta_angle = 1 * (360 // frames)  # 1 for counterclockwise

    for i in range(frames):
        image_rot = image_pil.rotate(i * delta_angle, resample=RESAMPLE)
        video[i] = np.array(image_rot, dtype=DTYPE)
    # ------------    
    
    # move horizontal
    right_first = np.random.randint(2)
    velocity_1 = (img_x // 2) // (frames // 2)
    velocity_2 = round(img_x / (frames // 2))

    print('right first: %d' % right_first)
    if right_first:  # 1 == move right
        last_loc = 0
        for i in range(frames // 2):
            image_arr = video[i]
            video[i, :, i * velocity_1:] = image_arr[:, 0:img_x - i * velocity_1]
            last_loc = i * velocity_1
        place = 0
        for i in range(frames // 2, frames):
            image_arr = video[i]
            loc_v = last_loc - place * velocity_2
            loc_i = img_x - last_loc + place * velocity_2
            if loc_v < 0:
                loc_i = img_x + loc_v
                video[i, :, 0:img_x + loc_v] = image_arr[:, 0:loc_i]
            else:
                video[i, :, loc_v:] = image_arr[:, 0:loc_i]
            place += 1
    else:
        last_loc = 0
        for i in range(frames // 2):
            image_arr = video[i]
            video[i, :, 0:img_x - i * velocity_1] = image_arr[:, i * velocity_1:]
            last_loc = img_x - i * velocity_1
        place = 0
        place_2 = 0
        for i in range(frames // 2, frames):
            if i == 28:
                # TODO: properly fix this; make the image move left again
                video[i] = video[i-2]
            elif i == 29:
                video[i] = video[i - 3]
            else:
                image_arr = video[i]
                loc_v = last_loc + (place + 1) * velocity_2
                loc_i = img_x - loc_v
                if loc_i < 0:
                    video[i, :, (place_2 + 1) * velocity_2:] = image_arr[:, 0:img_x - (place_2 + 1) * velocity_2]
                    place_2 += 1
                else:
                    video[i, :, 0:loc_v] = image_arr[:, loc_i:]
                    place += 1
    
    return video


def rotate_clock_and_counter(image, frames=FRAMES):
    video_1 = rotate(direction=-1, image=image, frames=frames//2)
    video_2 = rotate(direction=1, image=image, frames=frames//2)

    img_x, img_y = SIDE, SIDE
    video = np.zeros((frames, img_x, img_y), dtype=DTYPE)

    video[0:15] = video_1
    video[15:] = video_2

    return video


def move_in_circle():
    pass


def random_transformations(image, frames=FRAMES):
    pass



def create_moving_mnist(frames=FRAMES):
    '''
    Method to create the moving MNIST dataset
    Moving MNIST is MNIST but each digit class moves in a specific way, for at least 10 frames
    Affine transformations: rotate, scale, translate x y
    0	moves only horizontally TODO does not move left (high)
    1	moves only vertically TODO does not move up (high)
    2	scales down and then up GOOD
    3	rotates clockwise GOOD
    4	rotates counter clockwise GOOD
    TODO 5	moves in circle
    6	scale up while rotating clockwise TODO: try rotate and then resize (low priority)
    7	moves horizontally while rotating counter clockwise TODO: doesn't move horizontal (high)
    8	rotate clockwise and then counter clockwise GOOD
    TODO: 9	random movements
    '''

    mov_mnist_data_folder = os.path.join(PP.moving_mnist_location, 'debugging_data')
    if not os.path.exists(mov_mnist_data_folder):
        os.mkdir(mov_mnist_data_folder)

    # get original mnist
    mnist_all = D.load_mnist(PV)
    data = np.array(mnist_all[1][0])[0:100]
    labels = np.array(mnist_all[2][0])[0:100]

    # make data
    represent = [1, 0, 2, 3, 4, 11, 18, 61]

    for i in represent:
        lab = labels[i]
        print(i, lab)
        image = data[i][0]
        video = None

        if lab == 0:
            video = move_horizontal(image, frames)
        elif lab == 1:
            video = move_vertical(image, frames)
        elif lab == 2:
            video = scale(image, frames)
        elif lab == 3:
            video = rotate(-1, image, frames)
        elif lab == 4:
            video = rotate(1, image, frames)
        elif lab == 5:
            pass # TODO
        elif lab == 6:
            video = scale_up_rotate_clockwise(image, frames)
        elif lab == 7:
            video = move_horizontal_rotate_counter(image, frames)
        elif lab == 8:
            video = rotate_clock_and_counter(image, frames)
        elif lab == 9:
            pass # TODO
        else:
            print('Error: lab with value not expected %d' % lab)

        if lab not in [5, 9]:
            # save arrays as jpgs
            folder = os.path.join(mov_mnist_data_folder, '%d' % lab)
            if not os.path.exists(folder):
                os.mkdir(folder)

            for j in range(FRAMES):
                im = Image.fromarray(video[j].astype(DTYPE), mode=MODE)
                path = os.path.join(folder, '%d.jpg' % j)
                im.save(path)




create_moving_mnist()
