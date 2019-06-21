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


def get_next_x(x_current, y_current, direction, speed):
    x_next = x_current + direction * speed
    if x_next > (SIDE - 1) or x_next < 1:
        direction *= -1
        speed += 1
        x_next = x_current + direction * speed
        if x_next > (SIDE - 1) or x_next < 1:
            print('Something is wrong: x_next=%d' % x_next)
            return 0, direction, speed

    loc = (x_current, 0, x_current + SIDE, SIDE)
    return loc, x_next, y_current, direction, speed


def move(image, move_fun, frames=FRAMES, direction=None):
    '''
    move_fun:   for horizontal/vertical movement: get_next_x
                for circular movement: get_next_pos
    '''
    img_x, img_y = SIDE, SIDE
    video = np.zeros((frames, img_x, img_y), dtype=DTYPE)

    if direction is None:
        right_first = np.random.randint(2)
        if not right_first:
            right_first = -1
    else:
        right_first = direction

    image_pil = Image.fromarray(image)

    velocity = np.random.randint(2) + 1
    # x, y positions at the start
    x_position = SIDE // 2
    y_position = SIDE // 2

    for i in range(frames):
        canvas = Image.new(image_pil.mode, size=(2 * SIDE, SIDE))

        location, x_position, y_position, right_first, velocity = move_fun(x_position, y_position, right_first, velocity)

        canvas.paste(image_pil, location)
        box = (SIDE // 2, 0, SIDE + SIDE // 2, SIDE)
        canvas = canvas.crop(box)

        video[i] = np.array(canvas, dtype=DTYPE)

    return video


def move_horizontal(image, frames=FRAMES, direction=None):
    video = move(image, get_next_x, frames, direction)
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

        canvas = Image.new(image_pil.mode, size=image_pil.size)

        image_rotate = image_pil.rotate(i * delta_angle, resample=RESAMPLE)
        image_resized = image_rotate.resize((side, side), RESAMPLE)

        if side > SIDE:
            offset = (side - img_x) // 2
            top_left = (-offset, -offset)
            canvas.paste(image_resized, top_left)
        else:
            top_left = (img_x // 2 - i, img_x // 2 - i)
            canvas.paste(image_resized, top_left)

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
    # ------------
    # ------------
    right_first = np.random.randint(2)
    if not right_first:
        right_first = -1

    velocity = np.random.randint(2) + 1
    x_position = SIDE // 2
    y_position = SIDE // 2

    for i in range(frames):
        image = video[i]
        image_pil = Image.fromarray(image)
        canvas = Image.new(image_pil.mode, size=(2 * SIDE, SIDE))

        location, x_position, y_position, right_first, velocity = get_next_x(x_position, y_position, right_first,
                                                                           velocity)
        # x_position, right_first, velocity = get_next_x(x_position, right_first, velocity)

        # location = (x_position, 0, x_position + SIDE, SIDE)

        canvas.paste(image_pil, location)
        box = (SIDE // 2, 0, SIDE + SIDE // 2, SIDE)
        canvas = canvas.crop(box)

        video[i] = np.array(canvas, dtype=DTYPE)

    return video


def rotate_clock_and_counter(image, frames=FRAMES):
    video_1 = rotate(direction=-1, image=image, frames=frames//2)
    video_2 = rotate(direction=1, image=image, frames=frames//2)

    img_x, img_y = SIDE, SIDE
    video = np.zeros((frames, img_x, img_y), dtype=DTYPE)

    video[0:15] = video_1
    video[15:] = video_2

    return video


def move_in_circle(image, frames=FRAMES):
    img_x, img_y = SIDE, SIDE
    video = np.zeros((frames, img_x, img_y), dtype=DTYPE)

    direction = np.random.randint(2)
    print('direction: ', direction)

    radius = np.random.randint(4, SIDE//2 -1)
    delta_angle = 360 // frames
    list_angles = np.arange(0, 360, delta_angle)
    x_list = np.round(radius * np.cos(list_angles)) + SIDE
    y_list = np.round(radius * np.sin(list_angles)) + SIDE

    if not direction:
        x_list = np.flip(x_list)
        y_list = np.flip(y_list)

    def to_top_left(x, y):
        x -= SIDE//2
        y -= SIDE//2
        return int(x), int(y)

    image_pil = Image.fromarray(image)
    for i in range(frames):
        canvas = Image.new(image_pil.mode, size=(2 * SIDE, 2 * SIDE))
        top_left = to_top_left(x_list[i], y_list[i])
        canvas.paste(image_pil, top_left)
        box = (SIDE // 2, SIDE // 2, SIDE + SIDE // 2, SIDE + SIDE // 2)
        canvas = canvas.crop(box)
        video[i] = np.array(canvas, dtype=DTYPE)

    return video


def random_transformations(image, frames=FRAMES):
    img_x, img_y = SIDE, SIDE
    image_pil = Image.fromarray(image)

    video = np.zeros((frames, img_x, img_y), dtype=DTYPE)
    video[0] = image

    last_image = image_pil
    angle_memory = []

    # rotate
    for i in range(1, frames):
        direction = np.random.randint(-1, 2)
        delta_angle = direction * (360 // frames)
        angle_memory.append(delta_angle)
        rot_image = image_pil.rotate(sum(angle_memory), resample=RESAMPLE)
        video[i] = np.array(rot_image, dtype=DTYPE)
        # last_image = last_image.rotate(delta_angle, resample=RESAMPLE)
        # video[i] = np.array(last_image, dtype=DTYPE)

    # move
    # sample random movement in x y plane
    x_start = SIDE // 4 # 7
    y_start = SIDE // 4

    x_list = [x_start]
    y_list = [y_start]

    for i in range(1, frames):
        
        next_x = x_list[i-1]+np.random.randint(-1, 2)
        while (SIDE // 2 - 1)  < next_x < 0:
            next_x = x_list[i - 1] + np.random.randint(-1, 2)

        next_y = y_list[i - 1] + np.random.randint(-1, 2)
        while (SIDE // 2 - 1) < next_y < 0:
            next_y = y_list[i - 1] + np.random.randint(-1, 2)
        
        x_list.append(next_x)
        y_list.append(next_y)

    for i in range(frames):
        canvas = Image.new(image_pil.mode, size=(int(1.5 * SIDE), int(1.5 * SIDE)))
        image_pil = Image.fromarray(video[i])

        top_left = (x_list[i], y_list[i])
        canvas.paste(image_pil, top_left)

        box = (SIDE // 4, SIDE // 4, SIDE + SIDE // 4, SIDE + SIDE // 4)
        canvas = canvas.crop(box)
        video[i] = np.array(canvas, dtype=DTYPE)

    return video


def create_moving_mnist(frames=FRAMES):
    '''
    Method to create the moving MNIST dataset
    Moving MNIST is MNIST but each digit class moves in a specific way, for at least 10 frames
    Affine transformations: rotate, scale, translate x y
    0	moves only horizontally
    1	moves only vertically
    2	scales down and then up GOOD
    3	rotates clockwise GOOD
    4	rotates counter clockwise GOOD
    5	moves in circle
    6	scale up while rotating clockwise
    7	moves horizontally while rotating counter clockwise
    8	rotate clockwise and then counter clockwise GOOD
    9	random movements
    '''

    mov_mnist_data_folder = os.path.join(PP.moving_mnist_location, 'debugging_data')
    if not os.path.exists(mov_mnist_data_folder):
        os.mkdir(mov_mnist_data_folder)

    # get original mnist
    mnist_all = D.load_mnist(PV)
    data = np.array(mnist_all[1][0])[0:100]
    labels = np.array(mnist_all[2][0])[0:100]

    # make data
    represent = [7, 8, 1, 0, 2, 3, 4, 11, 18, 61]

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
            video = move_in_circle(image, frames)
        elif lab == 6:
            video = scale_up_rotate_clockwise(image, frames)
        elif lab == 7:
            video = move_horizontal_rotate_counter(image, frames)
        elif lab == 8:
            video = rotate_clock_and_counter(image, frames)
        elif lab == 9:
            video = random_transformations(image, frames)
        else:
            print('Error: lab with value not expected %d' % lab)

        # save arrays as pngs
        folder = os.path.join(mov_mnist_data_folder, '%d' % lab)
        if not os.path.exists(folder):
            os.mkdir(folder)

        for j in range(FRAMES):
            im = Image.fromarray(video[j].astype(DTYPE), mode=MODE)
            path = os.path.join(folder, '%d.png' % j)
            # path = os.path.join(folder, '%d.jpg' % j)
            im.save(path)


create_moving_mnist()
