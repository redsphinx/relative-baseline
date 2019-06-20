import numpy as np


def create_moving_mnist():
    '''
    Method to create the moving MNIST dataset
    Moving MNIST is MNIST but each digit class moves in a specific way, for at least 10 frames
    Affine transformations: rotate, scale, translate x y
    0	moves only horizontally
    1	moves only vertically
    2	scales up then down
    3	rotates clockwise
    4	rotates counter clockwise
    5	moves in circle
    6	scale up while rotating clockwise
    7	moves horizontally while rotating counter clockwise
    8	rotate clockwise and then counter clockwise
    9	random movements
    '''

    # moves horizontally
    def mov_hor(image, frames=10):#30):
        img_x, img_y = 10, 10 #28, 28
        video = np.zeros((frames, img_x, img_y), dtype=np.uint8)
        right_first = 0 #np.random.randint(2)

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