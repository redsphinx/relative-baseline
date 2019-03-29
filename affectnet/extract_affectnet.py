import os
import numpy as np
from relative_baseline.omg_emotion import project_paths as PP
from multiprocessing import Pool, Queue
import time
import shutil
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

global_queue = Queue()


def is_img_in_folder(name):
    path = os.path.join(PP.affectnet_jpgs, name)
    if not os.path.exists(path):
        global_queue.put(name)


def parallel_find(paths, processes=40):
    func = is_img_in_folder
    pool = Pool(processes=processes)
    pool.apply_async(is_img_in_folder)
    pool.map(func, paths)
    return pool


def find_missing_images():
    all_labels = np.genfromtxt(PP.affectnet_labels_train, skip_header=True, delimiter=',', dtype=str)
    names = all_labels[:, 0]
    names = [names[i].split('/')[-1] for i in range(len(names))]

    print('parallel finding images...')
    start = time.time()
    pool = parallel_find(names)
    print('duration: %f seconds' % (time.time() - start))

    if global_queue.empty():
        print('nothing is missing, everything is fine!')
    else:
        missing = os.path.join(PP.affect_net_base, 'missing_images.csv')
        print('uh oh...%d images are missing' % global_queue.qsize())
        start = time.time()
        with open(missing, 'a') as my_file:
            for i in range(global_queue.qsize()):
                my_file.write(global_queue.get()+'\n')
        print('duration: %f seconds' % (time.time() - start))

    pool.terminate()

    # val:
    # parallel finding images...
    # duration: 0.139436 seconds
    # nothing is missing, everything is fine!

    # train:
    # parallel finding images...
    # duration: 5.059090 seconds
    # uh oh...1 images are missing
    # duration: 0.000156 seconds



# find_missing_images()


def split_data():
    data_path = os.path.join(PP.affect_net_base, 'train_images')
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    all_jpgs = os.listdir(PP.affectnet_jpgs)

    all_labels = np.genfromtxt(PP.affectnet_labels_train, skip_header=True, delimiter=',', dtype=str)
    names = all_labels[:, 0]
    names = [names[i].split('/')[-1] for i in range(len(names))]

    for i in range(len(names)):
        if names[i] in all_jpgs:
            src = os.path.join(PP.affectnet_jpgs, names[i])
            dst = os.path.join(data_path, names[i])
            shutil.move(src, dst)
        else:
            print('%s not in all_images' % names[i])


# split_data()

def get_biggest_image():
    # which = 'train_images'
    which = 'val_images'
    data_path = os.path.join(PP.affectnet_base, which)

    list_data = os.listdir(data_path)

    largest_x = 0
    largest_y = 0

    smallest_x = 100000000
    smallest_y = 100000000

    mean_pixel = [0, 0, 0]

    for i in tqdm(range(len(list_data))):
        p = os.path.join(data_path, list_data[i])
        img = Image.open(p)
        img_x = img.width
        img_y = img.height

        if img_x > largest_x:
            largest_x = img_x
        if img_y > largest_y:
            largest_y = img_y

        if img_x < smallest_x:
            smallest_x = img_x
        if img_y < smallest_y:
            smallest_y = img_y

        img = np.array(img)
        mean_pixel += list(img.mean(axis=0).mean(axis=0))

    print('%s\n biggest x: %d, biggest y: %d, smallest x: %d, smallest y: %d' %
          (which, largest_x, largest_y, smallest_x, smallest_y))
    print('mean pixel: ', np.array(mean_pixel, dtype=int)//len(list_data))


# get_biggest_image()

def get_histogram_image_sizes():
    which = 'val_images'
    pkl_path = '/huge/gabras/AffectNet/misc/%s_sizes.pkl' % which

    if not os.path.exists(pkl_path):

        data_path = os.path.join(PP.affectnet_base, which)

        list_data = os.listdir(data_path)


        all_sizes = np.zeros(shape=(len(list_data), 2))

        for i in tqdm(range(len(list_data))):
            p = os.path.join(data_path, list_data[i])
            img = Image.open(p)
            all_sizes[i] = [img.width, img.height]

        np.save(pkl_path, all_sizes)
    else:
        all_sizes = np.load(pkl_path)

    # hist
    n, bins, patches = plt.hist(all_sizes[:, 0], 50, density=True, facecolor='g', alpha=0.75)

    plt.xlabel('width')
    plt.ylabel('Probability')
    plt.title('Histogram of OMG Emotion Image size')
    plt.grid(True)
    plt.savefig('/huge/gabras/AffectNet/misc/histo_size.jpg')


# get_histogram_image_sizes()

which = 'train_images'

def resize_this(name):
    src_folder = os.path.join(PP.affectnet_base, which)
    d = which + '_250'
    dest_folder = os.path.join(PP.affectnet_base, d)
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    img_path = os.path.join(src_folder, name)
    try:
        img = Image.open(img_path)
        img = img.resize((250, 250))
        # name = name.split('.')[0] + '.png'
        dest_path = os.path.join(dest_folder, name)
        img.save(dest_path)
    except OSError:
        print('cannot deal with: ', img_path)


def parallel_resize(paths, processes=50):
    func = resize_this
    pool = Pool(processes=processes)
    pool.apply_async(func)
    pool.map(func, paths)
    pool.terminate()


def resize_omg_emotion():
    # given histogram, resize to 250x250
    src_folder = os.path.join(PP.affectnet_base, which)
    paths = os.listdir(src_folder)

    done_paths = os.listdir('/huge/gabras/AffectNet/manually_annotated/train_images_250')

    todo = list(set(paths) - set(done_paths))

    paths = todo

    for i in todo:
        resize_this(i)

    # parallel_resize(paths)


# resize_omg_emotion()
