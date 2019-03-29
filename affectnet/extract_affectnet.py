import os
import numpy as np
from relative_baseline.omg_emotion import project_paths as PP
from multiprocessing import Pool, Queue
import time
import shutil

# what are the 6998 too many files?

# are all the files in the labels present?
'''
get all labels
for each label
does image exist
if yes: do nothing
if no: remember
'''

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
