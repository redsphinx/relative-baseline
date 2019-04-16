import numpy as np
import os
import wget
import subprocess


def cat(fn, read_this_many, left_of):

    total = 964873

    if left_of + read_this_many < total - left_of:
        # cat, head, tail
        command = "cat %s | head -n%d | tail -n%d" % (fn, left_of+read_this_many, read_this_many)
    else:
        # cat, tail, head
        command = "cat %s | tail -n%d | head -n%d" % (fn, total-left_of, read_this_many)

    arr = []
    l = subprocess.check_output(command, shell=True).decode('utf-8')
    l = l.split('\n')
    l = l[0:-1]
    for i in range(len(l)):
        p0 = l[i].split(',')[0]
        p1 = l[i].split(',')[1]
        arr.append([p0, p1])

    return arr


def which_done():
    dest_path = '/huge/gabras/IBM_DiF_v1/images'

    # assume they download in order
    num_done = len(os.listdir(dest_path))

    not_download = np.genfromtxt('/huge/gabras/IBM_DiF_v1/not_downloaded.txt', delimiter=',', dtype=str)
    num_done += len(not_download)

    return num_done, dest_path


def grab_em_by_the_links():
    path = '/huge/gabras/IBM_DiF_v1/DiF_v1b.csv'

    num_done, dest_path = which_done()
    links = cat(path, 10000, num_done)

    could_not_download = '/huge/gabras/IBM_DiF_v1/not_downloaded.txt'


    for i in range(len(links)):
        ext = links[i][1].split('.')[-1]
        save_name = os.path.join(dest_path, links[i][0] + '.' + ext)
        try:
            img = wget.download(links[i][1], save_name)
            print('download %s successful' % img)
        except:
            line = '%s,%s\n' % (links[i][0], links[i][1])
            with open(could_not_download, 'a') as my_file:
                my_file.write(line)

            print('could not download %s' % links[i][0])


grab_em_by_the_links()
