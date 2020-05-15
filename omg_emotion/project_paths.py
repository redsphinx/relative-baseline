import os


server = 'lovelace'

if server == 'schmidhuber':
    server_path = '/scratch/users/gabras/data'
elif server == 'lovelace':
    server_path = '/huge/gabras'
    # server_path = '/home/gabras/scratch'
elif server == 'godel':
    server_path = '/fast/gabras'
else:
    print('server unknown')
    server_path = None

# OMG_emotion
saving_data = os.path.join(server_path, 'omg_emotion/saving_data')
data_path = os.path.join(server_path, 'omg_emotion')
omg_emotion_jpg = 'jpg_full_body_background_1280_720'
omg_emotion_jpg_face = 'jpg_face_cropped_96_96'
models = os.path.join(saving_data, 'models')
writer_path = os.path.join(saving_data, 'tensorboardX')

# for dlib cropping face
predictor = '/home/gabras/deployed/relative_baseline/omg_emotion/shape_predictor_68_face_landmarks.dat'

# affect net
affectnet_base = os.path.join(server_path, 'AffectNet/manually_annotated')
affectnet_jpgs = os.path.join(affectnet_base, 'all_images')
affectnet_labels_train = os.path.join(affectnet_base, 'training.csv')
affectnet_labels_val = os.path.join(affectnet_base, 'validation.csv')

# MNIST stuff on lovelace
mnist_location = '/home/gabras/deployed/mnist'
dummy_location = '/scratch/users/gabras/data/convttn3d_project/dummy_data'
moving_mnist_location = '/scratch/users/gabras/data/convttn3d_project/moving_mnist'
moving_mnist_png = '/scratch/users/gabras/data/convttn3d_project/moving_mnist/png'
mov_mnist_mean_std = '/scratch/users/gabras/data/convttn3d_project/moving_mnist/mean_std.npy'

# KTH
kth_location = '/huge/gabras/kth_actions/avi'
kth_png = '/huge/gabras/kth_actions/png'
kth_png_60_60 = '/huge/gabras/kth_actions/png6060'
kth_metadata = '/huge/gabras/kth_actions/metadata.txt'

# Marcel gestures
marcel_gestures_location = '/huge/gabras/marcel_gestures'

# DHG
dhg_location = '/huge/gabras/DHG'
dhg_hand_only_28_28 = '/huge/gabras/DHG_hand_only_28_28'
dhg_hand_only_28_28_50_frames = '/huge/gabras/DHG_hand_only_28_28_50_frames'
dhg_mean_std = '/huge/gabras/DHG/mean_std.npy'

# Jester
# symbolic link on godel:   /scratch/users -> /fast
jester_location = '/scratch/users/gabras/jester'
jester_data = '/scratch/users/gabras/jester/data'
jester_data_50_75 = '/scratch/users/gabras/jester/data_50_75'
jester_data_50_75_avi = '/scratch/users/gabras/jester/data_50_75_avi'
jester_data_50_75_avi_clean = '/scratch/users/gabras/jester/data_50_75_avi_clean'
jester_data_224_336 = '/scratch/users/gabras/jester/data_224_336'
jester_data_224_336_avi = '/scratch/users/gabras/jester/data_224_336_avi'
fast_jester_data_224_336_avi = '/fast/gabras/jester/data_224_336_avi'
fast_jester_data_150_224 = '/fast/gabras/jester/data_150_224'
fast_jester_data_150_224_avi = '/fast/gabras/jester/data_150_224_avi'
jester_frames = '/scratch/users/gabras/jester/frames.txt'
jester_zero = '/scratch/users/gabras/jester/zero.txt'

# UCF101
ucf101_root = '/fast/gabras/ucf101/og_data'
ucf101_annotations = '/fast/gabras/ucf101/og_labels'

# visualization saving locations
pacman_location = '/home/gabras/deployed/relative_baseline/omg_emotion/images/pacman.jpg'
erhan2009 = '/home/gabras/deployed/relative_baseline/omg_emotion/images/erhan2009'
zeiler2014 = '/home/gabras/deployed/relative_baseline/omg_emotion/images/zeiler2014'

xai_visualizations = '/huge/gabras/omg_emotion/saving_data/xai'
our_method = '/huge/gabras/omg_emotion/saving_data/xai/our_method'

# NAS saving location
nas_location = '/huge/gabras/omg_emotion/saving_data/nas'

# todo: IMPORTANT: DO NOT CHANGE EXISTING PATHS. SHIT WILL BE DELETED.
# todo: IF YOU REALLY REALLY WANT TO, MAKE SURE WRITER_PATH IS VALID -> main_file.py

