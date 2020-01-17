import os

server = 'lovelace'
# server = 'schmidhuber'

if server == 'schmidhuber':
    server_path = '/scratch/users/gabras/data'
elif server == 'lovelace':
    server_path = '/huge/gabras'
    # server_path = '/home/gabras/scratch'
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

# KTH
kth_location = '/huge/gabras/kth_actions/avi'
kth_png = '/huge/gabras/kth_actions/png'
kth_png_60_60 = '/huge/gabras/kth_actions/png6060'
kth_metadata = '/huge/gabras/kth_actions/metadata.txt'

# Marcel gestures
marcel_gestures_location = '/huge/gabras/marcel_gestures'

# DHG
dhg_location = '/huge/gabras/DHG'



# todo: IMPORTANT: DO NOT CHANGE EXISTING PATHS. SHIT WILL BE DELETED.
# todo: IF YOU REALLY REALLY WANT TO, MAKE SURE WRITER_PATH IS VALID -> main_file.py

