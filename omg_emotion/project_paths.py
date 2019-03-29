import os

server = 'lovelace'
# server = 'schmidhuber'

if server == 'schmidhuber':
    server_path = '/scratch/users/gabras/data'
elif server == 'lovelace':
    server_path = '/home/gabras/scratch'
else:
    print('server unknown')
    server_path = None

saving_data = os.path.join(server_path, 'omg_emotion/saving_data')
data_path = os.path.join(server_path, 'omg_emotion')

omg_emotion_jpg = 'jpg_full_body_background_1280_720'

models = os.path.join(saving_data, 'models')

writer_path = os.path.join(saving_data, 'tensorboardX')

affect_net_base = '/home/gabras/scratch/AffectNet/manually_annotated'
affectnet_jpgs = '/home/gabras/scratch/AffectNet/manually_annotated/all_images'
affectnet_labels_train = '/home/gabras/scratch/AffectNet/manually_annotated/training.csv'
affectnet_labels_val = '/home/gabras/scratch/AffectNet/manually_annotated/validation.csv'
