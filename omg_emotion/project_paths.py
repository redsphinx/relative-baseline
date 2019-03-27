import os

# server = 'lovalace'
server = 'schmidhuber'

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