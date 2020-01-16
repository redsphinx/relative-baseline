import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import data_loading as D


project_variable = ProjectVariable(debug_mode=True)

project_variable.dataset = 'omg_emotion'
project_variable.data_points = [10 * 7, 0, 0]
project_variable.train = True
project_variable.test = False
project_variable.val = False
project_variable.label_type = 'categories'

all_data = D.load_data(project_variable, seed=42)

data = all_data[1][0]
labels = all_data[2][0]

print('gabi')


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample