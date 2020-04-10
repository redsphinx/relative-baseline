from __future__ import print_function, division
import os
import torch
import numpy as np
# import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")

from relative_baseline.omg_emotion import project_paths as PP


class JesterDataset(Dataset):

    def __init__(self, which, project_variable):
        phases = ['train', 'val', 'test']
        assert which in phases
        idx_which = phases.index(which)

        label_path = os.path.join(PP.jester_location, 'labels_%s.npy' % which)
        self.labels = np.load(label_path)[:project_variable.data_points[idx_which]]
        self.root_dir = os.path.join(PP.jester_data_50_75)

    def __len__(self):
        return len(self.labels)

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


jester_dataset = JesterDataset()
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

