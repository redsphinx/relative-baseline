import os
import deepimpression2.constants as C
from relative_baseline import utils as U
from relative_baseline import data_loading as D
import numpy as np
import random
from . import project_paths as PP


# temporary for debugging
from .settings import ProjectVariable


# Training: {0: 262, 1: 96, 2: 54, 3: 503, 4: 682, 5: 339, 6: 19}
# Validation: {0: 51, 1: 34, 2: 17, 3: 156, 4: 141, 5: 75, 6: 7}
# Test: {0: 329, 1: 135, 2: 50, 3: 550, 4: 678, 5: 231, 6: 16}
# 0 - Anger
# 1 - Disgust
# 2 - Fear
# 3 - Happy
# 4 - Neutral
# 5 - Sad
# 6 - Surprise

def load_labels(project_variable, which):
    project_variable = ProjectVariable()
    path = os.path.join(PP.data_path, which, 'Annotations', 'annotations.csv')




    pass





def load_data(project_variable):
    project_variable = ProjectVariable()
    # get list of all utterance paths -> load labels first
    # for each path random sample a frame
    # make list of framepaths each epoch
    # for each step in epoch, load in memory from batchlist of paths



    pass

