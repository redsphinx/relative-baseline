from torch.optim import Adam
from torchvision.models import resnet18
import numpy as np
import os
from . import training as TR
from . import validation as VA
from . import test as TE
from .settings import ProjectVariable as PV

# --------------------------------------------------------------------------------------------------------

# easy debug  mode
DEBUG = True

if DEBUG:
    epochs = 2
    batches = 16
    train_total_steps = 10
    val_total_steps = 1
    test_total_steps = 1

else:
    epochs = 100
    batches = 32
    train_total_steps = 50
    val_total_steps = 10
    test_total_steps = 10
# --------------------------------------------------------------------------------------------------------

# loads a previously trained on OMG Emotion model
LOAD_MODEL = False

if LOAD_MODEL:
    pass
else:
    my_model = resnet18(pretrained=True)

# TODO: optimizer per layer?
my_optimizer = Adam()
# --------------------------------------------------------------------------------------------------------

exp_number = 0
mod_num = 0

for e in range(0, epochs):

    # ----------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------
    loss_train = TR.run_basic(my_model, my_optimizer, e, mod_num, exp_number)
    # ----------------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------------
    loss_val = VA.run(which='val', model=my_model, optimizer=my_optimizer, model_num=mod_num, experiment_number=exp_number,
                   epoch=e)

    print('epoch %d, train_loss: %f, val_loss: %f' % (e, float(np.mean(loss_train)), float(np.mean(loss_val))))
# --------------------------------------------------------------------------------------------------------