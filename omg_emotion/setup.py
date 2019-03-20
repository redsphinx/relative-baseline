# from relative_baseline.omg_emotion.settings import ProjectVariable
from torchvision.models import resnet18
from torch.optim.adam import Adam


def get_model(project_variable):
    if project_variable.model_number == 0:
        model = resnet18(pretrained=True)
    else:
        model = None

    return model


def get_optimizer(project_variable, model):
    # TODO: multiple optimizers
    if project_variable.optimizer[0] == 'adam':
        optimizer = Adam(params=model.parameters(), lr=project_variable.learning_rate[0])
    else:
        optimizer = None

    return optimizer


def get_device(project_variable):
    if project_variable.device is None:
        device = 'cpu'
    elif type(project_variable.device) is int:
        device = 'cuda:%d' % project_variable.device
    else:
        device = None

    return device
