from .settings import ProjectVariable
from torchvision.models import resnet18



def get_model(project_variable):
    project_variable = ProjectVariable()

    if project_variable.model_number == 0:
        model = resnet18(pretrained=True)
    else:
        model = None

    return model


def get_optimizer(project_variable):
    project_variable = ProjectVariable()


def get_device(project_variable):
    # should be a string
    project_variable = ProjectVariable()
