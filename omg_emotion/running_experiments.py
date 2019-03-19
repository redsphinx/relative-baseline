from .settings import ProjectVariable
from . import main


project_variable = ProjectVariable()

def test():
    project_variable.model_number = 0
    project_variable.experiment_number = 0
    project_variable.debug_mode = True
    main.run(project_variable)

