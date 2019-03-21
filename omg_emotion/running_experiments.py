from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


project_variable = ProjectVariable()


def pilot():
    project_variable.device = 0
    project_variable.model_number = 0
    project_variable.experiment_number = 0
    project_variable.debug_mode = True
    main_file.run(project_variable)


pilot()
