from relative_baseline.omg_emotion.settings import ProjectVariable
from relative_baseline.omg_emotion import main_file


project_variable = ProjectVariable(debug_mode=False)


def pilot():
    project_variable.device = 0
    project_variable.model_number = 0
    project_variable.experiment_number = 0

    project_variable.batch_size = 16

    project_variable.save_model = False
    project_variable.save_data = False

    main_file.run(project_variable)


pilot()
