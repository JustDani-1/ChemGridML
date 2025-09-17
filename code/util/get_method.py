import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from experiments import ExperimentRegistry

def get_method(experiment_name, task_id):
    experiment_registry = ExperimentRegistry()
    experiment = experiment_registry.get_experiment(experiment_name)
    method = experiment.get_method(task_id)
    return method.feature, method.model, method.dataset


if __name__ == '__main__':
    experiment_name = sys.argv[1]
    task_id = int(sys.argv[2])
    print(get_method(experiment_name, task_id))

