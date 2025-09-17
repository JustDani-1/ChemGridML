import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from experiments import ExperimentRegistry


experiment_name = sys.argv[1]
task_id = int(sys.argv[2])

def get_method(experiment_name, task_id):
    experiment_registry = ExperimentRegistry()
    experiment = experiment_registry.get_experiment(experiment_name)
    method = experiment.methods[task_id-1]
    return method.feature, method.model, method.dataset

print(get_method(experiment_name, task_id))

