from experiments import ExperimentRegistry
import sys

experiment_name = sys.argv[1]
task_id = int(sys.argv[2]) - 1

experiment_registry = ExperimentRegistry()
experiment = experiment_registry.get_experiment(experiment_name)

print(experiment.methods[task_id])

