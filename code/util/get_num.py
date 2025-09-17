import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from experiments import ExperimentRegistry

experiment_name = sys.argv[1]
experiment_registry = ExperimentRegistry()
experiment = experiment_registry.get_experiment(experiment_name)
feature = sys.argv[2]
model = sys.argv[3]
dataset = sys.argv[4]

for group in experiment.groups:
    for i, method in enumerate(group.methods):
        if str(method) == f"{feature}_{model}_{dataset}":
            print(f"Task ID: {group.lower+i}")
            break

