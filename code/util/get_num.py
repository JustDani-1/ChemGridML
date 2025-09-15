import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from experiments import ExperimentRegistry

experiment_name = sys.argv[1]
experiment_registry = ExperimentRegistry()
experiment = experiment_registry.get_experiment(experiment_name)
fingerprint = sys.argv[2]
model = sys.argv[3]
dataset = sys.argv[4]

for i, method in enumerate(experiment.methods):
    if str(method) == f"{fingerprint}_{model}_{dataset}":
        print(f"Task ID: {i+1}")
        break
else:
    print("Not found")

