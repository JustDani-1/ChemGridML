# main.py
import time, sys
from study_manager import StudyManager
from experiments import ExperimentRegistry

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python main.py <master_job_id> <experiment_name> <task_id>")
        sys.exit(1)
    
    # Parse arguments
    master_job_id = sys.argv[1]
    experiment_name = sys.argv[2]
    task_id = int(sys.argv[3])

    # Get experiment and method
    experiment_registry = ExperimentRegistry()
    experiment = experiment_registry.get_experiment(experiment_name)
    method = experiment.get_method(task_id)
    
    # Run study
    start_time = time.time()

    # Create output paths using master job ID and experiment name
    studies_path = f"./studies/{master_job_id}/{experiment_name}/studies/"
    predictions_path = f"./studies/{master_job_id}/{experiment_name}/predictions.db"
    
    manager = StudyManager(method, studies_path, predictions_path)
    manager.run_nested_cv()

    end_time = time.time()

    print(f"Total execution time: {end_time - start_time:.2f} seconds")