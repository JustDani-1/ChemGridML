# main.py
import env, time, sys
from study_manager import StudyManager
from methods import MethodRegistry

if __name__ == '__main__':
    # Array job calculations
    env.TIMESTAMP = sys.argv[1]
    task_id = int(sys.argv[2])

    # Initialize method registry
    method_registry = MethodRegistry()
    
    # Calculate indices for array job
    total_methods = method_registry.total_methods()
    total_datasets = len(env.DATASETS)
    
    # Get method and dataset from task_id
    method_idx = task_id % total_methods
    dataset_idx = (task_id // total_methods) % total_datasets
    
    method = method_registry.get_method(method_idx)
    dataset_name = env.DATASETS[dataset_idx]

    # Run study
    start_time = time.time()

    manager = StudyManager(f"./studies/{env.TIMESTAMP}/studies/", f"./studies/{env.TIMESTAMP}/predictions.db") 
    manager.run_nested_cv(method, dataset_name)

    end_time = time.time()

    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")