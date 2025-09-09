import env, time, sys
from study_manager import StudyManager

if __name__ == '__main__':
    # array job calculations
    env.TIMESTAMP = sys.argv[1]
    task_id = int(sys.argv[2])

    fps, models, datasets = len(env.FEATURES), len(env.MODELS), len(env.DATASETS)
    fingerprint = env.FEATURES[task_id % fps]
    model_name = env.MODELS[(task_id // fps) % models]
    dataset_name = env.DATASETS[(task_id // (fps * models)) % datasets]

    # run study
    start_time = time.time()

    manager = StudyManager(f"./studies/{env.TIMESTAMP}/studies/", f"./studies/{env.TIMESTAMP}/predictions.db") 
    manager.run_nested_cv(fingerprint, model_name, dataset_name)

    end_time = time.time()

    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
