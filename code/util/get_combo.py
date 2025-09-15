import env, sys

def get_combo(task_id):
    fps, models, datasets = len(env.FEATURES), len(env.MODELS), len(env.DATASETS)
    fingerprint = env.FEATURES[task_id % fps]
    model_name = env.MODELS[(task_id // fps) % models]
    dataset_name = env.DATASETS[(task_id // (fps * models)) % datasets]
    return fingerprint, model_name, dataset_name

task_id = int(sys.argv[1])

print(get_combo(task_id))
