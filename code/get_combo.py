import env, sys


task_id = int(sys.argv[1])

fps, models, datasets = len(env.FINGERPRINTS), len(env.MODELS), len(env.DATASETS)
fingerprint = env.FINGERPRINTS[task_id % fps]
model_name = env.MODELS[(task_id // fps) % models]
dataset_name = env.DATASETS[(task_id // (fps * models)) % datasets]

print(fingerprint, model_name, dataset_name)
