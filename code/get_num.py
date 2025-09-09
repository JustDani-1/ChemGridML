import sys, env

fingerprint = sys.argv[1]
model_name = sys.argv[2]
dataset_name = sys.argv[3]

fps, models, datasets = len(env.FEATURES), len(env.MODELS), len(env.DATASETS)
fp_idx, model_idx, dataset_idx = env.FEATURES.index(fingerprint), env.MODELS.index(model_name), env.DATASETS.index(dataset_name)
num = fp_idx + fps * model_idx + fps * models * dataset_idx

print(num)