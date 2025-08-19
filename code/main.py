
import study_manager,benchmark_manager, env, time

def main():
    manager = study_manager.StudyManager()
    results = benchmark_manager.BenchmarkManager()
    
    total_combinations = len(env.FINGERPRINTS) * len(env.MODELS) * len(env.DATASETS)
    current_combination = 0
    timestamp = int(time.time())
    
    print(f"Starting benchmark with {total_combinations} total combinations...")
    
    for fingerprint in env.FINGERPRINTS:
        for model_name in env.MODELS:
            for dataset_name in env.DATASETS:
                current_combination += 1
                print(f"\n[{current_combination}/{total_combinations}] Running: {fingerprint} + {model_name} + {dataset_name}")
                
                try:
                    study, score = manager.run_study(fingerprint, model_name, dataset_name, timestamp)
                    
                    results.add_result(
                        fingerprint, model_name, dataset_name,
                        score, study.best_params
                    )
                    
                    metric_name = list(score.keys())[0] if isinstance(score, dict) else "Score"
                    score_value = score[metric_name] if isinstance(score, dict) else score
                    print(f"Success: {metric_name} = {score_value:.4f}")
                    
                except Exception as e:
                    print(f"Failed: {fingerprint}_{model_name}_{dataset_name}: {e}")
                    results.add_failed_result(fingerprint, model_name, dataset_name, str(e))
                    continue
    
    print(f"\n{'='*60}")
    print("BENCHMARKING COMPLETE - GENERATING ANALYSIS")
    print("="*60)
    
    # Run comprehensive analysis
    final_stats = results.analyze_and_visualize()
    
    return results, final_stats

if __name__ == "__main__":
    results, stats = main()

        