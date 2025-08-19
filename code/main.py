
import study_manager, env

def main():
    manager = study_manager.StudyManager()
    results = study_manager.BenchmarkResults()
    
    for fingerprint in env.FINGERPRINTS:
        for model_name in env.MODELS:
            for dataset_name in env.DATASETS:
                print(f"\nRunning: {fingerprint} + {model_name} + {dataset_name}")
                
                try:
                    study = manager.run_study(fingerprint, model_name, dataset_name)
                    
                    # Store results
                    results.add_result(
                        fingerprint, model_name, dataset_name,
                        study.best_value, study.best_params
                    )
                    
                except Exception as e:
                    print(f"Failed: {fingerprint}_{model_name}_{dataset_name}: {e}")
                    continue
    
    # Generate summary
    #results.save_summary()
    #results.plot_results()


main()


        