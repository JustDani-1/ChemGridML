import env
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from study_manager import StudyManager

# Thread-safe printing
print_lock = Lock()

def safe_print(message):
    with print_lock:
        print(message)

def run_single_study(args):
    """
    Run a single study combination
    
    Args:
        args: tuple of (fingerprint, model_name, dataset_name)
    
    Returns:
        success: bool
    """
    fingerprint, model_name, dataset_name = args
    
    try:
        # Use shared database for all studies
        manager = StudyManager(f"./studies/{env.TIMESTAMP}/studies.db", f"./studies/{env.TIMESTAMP}/predictions.db")
        
        manager.run_complete_study(fingerprint, model_name, dataset_name)
        
        return True
        
    except Exception as e:
        safe_print(f"Failed on {fingerprint}_{model_name}_{dataset_name}: {e}")
        return False
    
def get_num_threads():
    """
    Get the number of threads to use based on cluster allocation
    """
    # First try to get from OMP_NUM_THREADS (set by cluster scheduler)
    omp_threads = os.environ.get('OMP_NUM_THREADS')
    if omp_threads:
        try:
            return int(omp_threads)
        except ValueError:
            pass
    
    return os.cpu_count() or 1

def main():
    
    # Generate all combinations
    combinations = []
    for fingerprint in env.FINGERPRINTS:
        for model_name in env.MODELS:
            for dataset_name in env.DATASETS:
                combinations.append((fingerprint, model_name, dataset_name))
    
    num_threads = get_num_threads()
    total_combinations = len(env.FINGERPRINTS) * len(env.MODELS) * len(env.DATASETS)
    
    # Use ThreadPoolExecutor to parallelize the work
    with ThreadPoolExecutor(max_workers=num_threads) as executor:

        future_to_combination = {
            executor.submit(run_single_study, combo): combo 
            for combo in combinations
        }
        
        # Process completed jobs as they finish
        successful_combinations = 0
        for future in as_completed(future_to_combination):
            success = future.result()
            
            if success:
                successful_combinations += 1
    
    # Final database stats and summary
    print(f"\n" + "="*80)
    print(f"EXECUTION SUMMARY")
    print(f"="*80)
    print(f"Total combinations: {total_combinations}")
    print(f"Successfully completed: {successful_combinations}")
    print(f"Failed: {total_combinations-successful_combinations}")
    print(f"Threads used: {num_threads}")

def generate_summary_report(results):
    """Generate a summary report of all completed studies"""
    print(f"\n" + "="*80)
    print(f"RESULTS SUMMARY")
    print(f"="*80)
    
    # Try to create a simple summary table
    try:
        manager = StudyManager(f"./studies/predictions.db")
        summary_df = manager.db.get_test_scores_summary()
        
        if not summary_df.empty:
            print(f"\nTest Score Summary (across all seeds):")
            print(f"{'Combination':<40} {'Metric':<12} {'Mean±Std':<15} {'Min':<8} {'Max':<8}")
            print("-" * 85)
            
            for _, row in summary_df.iterrows():
                combo = f"{row['fingerprint']}_{row['model_name']}_{row['dataset_name']}"
                metric = row['metric_name']
                mean_std = f"{row['mean_score']:.4f}±{row['std_score']:.4f}"
                min_val = f"{row['min_score']:.4f}"
                max_val = f"{row['max_score']:.4f}"
                
                print(f"{combo:<40} {metric:<12} {mean_std:<15} {min_val:<8} {max_val:<8}")
        
        print(f"\nDetailed results and predictions are stored in the database:")
        print(f"  - Database location: ./studies/predictions.db")
        print(f"  - Use manager.db.get_predictions_dataframe() to access raw predictions")
        print(f"  - Use manager.db.get_test_scores_summary() for metric summaries")
        
    except Exception as e:
        print(f"Could not generate summary report: {e}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")