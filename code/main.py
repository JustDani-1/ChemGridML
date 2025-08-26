import env
import time
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Event
from study_manager import StudyManager

# Thread-safe printing and shutdown event
print_lock = Lock()
shutdown_event = Event()

def safe_print(message):
    with print_lock:
        print(message)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    shutdown_event.set()

def run_single_study(args):
    """
    Run a single study combination
    
    Args:
        args: tuple of (fingerprint, model_name, dataset_name)
    
    Returns:
        success: bool
    """
    fingerprint, model_name, dataset_name = args
    
    # Check if shutdown was requested before starting
    if shutdown_event.is_set():
        return False
    
    try:
        # Use shared database for all studies
        manager = StudyManager(f"./studies/{env.TIMESTAMP}/studies.db", f"./studies/{env.TIMESTAMP}/predictions.db")
        
        manager.run_complete_study(fingerprint, model_name, dataset_name)
        
        return True
        
    except KeyboardInterrupt:
        return False
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
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Generate all combinations
    combinations = []
    for fingerprint in env.FINGERPRINTS:
        for model_name in env.MODELS:
            for dataset_name in env.DATASETS:
                combinations.append((fingerprint, model_name, dataset_name))
    
    num_threads = get_num_threads()
    total_combinations = len(env.FINGERPRINTS) * len(env.MODELS) * len(env.DATASETS)
    
    # Use ThreadPoolExecutor to parallelize the work
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all jobs
            future_to_combination = {
                executor.submit(run_single_study, combo): combo 
                for combo in combinations
            }
            
            # Process completed jobs as they finish
            successful_combinations = 0
            
            for future in as_completed(future_to_combination):
                if shutdown_event.is_set():
                    # Cancel remaining futures
                    for f in future_to_combination:
                        if not f.done():
                            f.cancel()
                    break
                
                success = future.result()
                    
                if success:
                    successful_combinations += 1
                    safe_print(f"Successful: {successful_combinations}/{total_combinations}")
    
    except KeyboardInterrupt:
        sys.exit(1)
    
    # Final database stats and summary
    print(f"\n" + "="*80)
    print(f"EXECUTION SUMMARY")
    print(f"="*80)
    print(f"Total combinations: {total_combinations}")
    print(f"Successfully completed: {successful_combinations}")
    print(f"Failed/Cancelled: {total_combinations-successful_combinations}")
    print(f"Threads used: {num_threads}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")