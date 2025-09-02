import os
import re
import sys
from pathlib import Path
from get_combo import get_combo

def extract_runtime_from_file(file_path):
    """
    Extract runtime from output file.
    Returns execution time in seconds, or -1 if file is empty or no time found.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Check if file is empty
        if not content:
            return -1
            
        # Look for pattern: "Total execution time: X seconds"
        pattern = r'Total execution time:\s*(\d+(?:\.\d+)?)\s*seconds'
        match = re.search(pattern, content)
        
        if match:
            return float(match.group(1))
        else:
            return -1
            
    except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
        print(f"Error reading file {file_path}: {e}")
        return -1

def extract_task_id_from_filename(filename, directory_name):
    """
    Extract task ID from filename.
    Assumes format like: AllArray.o369722.135 where 369722 is directory name
    and 135 is the task ID.
    """
    
    # Try to extract task ID from filename
    # Pattern: AllArray.[eo]<directory_name>.<task_id>
    pattern = f"AllArray\\.[eo]{re.escape(directory_name)}\\.(.+)"
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)
    else:
        # Fallback: try to extract any number at the end
        pattern = r"\.(\d+)$"
        match = re.search(pattern, filename)
        return match.group(1) if match else filename

def analyze_runtimes(base_directory, job_id):
    """
    Analyze all output files in the given directory and return sorted runtime data.
    """
    base_path = Path(base_directory)
    
    if not base_path.exists():
        print(f"Error: Directory {base_directory} does not exist!")
        return []
    
    runtime_data = []
    
    for file_path in base_path.iterdir():
        if file_path.is_file():
            filename = file_path.name
            
            # Skip error files 
            if filename.count(f".o{job_id}."):
    
                runtime = extract_runtime_from_file(file_path)
                task_id = extract_task_id_from_filename(filename, job_id)
                
                runtime_data.append({
                    'task_id': task_id,
                    'runtime': runtime
                })
            
    
    return runtime_data

def main():
    """Main function"""
    job_id = sys.argv[1]
    base_directory = f"./output/{job_id}"
    
    print(f"Analyzing runtime data in: {base_directory}")
    print("=" * 50)
    
    # Analyze runtimes
    runtime_data = analyze_runtimes(base_directory, job_id)
    
    if not runtime_data:
        print("No runtime data found!")
        return
    
    # Sort by runtime (descending), with -1 values at the end
    runtime_data.sort(key=lambda x: (x['runtime'] == -1, -x['runtime']))
    
    # Display results
    print(f"\nFound {len(runtime_data)} tasks total")
    print("\nTop 10 longest running tasks:")
    print("-" * 90)
    print(f"{'Rank':<5} {'Task ID':<15} {'FP':<10} {'Model':<10} {'Dataset':<30} {'Runtime (s)':<12}")
    print("-" * 90)
    
    for i, data in enumerate(runtime_data[:10], 1):
        runtime_str = f"{data['runtime']:.2f}" if data['runtime'] != -1 else "EMPTY/ERROR"
        fp, model, dataset = get_combo(int(data['task_id']))
        print(f"{i:<5} {data['task_id']:<15} {fp:<10} {model:<10} {dataset:<30}{runtime_str:<12}")
    
    # Show some statistics
    valid_runtimes = [d['runtime'] for d in runtime_data if d['runtime'] != -1]
    empty_count = len([d for d in runtime_data if d['runtime'] == -1])
    
    if valid_runtimes:
        print(f"\nStatistics:")
        print(f"- Tasks with valid runtime: {len(valid_runtimes)}")
        print(f"- Empty/error files: {empty_count}")
        print(f"- Average runtime: {sum(valid_runtimes)/len(valid_runtimes):.2f} seconds")
        print(f"- Max runtime: {max(valid_runtimes):.2f} seconds")
        print(f"- Min runtime: {min(valid_runtimes):.2f} seconds")

if __name__ == "__main__":
    main()