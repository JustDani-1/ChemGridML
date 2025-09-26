# master.py
import sys
import os
import subprocess
from typing import List
from experiments import Experiment, ExperimentRegistry, Group

def create_job_script(master_job_id: str, experiment: Experiment, group: Group) -> str:
    """Create a job submission script for an experiment"""

    gpu_line = "#$ -l gpu=1\n" if group.resources.gpu else ""
    
    script_content = f"""#!/bin/bash -l

#$ -l h_rt={group.resources.wall_time}
#$ -l mem={group.resources.memory}G
#$ -pe smp {group.resources.cores}
{gpu_line}#$ -t {group.lower}-{group.upper}

#$ -N {group.name}

#$ -j y
#$ -o $HOME/Scratch/UCL_internship/output/{master_job_id}/{experiment.name}/
#$ -wd $HOME/Scratch/UCL_internship

conda activate internship

python ./code/main.py {master_job_id} {experiment.name} $SGE_TASK_ID

"""
    
    return script_content

def submit_experiment(master_job_id: str, experiment: Experiment):
    """Create and submit a job script for an experiment"""
    
    os.makedirs(f"./output/{master_job_id}/{experiment.name}", exist_ok=True)

    for group in experiment.groups:

        # Create job script content
        script_content = create_job_script(master_job_id, experiment, group)
        
        # Write script to file
        script_filename = f"{group.name}.sh"
        with open(script_filename, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_filename, 0o755)
        
        # Submit job
        try:
            result = subprocess.run(['qsub', script_filename], 
                                capture_output=True, text=True, check=True)
            job_id = result.stdout.strip()
            print(f"Submitted experiment '{experiment.name}': {job_id}")
            print(f"  - Group: {group.name}")
            print(f"  - Resources: {group.resources.memory}G RAM, {group.resources.cores} cores, {group.resources.wall_time} time, GPU: {group.resources.gpu}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit group '{group.name}': {e}")
            print(f"Error output: {e.stderr}")
        finally:
            # Clean up script file
            if os.path.exists(script_filename):
                os.remove(script_filename)


def run_experiment(master_job_id: str, experiment: Experiment):
    """Run an experiment locally by calling main.py for each method"""
    
    os.makedirs(f"./output/{master_job_id}/{experiment.name}", exist_ok=True)
    
    for task_id in range(1, experiment.total_methods() + 1):
        print(f"\nRunning method {task_id}/{experiment.total_methods()}")
        
        try:
            # Call main.py with the same arguments as would be used in the cluster
            # Let output stream through directly
            result = subprocess.run([
                'python', './code/main.py', 
                master_job_id, 
                experiment.name, 
                str(task_id)
            ], check=True)
            
            print(f"Method {task_id} completed successfully")
                
        except subprocess.CalledProcessError as e:
            print(f"Method {task_id} failed with exit code {e.returncode}")
            # Continue with next method even if one fails
            continue
    
    print(f"\nCompleted experiment '{experiment.name}'")



                
def print_experiments():
    registry = ExperimentRegistry()
    for exp_name in registry.list_experiments():
        exp = registry.get_experiment(exp_name)
        print(f"  {exp_name}: {exp.total_methods()} methods")

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == 'CLUSTER':
            if len(sys.argv) < 4:
                print("Usage: python master.py CLUSTER <master_job_id> <experiment1> [experiment2] ...")
                print("\nAvailable experiments:")
                print_experiments()
                sys.exit(1)
            
            master_job_id = sys.argv[2]
            experiments = sys.argv[3:]
            
            experiment_registry = ExperimentRegistry()
            
            # Submit each experiment
            for exp in experiments:
                experiment = experiment_registry.get_experiment(exp)
                submit_experiment(master_job_id, experiment)
        else:
            if len(sys.argv) < 3:
                print("Usage: python master.py <master_job_id> <experiment1> [experiment2] ...")
                print("\nAvailable experiments:")
                print_experiments()
                sys.exit(1)
            
            master_job_id = sys.argv[1]
            experiments = sys.argv[2:]
            
            experiment_registry = ExperimentRegistry()
            
            # Submit each experiment
            for exp in experiments:
                experiment = experiment_registry.get_experiment(exp)
                run_experiment(master_job_id, experiment)     
    else:
        print("Usage: python master.py <master_job_id> <experiment1> [experiment2] ...")
        print("\nAvailable experiments:")
        print_experiments()
        sys.exit(1)

    

if __name__ == "__main__":
    main()