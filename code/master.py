# master.py
import sys
import os
import subprocess
from typing import List
from experiments import Experiment, ExperimentRegistry

def create_job_script(master_job_id: str, experiment: Experiment) -> str:
    """Create a job submission script for an experiment"""

    gpu_line = "#$ -l gpu=1\n" if experiment.resources.gpu else ""
    
    script_content = f"""#!/bin/bash -l

#$ -l h_rt={experiment.resources.wall_time}
#$ -l mem={experiment.resources.memory}G
#$ -pe smp {experiment.resources.cores}
{gpu_line}#$ -t 1-{len(experiment.methods)}

#$ -N {experiment.name}

#$ -j y
#$ -o ./output/{master_job_id}/{experiment.name}/
#$ -wd $HOME/Scratch/UCL_internship

conda activate internship

python ./code/main.py {master_job_id} {experiment.name} $SGE_TASK_ID

"""
    
    return script_content

def submit_experiment(master_job_id: str, experiment: Experiment):
    """Create and submit a job script for an experiment"""
    
    os.makedirs(f"./output/{master_job_id}/{experiment.name}", exist_ok=True)

    # Create job script content
    script_content = create_job_script(master_job_id, experiment)
    
    # Write script to file
    script_filename = f"{experiment.name}.sh"
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
        print(f"  - Methods: {len(experiment.methods)}")
        print(f"  - Resources: {experiment.resources.memory}G RAM, {experiment.resources.cores} cores, {experiment.resources.wall_time} time, GPU: {experiment.resources.gpu}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit experiment '{experiment.name}': {e}")
        print(f"Error output: {e.stderr}")
    finally:
        # Clean up script file
        if os.path.exists(script_filename):
            os.remove(script_filename)

def main():
    if len(sys.argv) < 3:
        print("Usage: python master.py <master_job_id> <experiment1> [experiment2] ...")
        print("\nAvailable experiments:")
        registry = ExperimentRegistry()
        for exp_name in registry.list_experiments():
            exp = registry.get_experiment(exp_name)
            print(f"  {exp_name}: {len(exp.methods)} methods, "
                  f"{exp.resources.memory} RAM, {exp.resources.cores} cores")
        sys.exit(1)
    
    master_job_id = sys.argv[1]
    experiments = sys.argv[2:]
    
    # Initialize experiment registry
    experiment_registry = ExperimentRegistry()
    
    submitted_jobs = []
    
    # Submit each experiment
    for exp in experiments:
        try:
            experiment = experiment_registry.get_experiment(exp)
            job_id = submit_experiment(master_job_id, experiment)
            if job_id:
                submitted_jobs.append((exp, job_id))
        except ValueError as e:
            continue
    
    print("-" * 60)
    print(f"Summary: Successfully submitted {len(submitted_jobs)}/{len(experiments)} experiments")

if __name__ == "__main__":
    main()