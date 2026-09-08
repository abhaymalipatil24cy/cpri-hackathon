# Master End-to-End Execution Pipeline for CPRI Project
import sys
import os
import time
import subprocess
from pathlib import Path

def run_full_pipeline():
    print("=== STARTING CPRI END-TO-END PIPELINE EXECUTION ===")
    t_start = time.time()
    
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    
    scripts = [
        'src/ingest.py',
        'src/clean_impute.py',
        'src/eda.py',
        'src/feature_engineering.py',
        'src/sensor_consistency.py',
        'src/validity_engine.py',
        'src/reference_regression.py',
        'src/uncertainty_pipeline.py',
        'src/final_submission_pipeline.py'
    ]
    
    for script in scripts:
        t0 = time.time()
        print(f"Executing stage: {script} ...")
        res = subprocess.run([sys.executable, script], capture_output=True, text=True, env=env)
        dt = time.time() - t0
        if res.returncode != 0:
            print(f"ERROR in {script} (exit code {res.returncode}):")
            print(res.stderr)
            sys.exit(res.returncode)
        else:
            print(f"Completed {script} in {dt:.2f}s")
            
    total_time = time.time() - t_start
    print(f"=== CPRI PIPELINE EXECUTION SUCCESSFUL (Total Runtime: {total_time:.2f}s) ===")

if __name__ == '__main__':
    run_full_pipeline()
