from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import subprocess
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'data-engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 9, 1),
}

def run_model_generation():
    """Run the intelligent model generator"""
    project_dir = os.path.expanduser('~/si-dbt-test/my_dbt_project')
    script_path = os.path.join(project_dir, 'generate_intelligent_models.py')
    
    # Activate virtual environment and run script
    venv_python = os.path.expanduser('~/si-dbt-test/env/bin/python')
    
    result = subprocess.run([
        venv_python, script_path
    ], capture_output=True, text=True, cwd=project_dir)
    
    if result.returncode != 0:
        logger.error(f"Model generation failed: {result.stderr}")
        raise Exception(f"Model generation failed: {result.stderr}")
    
    logger.info(f"Model generation output: {result.stdout}")
    return result.stdout

def validate_environment():
    """Validate that all requirements are met"""
    # Check for API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set!")
    
    # Check for CSV files
    data_dir = os.path.expanduser('~/si-dbt-test/my_dbt_project/data')
    if not os.path.exists(data_dir):
        raise ValueError("Data directory not found!")
        
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError("No CSV files found in data directory!")
    
    logger.info(f"Environment validated. Found {len(csv_files)} CSV files.")

def check_dbt_installation():
    """Check if dbt is properly installed"""
    try:
        result = subprocess.run(['dbt', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"dbt version: {result.stdout}")
        else:
            raise Exception("dbt command not found")
    except Exception as e:
        raise Exception(f"dbt installation check failed: {e}")

dag = DAG(
    'dbt_auto_csv_pipeline',
    default_args=default_args,
    description='AI-assisted dbt pipeline with Gemini - Python 3.11 Optimized',
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=['dbt', 'ai', 'gemini', 'python311'],
    max_active_runs=1,
)

# Start task
start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
)

# Task 1: Validate environment
validate_task = PythonOperator(
    task_id='validate_environment',
    python_callable=validate_environment,
    dag=dag,
)

# Task 2: Check dbt installation
dbt_check_task = PythonOperator(
    task_id='check_dbt_installation',
    python_callable=check_dbt_installation,
    dag=dag,
)

# Task 3: Generate models using AI
generate_task = PythonOperator(
    task_id='generate_intelligent_models',
    python_callable=run_model_generation,
    dag=dag,
)

# Task 4: Run dbt seed
seed_task = BashOperator(
    task_id='dbt_seed',
    bash_command='cd ~/si-dbt-test/my_dbt_project && source ../env/bin/activate && dbt seed --show',
    dag=dag,
)

# Task 5: Run dbt staging models
run_staging_task = BashOperator(
    task_id='dbt_run_staging',
    bash_command='cd ~/si-dbt-test/my_dbt_project && source ../env/bin/activate && dbt run --select models/staging/',
    dag=dag,
)

# Task 6: Run dbt generated models
run_generated_task = BashOperator(
    task_id='dbt_run_generated',
    bash_command='cd ~/si-dbt-test/my_dbt_project && source ../env/bin/activate && dbt run --select models/generated/',
    dag=dag,
)

# Task 7: Run dbt tests
test_task = BashOperator(
    task_id='dbt_test',
    bash_command='cd ~/si-dbt-test/my_dbt_project && source ../env/bin/activate && dbt test --show',
    dag=dag,
)

# Task 8: Generate documentation
docs_task = BashOperator(
    task_id='dbt_docs_generate',
    bash_command='cd ~/si-dbt-test/my_dbt_project && source ../env/bin/activate && dbt docs generate',
    dag=dag,
)

# End task
end_task = DummyOperator(
    task_id='pipeline_complete',
    dag=dag,
)

# Set task dependencies
start_task >> [validate_task, dbt_check_task]
[validate_task, dbt_check_task] >> generate_task
generate_task >> seed_task >> run_staging_task >> run_generated_task
run_generated_task >> test_task >> docs_task >> end_task
