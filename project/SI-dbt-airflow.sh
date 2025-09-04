#!/bin/bash

echo "============================================="
echo "SI-dbt Pipeline Setup for WSL Debian"
echo "Python 3.11.9 Optimized Version"
echo "============================================="

# Check Python version compatibility
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Detected Python version: $PYTHON_VERSION"

# Ensure we're using Python 3.11.9 or compatible version
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    sudo apt update
    sudo apt install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install python3.11 python3.11-venv python3.11-pip python3.11-dev -y
fi

PYTHON_CMD="python3.11"
echo "Using Python 3.11 for optimal compatibility"

# Update and upgrade system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential system packages
echo "Installing essential packages..."
sudo apt install git curl wget build-essential sqlite3 libsqlite3-dev pkg-config -y

# Setup python virtual environment
echo "Setting up Python environment..."
$PYTHON_CMD -m pip install --upgrade pip

# Create project directory structure
echo "Creating project structure..."
mkdir -p ~/si-dbt-test/my_dbt_project/{data,seeds,models/generated,dags,logs,analyses,tests,macros,snapshots}
cd ~/si-dbt-test

# Initialize python virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
$PYTHON_CMD -m venv env
source env/bin/activate

# Verify Python version in virtual environment
VENV_PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Virtual environment Python version: $VENV_PYTHON_VERSION"

# Install required python packages with Python 3.11 optimized versions
echo "Installing Python packages optimized for Python 3.11..."
pip install --upgrade pip setuptools wheel

# Core packages with specific versions for Python 3.11 compatibility
pip install dbt-core==1.7.13
pip install dbt-sqlite==1.7.7
pip install apache-airflow==2.8.4
pip install apache-airflow-providers-sqlite==3.7.2
pip install google-generativeai==0.4.1
pip install pandas==2.1.4
pip install sqlalchemy==2.0.25

# Additional dependencies for stability
pip install typing-extensions==4.9.0
pip install pydantic==2.5.3
pip install cryptography==42.0.5

# Create airflow home directory
echo "Setting up Airflow..."
mkdir -p ~/airflow
export AIRFLOW_HOME=~/airflow

# Add Airflow home to bashrc for persistence
echo "export AIRFLOW_HOME=~/airflow" >> ~/.bashrc

# Initialize airflow database
airflow db init

# Create airflow user (will prompt for password)
echo "Creating Airflow admin user..."
echo "Please set a password when prompted (recommended: admin123):"
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Create dbt profile
echo "Creating dbt profile..."
mkdir -p ~/.dbt
cat > ~/.dbt/profiles.yml << 'EOF'
si_dbt_test:
  target: dev
  outputs:
    dev:
      type: sqlite
      path: ~/si-dbt-test/my_dbt_project/dbt.sqlite
      threads: 1
      timeout_seconds: 300
      retries: 3
EOF

# Initialize dbt project
echo "Initializing dbt project..."
cd ~/si-dbt-test/my_dbt_project
dbt init --skip-profile-setup my_dbt_project
cd my_dbt_project

# Create enhanced dbt_project.yml
cat > dbt_project.yml << 'EOF'
name: 'si_dbt_pipeline'
version: '1.0.0'
config-version: 2

profile: 'si_dbt_test'

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

target-path: "target"
clean-targets:
  - "target"
  - "dbt_packages"

models:
  si_dbt_pipeline:
    generated:
      +materialized: table
      +tags: ["ai_generated"]
    staging:
      +materialized: view
      +tags: ["staging"]

seeds:
  si_dbt_pipeline:
    +quote_columns: false
    +column_types:
      id: integer
EOF

# Create enhanced model generation script
cat > ~/si-dbt-test/my_dbt_project/generate_intelligent_models.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced Intelligent Model Generator using Google Gemini AI
Python 3.11.9 Optimized Version
"""
import os
import sys
import pandas as pd
import google.generativeai as genai
from pathlib import Path
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentModelGenerator:
    def __init__(self):
        self.api_key = os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set!")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        
        self.data_dir = Path('./data')
        self.models_dir = Path('./models/generated')
        self.staging_dir = Path('./models/staging')
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("IntelligentModelGenerator initialized successfully")

    def analyze_csv_structure(self, csv_file):
        """Analyze CSV structure and create comprehensive data profile"""
        try:
            # Read sample data
            df = pd.read_csv(csv_file, nrows=1000)  # Increased sample size
            
            # Enhanced analysis
            analysis = {
                'filename': csv_file.stem,
                'row_count': len(df),
                'columns': list(df.columns),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                'sample_data': df.head(5).to_dict('records'),
                'null_counts': df.isnull().sum().to_dict(),
                'unique_values': {col: df[col].nunique() for col in df.columns},
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                'date_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
                'text_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            logger.info(f"Successfully analyzed {csv_file.name}: {analysis['row_count']} rows, {len(analysis['columns'])} columns")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {csv_file}: {e}")
            return None

    def generate_staging_model(self, csv_analysis):
        """Generate staging model SQL"""
        
        prompt = f"""
        Generate a dbt staging SQL model for CSV data with the following requirements:

        **Data Profile:**
        - File: {csv_analysis['filename']}
        - Rows: {csv_analysis['row_count']}
        - Columns: {csv_analysis['columns']}
        - Data Types: {json.dumps(csv_analysis['data_types'], indent=2)}
        - Numeric Columns: {csv_analysis['numeric_columns']}
        - Text Columns: {csv_analysis['text_columns']}

        **Requirements:**
        1. Create a staging model that reads from seeds/{csv_analysis['filename']}.csv
        2. Use appropriate SQL casting for data types
        3. Add data quality checks (null handling, duplicates)
        4. Use consistent naming conventions (snake_case)
        5. Add basic data validation
        6. Include useful comments
        7. Follow dbt best practices

        Return ONLY the SQL code for the dbt model:
        """

        try:
            response = self.model.generate_content(prompt)
            sql_code = response.text.strip()
            
            # Clean SQL code
            if '```
                sql_code = sql_code.split('```sql').split('```
            elif '```' in sql_code:
                sql_code = sql_code.replace('```
                
            return sql_code
            
        except Exception as e:
            logger.error(f"Error generating staging model: {e}")
            return None

    def generate_analytics_model(self, csv_analysis):
        """Generate analytics model SQL"""
        
        prompt = f"""
        Generate a dbt analytics SQL model that transforms the staging data:

        **Source Data:**
        - Staging Model: stg_{csv_analysis['filename']}
        - Columns: {csv_analysis['columns']}
        - Numeric Columns: {csv_analysis['numeric_columns']}

        **Requirements:**
        1. Reference the staging model: {{{{ ref('stg_{csv_analysis['filename']}') }}}}
        2. Add meaningful aggregations and calculations
        3. Include business logic transformations
        4. Add derived columns that provide insights
        5. Group data logically if applicable
        6. Include data quality metrics
        7. Use window functions where beneficial

        Return ONLY the SQL code for the analytics model:
        """

        try:
            response = self.model.generate_content(prompt)
            sql_code = response.text.strip()
            
            # Clean SQL code
            if '```sql' in sql_code:
                sql_code = sql_code.split('``````').strip()
            elif '```
                sql_code = sql_code.replace('```', '').strip()
                
            return sql_code
            
        except Exception as e:
            logger.error(f"Error generating analytics model: {e}")
            return None

    def create_seed_file(self, csv_file):
        """Copy CSV file to seeds directory"""
        try:
            seeds_dir = Path('./seeds')
            seeds_dir.mkdir(exist_ok=True)
            
            import shutil
            seed_file = seeds_dir / csv_file.name
            shutil.copy2(csv_file, seed_file)
            
            logger.info(f"Created seed file: {seed_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating seed file: {e}")
            return False

    def process_csv_files(self):
        """Process all CSV files in data directory"""
        csv_files = list(self.data_dir.glob('*.csv'))
        
        if not csv_files:
            logger.warning("No CSV files found in ./data directory")
            logger.info("MANUAL STEP: Place your CSV files in ~/si-dbt-test/my_dbt_project/data/")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process...")
        
        for csv_file in csv_files:
            logger.info(f"Processing {csv_file.name}...")
            
            # Create seed file
            if not self.create_seed_file(csv_file):
                continue
                
            # Analyze CSV structure
            analysis = self.analyze_csv_structure(csv_file)
            if not analysis:
                continue
            
            # Generate staging model
            staging_sql = self.generate_staging_model(analysis)
            if staging_sql:
                staging_name = f"stg_{analysis['filename']}"
                staging_file = self.staging_dir / f"{staging_name}.sql"
                
                with open(staging_file, 'w') as f:
                    f.write(f"-- Staging model for {csv_file.name}\n")
                    f.write(f"-- Generated by Gemini AI on {datetime.now()}\n")
                    f.write(f"-- Python version: {sys.version}\n\n")
                    f.write(staging_sql)
                
                logger.info(f"Generated staging model: {staging_file}")
            
            # Generate analytics model
            analytics_sql = self.generate_analytics_model(analysis)
            if analytics_sql:
                analytics_name = f"analytics_{analysis['filename']}"
                analytics_file = self.models_dir / f"{analytics_name}.sql"
                
                with open(analytics_file, 'w') as f:
                    f.write(f"-- Analytics model for {csv_file.name}\n")
                    f.write(f"-- Generated by Gemini AI on {datetime.now()}\n")
                    f.write(f"-- Python version: {sys.version}\n\n")
                    f.write(analytics_sql)
                
                logger.info(f"Generated analytics model: {analytics_file}")
        
        logger.info("Model generation completed successfully!")

if __name__ == "__main__":
    try:
        generator = IntelligentModelGenerator()
        generator.process_csv_files()
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        print("MANUAL STEP: Add your Gemini API key to ~/.bashrc")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
EOF

# Make the script executable
chmod +x ~/si-dbt-test/my_dbt_project/generate_intelligent_models.py

# Create enhanced Airflow DAG
mkdir -p ~/airflow/dags
cat > ~/airflow/dags/dbt_auto_csv_dag.py << 'EOF'
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
EOF

# Create sample CSV files for testing
echo "Creating sample CSV files..."

# Sales data
cat > ~/si-dbt-test/my_dbt_project/data/sample_sales_data.csv << 'EOF'
id,customer_name,product,quantity,price,order_date,region
1,John Doe,Widget A,5,29.99,2025-01-15,North
2,Jane Smith,Widget B,3,45.50,2025-01-16,South
3,Bob Johnson,Widget A,2,29.99,2025-01-17,East
4,Alice Brown,Widget C,7,15.25,2025-01-18,West
5,Charlie Wilson,Widget B,4,45.50,2025-01-19,North
6,Diana Prince,Widget A,1,29.99,2025-01-20,South
7,Bruce Wayne,Widget C,10,15.25,2025-01-21,East
8,Clark Kent,Widget B,6,45.50,2025-01-22,West
EOF

# Customer data
cat > ~/si-dbt-test/my_dbt_project/data/customer_data.csv << 'EOF'
customer_id,customer_name,email,phone,registration_date,customer_type
1,John Doe,john.doe@email.com,555-0101,2024-12-01,Premium
2,Jane Smith,jane.smith@email.com,555-0102,2024-12-02,Standard
3,Bob Johnson,bob.johnson@email.com,555-0103,2024-12-03,Premium
4,Alice Brown,alice.brown@email.com,555-0104,2024-12-04,Standard
5,Charlie Wilson,charlie.wilson@email.com,555-0105,2024-12-05,Premium
EOF

# Create startup script for convenience
cat > ~/si-dbt-test/start_pipeline.sh << 'EOF'
#!/bin/bash

echo "Starting SI-dbt Pipeline (Python 3.11 Optimized)..."

# Activate virtual environment
source ~/si-dbt-test/env/bin/activate

# Set Airflow home
export AIRFLOW_HOME=~/airflow

# Check if Airflow processes are already running
if pgrep -f "airflow scheduler" > /dev/null; then
    echo "Airflow scheduler is already running"
else
    echo "Starting Airflow scheduler..."
    airflow scheduler --daemon
fi

if pgrep -f "airflow webserver" > /dev/null; then
    echo "Airflow webserver is already running"
else
    echo "Starting Airflow webserver on port 8080..."
    airflow webserver --port 8080 --daemon
fi

# Wait for services to start
sleep 10

echo "Pipeline started successfully!"
echo ""
echo "Services Status:"
echo "- Python version: $(python --version)"
echo "- dbt version: $(dbt --version | head -1)"
echo "- Airflow UI: http://localhost:8080"
echo "- Username: admin"
echo "- Default Password: admin123"
echo ""
echo "To stop services:"
echo "./stop_pipeline.sh"
echo ""
echo "To check logs:"
echo "tail -f ~/airflow/logs/scheduler/latest/*.log"
EOF

chmod +x ~/si-dbt-test/start_pipeline.sh

# Create stop script
cat > ~/si-dbt-test/stop_pipeline.sh << 'EOF'
#!/bin/bash

echo "Stopping SI-dbt Pipeline..."

# Stop Airflow processes
pkill -f "airflow scheduler"
pkill -f "airflow webserver"

echo "Pipeline stopped successfully!"
EOF

chmod +x ~/si-dbt-test/stop_pipeline.sh

# Create environment activation script
cat > ~/si-dbt-test/activate_env.sh << 'EOF'
#!/bin/bash
source ~/si-dbt-test/env/bin/activate
export AIRFLOW_HOME=~/airflow
cd ~/si-dbt-test/my_dbt_project
echo "Environment activated! Python version: $(python --version)"
echo "Current directory: $(pwd)"
EOF

chmod +x ~/si-dbt-test/activate_env.sh

# Final instructions
echo ""
echo "============================================="
echo "SETUP COMPLETE! (Python 3.11 Optimized)"
echo "============================================="
echo ""
echo "Configuration Summary:"
echo "- Python version: $(python --version)"
echo "- dbt-core version: 1.7.13"
echo "- Apache Airflow version: 2.8.4"
echo "- Virtual environment: ~/si-dbt-test/env"
echo ""
echo "MANUAL STEPS REQUIRED:"
echo ""
echo "1. ADD GEMINI API KEY:"
echo "   echo 'export GEMINI_API_KEY=\"AIzaSyBkoU4uxSqMLwV2B9MST48_3g7sATrl7QY\"' >> ~/.bashrc"
echo "   source ~/.bashrc"
echo ""
echo "2. VERIFY API KEY:"
echo "   echo \$GEMINI_API_KEY"
echo ""
echo "3. ADD YOUR CSV FILES (Optional - samples included):"
echo "   cp your_data.csv ~/si-dbt-test/my_dbt_project/data/"
echo ""
echo "4. START THE PIPELINE:"
echo "   cd ~/si-dbt-test"
echo "   ./start_pipeline.sh"
echo ""
echo "5. ACCESS AIRFLOW UI:"
echo "   http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin123 (or what you set)"
echo ""
echo "6. TRIGGER DAG:"
echo "   In Airflow UI, find and trigger: 'dbt_auto_csv_pipeline'"
echo ""
echo "PROJECT STRUCTURE:"
echo "~/si-dbt-test/"
echo "├── env/                           (Python 3.11 virtual environment)"
echo "├── my_dbt_project/"
echo "│   ├── data/                     (Your CSV files + samples)"
echo "│   ├── models/"
echo "│   │   ├── staging/             (Staging models)"
echo "│   │   └── generated/           (AI-generated analytics models)"
echo "│   ├── seeds/                   (dbt seed files)"
echo "│   ├── tests/                   (Data tests)"
echo "│   └── generate_intelligent_models.py"
echo "├── start_pipeline.sh            (Start services)"
echo "├── stop_pipeline.sh             (Stop services)"
echo "└── activate_env.sh              (Activate environment)"
echo ""
echo "USEFUL COMMANDS:"
echo "- Activate environment: source ~/si-dbt-test/activate_env.sh"
echo "- Test dbt: dbt debug"
echo "- Run models manually: dbt run"
echo "- Generate docs: dbt docs generate && dbt docs serve"
echo ""
echo "Sample CSV files created in data/ directory for immediate testing!"
echo ""
echo "Next steps:"
echo "1. Add your Gemini API key to ~/.bashrc"
echo "2. Run: source ~/.bashrc"
echo "3. Run: ./start_pipeline.sh"
echo "4. Open http://localhost:8080 and trigger the DAG"
echo ""
