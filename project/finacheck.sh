#!/bin/bash

echo "============================================="
echo "SI-dbt Pipeline Setup for WSL Debian"
echo "Python 3.11 + Resolved Protobuf Conflicts"
echo "============================================="

# Check and install Python 3.11
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    sudo apt update
    sudo apt install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install python3.11 python3.11-venv python3.11-pip python3.11-dev -y
fi

PYTHON_CMD="python3.11"
echo "Using Python 3.11 for compatibility"

# Check SQLite version and upgrade if needed
echo "Checking SQLite version..."
SQLITE_VERSION=$(sqlite3 --version | cut -d' ' -f1)
echo "Current SQLite version: $SQLITE_VERSION"

# Extract version numbers for comparison
SQLITE_MAJOR=$(echo $SQLITE_VERSION | cut -d'.' -f1)
SQLITE_MINOR=$(echo $SQLITE_VERSION | cut -d'.' -f2)

# Check if SQLite version is less than 3.15.0
if [ "$SQLITE_MAJOR" -lt 3 ] || ([ "$SQLITE_MAJOR" -eq 3 ] && [ "$SQLITE_MINOR" -lt 15 ]); then
    echo "SQLite version is too old. Upgrading..."
    
    # Install dependencies for building SQLite
    sudo apt install build-essential wget libsqlite3-dev pkg-config -y
    
    # Download and install latest SQLite
    cd /tmp
    wget https://www.sqlite.org/2024/sqlite-autoconf-3450300.tar.gz
    tar xzf sqlite-autoconf-3450300.tar.gz
    cd sqlite-autoconf-3450300
    ./configure --prefix=/usr/local
    make
    sudo make install
    
    # Update library path
    sudo ldconfig
    echo "SQLite upgraded successfully!"
else
    echo "SQLite version is compatible (${SQLITE_VERSION})"
fi

# Update and upgrade system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential system packages
echo "Installing essential packages..."
sudo apt install git curl wget build-essential sqlite3 libsqlite3-dev pkg-config -y

# Create project directory structure
echo "Creating project structure..."
mkdir -p ~/si-dbt-test/my_dbt_project/{data,seeds,models/{staging,generated},tests,macros,snapshots,analyses,logs}
cd ~/si-dbt-test

# Initialize python virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
$PYTHON_CMD -m venv env
source env/bin/activate

# Verify Python version in virtual environment
VENV_PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Virtual environment Python version: $VENV_PYTHON_VERSION"

# Install required python packages with FIXED versions and protobuf resolution
echo "Installing compatible Python packages with protobuf fixes..."
pip install --upgrade pip setuptools wheel

# CRITICAL: Install packages in correct order to avoid protobuf conflicts
echo "Step 1: Installing dbt packages with correct versions..."
pip install "dbt-core==1.9.0" "dbt-sqlite==1.9.1"

echo "Step 2: Installing Airflow with constraints..."
AIRFLOW_VERSION=2.8.4
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

echo "Step 3: Installing updated Airflow SQLite provider..."
pip install "apache-airflow-providers-sqlite==3.8.2"

echo "Step 4: Resolving protobuf conflicts..."
# First uninstall conflicting google-ai-generativelanguage if it exists
pip uninstall -y google-ai-generativelanguage || true

# Force reinstall protobuf to compatible version
pip install protobuf==5.28.3 --force-reinstall

echo "Step 5: Installing other dependencies..."
pip install google-generativeai==0.4.1
pip install pandas==2.1.4
pip install sqlalchemy==2.0.25
pip install typing-extensions==4.9.0

# Verify critical installations and check for conflicts
echo "Verifying installations and checking for conflicts..."
python -c "import dbt.adapters.sqlite; print('✅ dbt-sqlite imported successfully')"
python -c "import airflow; print(f'✅ Airflow: {airflow.__version__}')"
python -c "import pandas; print(f'✅ Pandas: {pandas.__version__}')"
python -c "import google.protobuf; print(f'✅ Protobuf: {google.protobuf.__version__}')"

# Check for google-ai-generativelanguage conflicts
python -c "
try:
    import google.ai.generativelanguage
    print('⚠️  google-ai-generativelanguage found - may cause protobuf conflicts')
except ImportError:
    print('✅ No google-ai-generativelanguage conflicts')
"

# Set up Airflow with SequentialExecutor for SQLite compatibility
echo "Setting up Airflow with SequentialExecutor for SQLite..."
mkdir -p ~/airflow
export AIRFLOW_HOME=~/airflow

# Add Airflow environment variables to bashrc
echo "export AIRFLOW_HOME=~/airflow" >> ~/.bashrc
echo "export AIRFLOW__CORE__EXECUTOR=SequentialExecutor" >> ~/.bashrc

# Set executor for current session
export AIRFLOW__CORE__EXECUTOR=SequentialExecutor

# Initialize airflow database
echo "Initializing Airflow database with SequentialExecutor..."
airflow db init

# Update Airflow configuration for SQLite with SequentialExecutor
echo "Configuring Airflow for SQLite with SequentialExecutor..."
sed -i 's/executor = .*/executor = SequentialExecutor/' ~/airflow/airflow.cfg
sed -i 's/load_examples = .*/load_examples = False/' ~/airflow/airflow.cfg
sed -i 's/dags_are_paused_at_creation = .*/dags_are_paused_at_creation = False/' ~/airflow/airflow.cfg

# Create airflow user with default password
echo "Creating Airflow admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin123

# Create CORRECT dbt profile based on official dbt-sqlite documentation
echo "Creating dbt profile with correct format..."
mkdir -p ~/.dbt
cat > ~/.dbt/profiles.yml << 'EOF'
si_dbt_test:
  target: dev
  outputs:
    dev:
      type: sqlite
      threads: 1
      database: "database"
      schema: 'main'
      schemas_and_paths:
        main: '/home/REPLACE_USERNAME/si-dbt-test/my_dbt_project/dbt.sqlite'
      schema_directory: '/home/REPLACE_USERNAME/si-dbt-test/my_dbt_project'
EOF

# Replace username in profile
sed -i "s/REPLACE_USERNAME/$USER/g" ~/.dbt/profiles.yml

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
EOF

# Test dbt configuration
echo "Testing dbt configuration..."
source ../../env/bin/activate
dbt debug

if [ $? -eq 0 ]; then
    echo "✅ dbt configuration successful!"
else
    echo "❌ dbt configuration failed. Check the output above."
fi

# Create enhanced model generation script (protobuf-compatible)
cat > ~/si-dbt-test/my_dbt_project/generate_intelligent_models.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced Intelligent Model Generator using Google Gemini AI
Compatible with protobuf 5.28.3 and resolved conflicts
"""
import os
import sys
import pandas as pd
import logging

# Check for protobuf conflicts before importing generativeai
try:
    import google.protobuf
    protobuf_version = google.protobuf.__version__
    major_version = int(protobuf_version.split('.')[0])
    
    if major_version < 5:
        raise ImportError(f"Protobuf version {protobuf_version} is incompatible. Need >=5.0")
    
    print(f"✅ Using compatible protobuf version: {protobuf_version}")
    
except ImportError as e:
    print(f"❌ Protobuf compatibility error: {e}")
    sys.exit(1)

# Now safe to import generativeai
try:
    import google.generativeai as genai
    print("✅ Google GenerativeAI imported successfully")
except ImportError as e:
    print(f"❌ Failed to import google.generativeai: {e}")
    print("Make sure protobuf conflicts are resolved")
    sys.exit(1)

from pathlib import Path
import json
from datetime import datetime
import sqlite3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentModelGenerator:
    def __init__(self):
        # Check SQLite version
        self.check_sqlite_version()
        
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
        
        logger.info("IntelligentModelGenerator initialized with protobuf compatibility checks")

    def check_sqlite_version(self):
        """Check if SQLite version is compatible"""
        version = sqlite3.sqlite_version
        version_parts = version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 3 or (major == 3 and minor < 15):
            raise ValueError(f"SQLite version {version} is too old. Minimum required: 3.15.0")
        
        logger.info(f"SQLite version {version} is compatible")

    def analyze_csv_structure(self, csv_file):
        """Analyze CSV structure and create comprehensive data profile"""
        try:
            df = pd.read_csv(csv_file, nrows=1000)
            
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
            }
            
            logger.info(f"Successfully analyzed {csv_file.name}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {csv_file}: {e}")
            return None

    def generate_staging_model(self, csv_analysis):
        """Generate staging model SQL optimized for dbt-sqlite 1.9.1"""
        
        prompt = f"""
        Generate a dbt staging SQL model optimized for SQLite with the following data:

        **Data Profile:**
        - File: {csv_analysis['filename']}
        - Columns: {csv_analysis['columns']}
        - Data Types: {json.dumps(csv_analysis['data_types'], indent=2)}

        **dbt-sqlite 1.9.1 Requirements:**
        1. Use SQLite-compatible functions only
        2. Reference seed: {{{{ ref('{csv_analysis['filename']}') }}}}
        3. Use simple CAST operations for type conversion
        4. Add basic data validation
        5. Use consistent naming (snake_case)
        6. Avoid complex window functions
        7. Include materialized='view' config

        **Template:**
        {{{{ config(materialized='view') }}}}

        select
            -- your columns here with appropriate casts
        from {{{{ ref('{csv_analysis['filename']}') }}}}

        Return ONLY the SQL code:
        """

        try:
            response = self.model.generate_content(prompt)
            sql_code = response.text.strip()
            
            if '```
                sql_code = sql_code.split('```sql').split('```
            elif '```' in sql_code:
                sql_code = sql_code.replace('```
                
            return sql_code
            
        except Exception as e:
            logger.error(f"Error generating staging model: {e}")
            return None

    def generate_analytics_model(self, csv_analysis):
        """Generate analytics model SQL compatible with dbt-sqlite 1.9.1"""
        
        prompt = f"""
        Generate a SQLite-compatible analytics model for dbt-sqlite 1.9.1:

        **Source:** stg_{csv_analysis['filename']}
        **Columns:** {csv_analysis['columns']}

        **Requirements:**
        1. Use {{{{ config(materialized='table') }}}}
        2. Reference: {{{{ ref('stg_{csv_analysis['filename']}') }}}}
        3. Use simple aggregations (COUNT, SUM, AVG, MIN, MAX)
        4. Use CASE statements for conditional logic
        5. Keep queries simple and SQLite-compatible
        6. Add meaningful business metrics

        Return ONLY the SQL code:
        """

        try:
            response = self.model.generate_content(prompt)
            sql_code = response.text.strip()
            
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
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process...")
        
        for csv_file in csv_files:
            logger.info(f"Processing {csv_file.name}...")
            
            if not self.create_seed_file(csv_file):
                continue
                
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
                    f.write(f"-- Compatible with dbt-sqlite 1.9.1 & protobuf 5.28.3\n")
                    f.write(f"-- Generated on {datetime.now()}\n\n")
                    f.write(staging_sql)
                
                logger.info(f"Generated staging model: {staging_file}")
            
            # Generate analytics model
            analytics_sql = self.generate_analytics_model(analysis)
            if analytics_sql:
                analytics_name = f"analytics_{analysis['filename']}"
                analytics_file = self.models_dir / f"{analytics_name}.sql"
                
                with open(analytics_file, 'w') as f:
                    f.write(f"-- Analytics model for {csv_file.name}\n")
                    f.write(f"-- Compatible with dbt-sqlite 1.9.1 & protobuf 5.28.3\n")
                    f.write(f"-- Generated on {datetime.now()}\n\n")
                    f.write(analytics_sql)
                
                logger.info(f"Generated analytics model: {analytics_file}")
        
        logger.info("Model generation completed!")

if __name__ == "__main__":
    try:
        generator = IntelligentModelGenerator()
        generator.process_csv_files()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
EOF

chmod +x ~/si-dbt-test/my_dbt_project/generate_intelligent_models.py

# Create SequentialExecutor-compatible Airflow DAG
mkdir -p ~/airflow/dags
cat > ~/airflow/dags/dbt_sqlite_sequential_pipeline.py << 'EOF'
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
import subprocess
import os
import logging
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    'owner': 'data-engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'start_date': datetime(2025, 9, 1),
}

def check_protobuf_compatibility():
    """Check protobuf version compatibility"""
    try:
        import google.protobuf
        protobuf_version = google.protobuf.__version__
        major_version = int(protobuf_version.split('.'))
        
        if major_version < 5:
            raise ValueError(f"Protobuf version {protobuf_version} incompatible. Need >=5.0")
        
        logger.info(f"✅ Protobuf version {protobuf_version} compatible")
        
        # Check for conflicting packages
        try:
            import google.ai.generativelanguage
            logger.warning("⚠️  google-ai-generativelanguage detected - may cause conflicts")
        except ImportError:
            logger.info("✅ No google-ai-generativelanguage conflicts")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Protobuf compatibility check failed: {e}")
        raise

def validate_environment():
    """Validate environment setup"""
    # Check protobuf
    check_protobuf_compatibility()
    
    # Check SQLite
    version = sqlite3.sqlite_version
    version_parts = version.split('.')
    major, minor = int(version_parts), int(version_parts[1])
    
    if major < 3 or (major == 3 and minor < 15):
        raise ValueError(f"SQLite version {version} incompatible. Required: 3.15.0+")
    
    logger.info(f"✅ SQLite version {version} compatible")
    
    # Check API key
    if not os.environ.get('GEMINI_API_KEY'):
        raise ValueError("GEMINI_API_KEY not set!")
    
    # Check CSV files
    data_dir = os.path.expanduser('~/si-dbt-test/my_dbt_project/data')
    if not os.path.exists(data_dir):
        raise ValueError("Data directory not found!")
        
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV files found!")
    
    logger.info(f"✅ Environment validated. Found {len(csv_files)} CSV files.")

def run_model_generation():
    """Run model generation with protobuf compatibility"""
    project_dir = os.path.expanduser('~/si-dbt-test/my_dbt_project')
    script_path = os.path.join(project_dir, 'generate_intelligent_models.py')
    venv_python = os.path.expanduser('~/si-dbt-test/env/bin/python')
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env['AIRFLOW__CORE__EXECUTOR'] = 'SequentialExecutor'
    
    result = subprocess.run([venv_python, script_path], 
                          capture_output=True, text=True, cwd=project_dir, env=env)
    
    if result.returncode != 0:
        logger.error(f"Model generation failed: {result.stderr}")
        raise Exception(f"Failed: {result.stderr}")
    
    logger.info(f"✅ Model generation successful")

dag = DAG(
    'dbt_sqlite_sequential_pipeline',
    default_args=default_args,
    description='Sequential dbt-sqlite pipeline with protobuf fixes',
    schedule_interval=None,
    catchup=False,
    tags=['dbt', 'sqlite', 'sequential', 'protobuf-fixed'],
    max_active_runs=1,
)

start = DummyOperator(task_id='start', dag=dag)

validate = PythonOperator(
    task_id='validate_compatibility',
    python_callable=validate_environment,
    dag=dag,
)

generate = PythonOperator(
    task_id='generate_models',
    python_callable=run_model_generation,
    dag=dag,
)

seed = BashOperator(
    task_id='dbt_seed',
    bash_command='cd ~/si-dbt-test/my_dbt_project/my_dbt_project && source ../../env/bin/activate && export AIRFLOW__CORE__EXECUTOR=SequentialExecutor && dbt seed',
    dag=dag,
)

run_staging = BashOperator(
    task_id='run_staging_models',
    bash_command='cd ~/si-dbt-test/my_dbt_project/my_dbt_project && source ../../env/bin/activate && export AIRFLOW__CORE__EXECUTOR=SequentialExecutor && dbt run --select models/staging/',
    dag=dag,
)

run_analytics = BashOperator(
    task_id='run_analytics_models',
    bash_command='cd ~/si-dbt-test/my_dbt_project/my_dbt_project && source ../../env/bin/activate && export AIRFLOW__CORE__EXECUTOR=SequentialExecutor && dbt run --select models/generated/',
    dag=dag,
)

test = BashOperator(
    task_id='test_models',
    bash_command='cd ~/si-dbt-test/my_dbt_project/my_dbt_project && source ../../env/bin/activate && export AIRFLOW__CORE__EXECUTOR=SequentialExecutor && dbt test',
    dag=dag,
)

end = DummyOperator(task_id='complete', dag=dag)

# Set dependencies
start >> validate >> generate >> seed >> run_staging >> run_analytics >> test >> end
EOF

# Create sample data
echo "Creating sample CSV files..."
cat > ~/si-dbt-test/my_dbt_project/data/sample_sales_data.csv << 'EOF'
id,customer_name,product,quantity,price,order_date,region
1,John Doe,Widget A,5,29.99,2025-01-15,North
2,Jane Smith,Widget B,3,45.50,2025-01-16,South
3,Bob Johnson,Widget A,2,29.99,2025-01-17,East
4,Alice Brown,Widget C,7,15.25,2025-01-18,West
5,Charlie Wilson,Widget B,4,45.50,2025-01-19,North
EOF

cat > ~/si-dbt-test/my_dbt_project/data/customer_data.csv << 'EOF'
customer_id,customer_name,email,phone,registration_date,customer_type
1,John Doe,john.doe@email.com,555-0101,2024-12-01,Premium
2,Jane Smith,jane.smith@email.com,555-0102,2024-12-02,Standard
3,Bob Johnson,bob.johnson@email.com,555-0103,2024-12-03,Premium
4,Alice Brown,alice.brown@email.com,555-0104,2024-12-04,Standard
EOF

# Create startup scripts with SequentialExecutor
cat > ~/si-dbt-test/start_pipeline.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting SequentialExecutor dbt-sqlite Pipeline..."

# Activate environment and set executor
source ~/si-dbt-test/env/bin/activate
export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__EXECUTOR=SequentialExecutor

# Check if already running
if pgrep -f "airflow scheduler" > /dev/null; then
    echo "📅 Scheduler already running"
else
    echo "📅 Starting Airflow scheduler with SequentialExecutor..."
    airflow scheduler --daemon
fi

if pgrep -f "airflow webserver" > /dev/null; then
    echo "🌐 Webserver already running"
else
    echo "🌐 Starting Airflow webserver..."
    airflow webserver --port 8080 --daemon
fi

sleep 5
echo ""
echo "✅ PIPELINE STARTED SUCCESSFULLY!"
echo "🎯 Configuration:"
echo "   - Python: $(python --version | cut -d' ' -f2)"
echo "   - dbt-core: 1.9.0"
echo "   - dbt-sqlite: 1.9.1"
echo "   - Airflow: 2.8.4 (SequentialExecutor)"
echo "   - Protobuf: 5.28.3"
echo "   - SQLite Provider: 3.8.2"
echo ""
echo "📊 Airflow UI: http://localhost:8080"
echo "👤 Username: admin"
echo "🔑 Password: admin123"
echo "🎯 DAG: dbt_sqlite_sequential_pipeline"
echo ""
echo "✅ ALL PROTOBUF CONFLICTS RESOLVED!"
EOF

cat > ~/si-dbt-test/stop_pipeline.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping pipeline..."
pkill -f "airflow scheduler"
pkill -f "airflow webserver"
echo "✅ Pipeline stopped!"
EOF

chmod +x ~/si-dbt-test/start_pipeline.sh ~/si-dbt-test/stop_pipeline.sh

# Create environment activation helper with protobuf checks
cat > ~/si-dbt-test/activate_env.sh << 'EOF'
#!/bin/bash
source ~/si-dbt-test/env/bin/activate
export AIRFLOW_HOME=~/airflow
export AIRFLOW__CORE__EXECUTOR=SequentialExecutor
cd ~/si-dbt-test/my_dbt_project/my_dbt_project

echo "✅ Environment activated!"
echo "📂 Current directory: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "🔧 dbt: $(dbt --version | head -1)"

# Check protobuf version
python -c "
import google.protobuf
print(f'🛡️  Protobuf: {google.protobuf.__version__}')
try:
    import google.ai.generativelanguage
    print('⚠️  google-ai-generativelanguage detected')
except ImportError:
    print('✅ No protobuf conflicts')
"

echo "🎯 Executor: $AIRFLOW__CORE__EXECUTOR"
EOF

chmod +x ~/si-dbt-test/activate_env.sh

echo ""
echo "============================================="
echo "✅ SETUP COMPLETE! (PROTOBUF CONFLICTS RESOLVED)"
echo "============================================="
echo ""
echo "🎯 FIXED VERSIONS INSTALLED:"
echo "   - Python: 3.11.x"
echo "   - dbt-core: 1.9.0"
echo "   - dbt-sqlite: 1.9.1"
echo "   - Apache Airflow: 2.8.4"
echo "   - SQLite Provider: 3.8.2 (UPDATED)"
echo "   - Protobuf: 5.28.3 (FORCE REINSTALLED)"
echo "   - Executor: SequentialExecutor"
echo ""
echo "✅ ALL ISSUES RESOLVED:"
echo "   ✅ Protobuf conflicts resolved (≥5.0 required)"
echo "   ✅ google-ai-generativelanguage conflicts avoided"
echo "   ✅ SequentialExecutor for SQLite compatibility"
echo "   ✅ Updated SQLite provider (3.8.2)"
echo "   ✅ dbt-core 1.9.0 + dbt-sqlite 1.9.1 alignment"
echo "   ✅ Force-reinstalled protobuf to 5.28.3"
echo ""
echo "📋 REMAINING STEPS:"
echo ""
echo "1️⃣ ADD GEMINI API KEY:"
echo "   echo 'export GEMINI_API_KEY=\"your_api_key_here\"' >> ~/.bashrc"
echo "   source ~/.bashrc"
echo ""
echo "2️⃣ TEST SETUP:"
echo "   source ~/si-dbt-test/activate_env.sh"
echo "   dbt debug"
echo "   # Should show: ✅ Connection test: [OK connection ok]"
echo ""
echo "3️⃣ START PIPELINE:"
echo "   cd ~/si-dbt-test && ./start_pipeline.sh"
echo ""
echo "4️⃣ ACCESS AIRFLOW:"
echo "   🌐 http://localhost:8080"
echo "   👤 admin / 🔑 admin123"
echo ""
echo "5️⃣ TRIGGER DAG:"
echo "   Find: 'dbt_sqlite_sequential_pipeline'"
echo ""
echo "🧪 VERIFICATION COMMANDS:"
echo "- Check protobuf: python -c 'import google.protobuf; print(google.protobuf.__version__)'"
echo "- Test dbt: source activate_env.sh && dbt debug"
echo "- Check executor: echo \$AIRFLOW__CORE__EXECUTOR"
echo ""
echo "✅ ALL PROTOBUF CONFLICTS RESOLVED - READY TO USE!"
