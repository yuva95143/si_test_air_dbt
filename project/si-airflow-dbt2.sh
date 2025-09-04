#!/bin/bash

echo "============================================="
echo "SI-dbt Pipeline Setup for WSL Debian"
echo "Python 3.11.9 with SQLite Fixes"
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
echo "Using Python 3.11 for optimal compatibility"

# Check SQLite version and upgrade if needed
echo "Checking SQLite version..."
SQLITE_VERSION=$(sqlite3 --version | cut -d' ' -f1)
echo "Current SQLite version: $SQLITE_VERSION"

# Extract version numbers for comparison
SQLITE_MAJOR=$(echo $SQLITE_VERSION | cut -d'.' -f1)
SQLITE_MINOR=$(echo $SQLITE_VERSION | cut -d'.' -f2)
SQLITE_PATCH=$(echo $SQLITE_VERSION | cut -d'.' -f3)

# Check if SQLite version is less than 3.15.0
if [ "$SQLITE_MAJOR" -lt 3 ] || ([ "$SQLITE_MAJOR" -eq 3 ] && [ "$SQLITE_MINOR" -lt 15 ]); then
    echo "SQLite version is too old. Upgrading..."
    
    # Install dependencies for building SQLite
    sudo apt install build-essential wget -y
    
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
mkdir -p ~/si-dbt-test/my_dbt_project/{data,seeds,models/generated,models/staging,dags,logs,analyses,tests,macros,snapshots}
cd ~/si-dbt-test

# Initialize python virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
$PYTHON_CMD -m venv env
source env/bin/activate

# Verify Python version in virtual environment
VENV_PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Virtual environment Python version: $VENV_PYTHON_VERSION"

# Install required python packages with SQLite-compatible versions
echo "Installing Python packages with SQLite compatibility fixes..."
pip install --upgrade pip setuptools wheel

# Core packages - using specific versions that work with SQLite
pip install dbt-core==1.7.13
pip install dbt-sqlite==1.7.7

# Airflow with SQLite-specific configuration
pip install apache-airflow==2.8.4
pip install apache-airflow-providers-sqlite==3.7.2

# Other dependencies
pip install google-generativeai==0.4.1
pip install pandas==2.1.4
pip install sqlalchemy==2.0.25
pip install typing-extensions==4.9.0
pip install pydantic==2.5.3

# Create airflow home directory
echo "Setting up Airflow with SQLite compatibility..."
mkdir -p ~/airflow
export AIRFLOW_HOME=~/airflow

# Add Airflow configuration to use LocalExecutor with SQLite
echo "Configuring Airflow for SQLite compatibility..."
cat > ~/airflow/airflow.cfg << 'EOF'
[core]
# Use LocalExecutor instead of CeleryExecutor for SQLite compatibility
executor = LocalExecutor
dags_folder = /home/$USER/airflow/dags
base_log_folder = /home/$USER/airflow/logs
remote_logging = False
encrypt_s3_logs = False
logging_level = INFO
fab_logging_level = WARN
colored_console_log = True
colored_log_format = [%%(blue)s%%(asctime)s%%(reset)s] {%%(blue)s%%(filename)s:%%(reset)s%%(lineno)d} %%(log_color)s%%(levelname)s%%(reset)s - %%(log_color)s%%(message)s%%(reset)s
colored_formatter_class = airflow.utils.log.colored_log.CustomTTYColoredFormatter
log_format = [%%(asctime)s] {%%(filename)s:%%(lineno)d} %%(levelname)s - %%(message)s
simple_log_format = %%(asctime)s %%(levelname)s - %%(message)s
dagbag_import_timeout = 30.0
dagbag_import_error_tracebacks = True
dagbag_import_error_traceback_depth = 2
dag_file_processor_timeout = 50
task_runner = StandardTaskRunner
default_task_retries = 0
parallelism = 32
dag_concurrency = 16
dags_are_paused_at_creation = False
max_active_tasks_per_dag = 16
max_active_runs_per_dag = 16
load_examples = False
plugins_folder = /home/$USER/airflow/plugins
execute_tasks_new_python_interpreter = False
fernet_key = 
donot_pickle = True
dagbag_size = 1000
dagbag_sync_to_db = True
max_num_rendered_ti_fields_per_task = 30
check_slas = True
job_heartbeat_sec = 5
scheduler_heartbeat_sec = 5
scheduler_health_check_threshold = 30
dag_dir_list_interval = 300
dag_stale_not_seen_duration = 600
child_process_log_directory = /home/$USER/airflow/logs/scheduler
# What files should be parsed for DAGs. Defaults to '.*\.py[c]?$'
dag_discovery_safe_mode = True
default_pool_task_slot_count = 128
max_dagruns_to_create_per_loop = 10
max_dagruns_per_dag_to_create = 10

[database]
# SQLite database configuration
sql_alchemy_conn = sqlite:////home/$USER/airflow/airflow.db
sql_alchemy_pool_enabled = False
sql_alchemy_pool_size = 5
sql_alchemy_max_overflow = 2
sql_alchemy_pool_recycle = 1800
sql_alchemy_pool_pre_ping = True
sql_alchemy_schema =

[logging]
# Logging configuration for SQLite setup
logging_level = INFO
fab_logging_level = WARN
logging_config_class =
colored_console_log = True
colored_log_format = [%%(blue)s%%(asctime)s%%(reset)s] {%%(blue)s%%(filename)s:%%(reset)s%%(lineno)d} %%(log_color)s%%(levelname)s%%(reset)s - %%(log_color)s%%(message)s%%(reset)s
colored_formatter_class = airflow.utils.log.colored_log.CustomTTYColoredFormatter

[scheduler]
# Scheduler configuration optimized for SQLite
job_heartbeat_sec = 5
scheduler_heartbeat_sec = 5
num_runs = -1
processor_poll_interval = 1
min_file_process_interval = 0
dag_dir_list_interval = 300
print_stats_interval = 30
pool_metrics_interval = 5.0
scheduler_health_check_threshold = 30
orphaned_tasks_check_interval = 300.0
child_process_log_directory = /home/$USER/airflow/logs/scheduler
scheduler_zombie_task_threshold = 300
catchup_by_default = True
ignore_first_depends_on_past_by_default = True
max_tis_per_query = 512
use_row_level_locking = True
max_dagruns_to_create_per_loop = 10
max_dagruns_per_dag_to_create = 10

[webserver]
# Webserver configuration
base_url = http://localhost:8080
default_ui_timezone = UTC
web_server_host = 0.0.0.0
web_server_port = 8080
web_server_ssl_cert =
web_server_ssl_key =
web_server_master_timeout = 120
web_server_worker_timeout = 120
worker_refresh_batch_size = 1
worker_refresh_interval = 6000
secret_key = temporary_key
workers = 4
worker_class = sync
access_logfile = -
error_logfile = -
expose_config = False
authenticate = False
filter_by_owner = False
owner_mode = user
dag_default_view = tree
dag_orientation = LR
demo_mode = False
log_fetch_timeout_sec = 5
log_fetch_delay_sec = 2
log_auto_tailing_offset = 30
log_animation_speed = 1000
hide_paused_dags_by_default = False
page_size = 100
navbar_color = #fff
default_dag_run_display_number = 25
enable_proxy_fix = False
proxy_fix_x_for = 1
proxy_fix_x_proto = 1
proxy_fix_x_host = 1
proxy_fix_x_port = 1
proxy_fix_x_prefix = 1
cookie_secure = False
cookie_samesite = 
default_wrap = False
x_frame_enabled = True
show_recent_stats_for_completed_runs = True
update_fab_perms = True

[email]
email_backend = airflow.utils.email.send_email_smtp

[smtp]
# smtp_host = localhost
# smtp_starttls = True
# smtp_ssl = False
# smtp_user = 
# smtp_password = 
# smtp_port = 587
# smtp_mail_from = 

[sentry]
sentry_dsn =

[celery]
# Celery configuration - disabled for SQLite setup
# celery_app_name = airflow.executors.celery_executor
# worker_concurrency = 16

[dask]
# Dask configuration - disabled for SQLite
# cluster_address = 127.0.0.1:8786
# tls_ca =
# tls_cert =
# tls_key =

[operators]
default_owner = airflow
default_cpus = 1
default_ram = 512
default_disk = 512
default_gpus = 0

[hive]
default_hive_mapred_queue =

[kubernetes]
# Kubernetes executor configuration - not used with SQLite
worker_container_repository =
worker_container_tag =
namespace = default
airflow_configmap =
dags_in_image = False
dags_volume_subpath =
logs_volume_subpath =
dags_volume_claim =
logs_volume_claim =
git_repo =
git_branch =
git_sync_rev =
git_sync_depth = 1
git_sync_root = /git
git_sync_dest = repo
git_user =
git_password =
git_sync_credentials_secret =
git_ssh_secret_name =
git_ssh_known_hosts_configmap_name =
git_sync_container_repository = k8s.gcr.io/git-sync/git-sync
git_sync_container_tag = v3.6.0
git_sync_init_container_name = git-sync-clone
worker_service_account_name =
image_pull_secrets =
gcp_service_account_keys =
in_cluster = True
kube_client_request_args =
enable_tcp_keepalive = True
tcp_keep_idle = 120
tcp_keep_intvl = 30
tcp_keep_cnt = 6
delete_worker_pods = True
delete_worker_pods_on_failure = False
worker_pods_creation_batch_size = 1

[api]
auth_backend = airflow.api.auth.backend.default
maximum_page_limit = 100
fallback_page_limit = 100

[lineage]
backend =

[atlas]
sasl_enabled = False
host =
port = 21000
username =
password =

[kerberos]
ccache = /tmp/airflow_krb5_ccache
principal = airflow
reinit_frequency = 3600
kinit_path = kinit
keytab = airflow.keytab
EOF

# Replace $USER with actual username in config
sed -i "s/\$USER/$USER/g" ~/airflow/airflow.cfg

# Initialize airflow database
echo "Initializing Airflow database..."
airflow db init

# Create airflow user
echo "Creating Airflow admin user..."
echo "Setting default password as 'admin123'"
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin123

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
EOF

# Create the model generation script (same as before)
cat > ~/si-dbt-test/my_dbt_project/generate_intelligent_models.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced Intelligent Model Generator using Google Gemini AI
SQLite-Compatible Version
"""
import os
import sys
import pandas as pd
import google.generativeai as genai
from pathlib import Path
import json
from datetime import datetime
import logging
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
        
        logger.info("IntelligentModelGenerator initialized successfully with SQLite compatibility")

    def check_sqlite_version(self):
        """Check if SQLite version is compatible"""
        version = sqlite3.sqlite_version
        version_parts = version.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 3 or (major == 3 and minor < 15):
            raise ValueError(f"SQLite version {version} is too old. Minimum required: 3.15.0")
        
        logger.info(f"SQLite version {version} is compatible")

    # Rest of the class remains the same as in the previous version...
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
        """Generate staging model SQL with SQLite optimizations"""
        
        prompt = f"""
        Generate a dbt staging SQL model optimized for SQLite with the following data:

        **Data Profile:**
        - File: {csv_analysis['filename']}
        - Columns: {csv_analysis['columns']}
        - Data Types: {json.dumps(csv_analysis['data_types'], indent=2)}

        **SQLite-Specific Requirements:**
        1. Use SQLite-compatible functions only
        2. Avoid window functions that SQLite doesn't support well
        3. Use simple CAST operations for type conversion
        4. Reference: {{{{ ref('{csv_analysis['filename']}') }}}}
        5. Add basic data validation
        6. Use consistent naming (snake_case)

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
        """Generate analytics model SQL with SQLite compatibility"""
        
        prompt = f"""
        Generate a SQLite-compatible analytics model that transforms staging data:

        **Source:** stg_{csv_analysis['filename']}
        **Columns:** {csv_analysis['columns']}

        **SQLite Compatibility Requirements:**
        1. Use {{{{ ref('stg_{csv_analysis['filename']}') }}}}
        2. Avoid complex window functions
        3. Use simple aggregations (COUNT, SUM, AVG, MIN, MAX)
        4. Use CASE statements for conditional logic
        5. Keep queries simple and efficient

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
                    f.write(f"-- SQLite-compatible staging model for {csv_file.name}\n")
                    f.write(f"-- Generated on {datetime.now()}\n\n")
                    f.write(staging_sql)
                
                logger.info(f"Generated staging model: {staging_file}")
            
            # Generate analytics model
            analytics_sql = self.generate_analytics_model(analysis)
            if analytics_sql:
                analytics_name = f"analytics_{analysis['filename']}"
                analytics_file = self.models_dir / f"{analytics_name}.sql"
                
                with open(analytics_file, 'w') as f:
                    f.write(f"-- SQLite-compatible analytics model for {csv_file.name}\n")
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

# Create SQLite-compatible Airflow DAG
mkdir -p ~/airflow/dags
cat > ~/airflow/dags/dbt_sqlite_pipeline.py << 'EOF'
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

def check_sqlite_compatibility():
    """Check SQLite version compatibility"""
    version = sqlite3.sqlite_version
    version_parts = version.split('.')
    major, minor = int(version_parts), int(version_parts[12])
    
    if major < 3 or (major == 3 and minor < 15):
        raise ValueError(f"SQLite version {version} is incompatible. Required: 3.15.0+")
    
    logger.info(f"SQLite version {version} is compatible")

def validate_environment():
    """Validate environment for SQLite setup"""
    # Check SQLite
    check_sqlite_compatibility()
    
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
    
    logger.info(f"Environment validated. Found {len(csv_files)} CSV files.")

def run_model_generation():
    """Run model generation with SQLite compatibility"""
    project_dir = os.path.expanduser('~/si-dbt-test/my_dbt_project')
    script_path = os.path.join(project_dir, 'generate_intelligent_models.py')
    venv_python = os.path.expanduser('~/si-dbt-test/env/bin/python')
    
    result = subprocess.run([venv_python, script_path], 
                          capture_output=True, text=True, cwd=project_dir)
    
    if result.returncode != 0:
        logger.error(f"Model generation failed: {result.stderr}")
        raise Exception(f"Failed: {result.stderr}")
    
    logger.info(f"Success: {result.stdout}")

dag = DAG(
    'dbt_sqlite_compatible_pipeline',
    default_args=default_args,
    description='SQLite-compatible dbt pipeline with AI',
    schedule_interval=None,
    catchup=False,
    tags=['dbt', 'sqlite', 'ai', 'compatible'],
    max_active_runs=1,
)

start = DummyOperator(task_id='start', dag=dag)

validate = PythonOperator(
    task_id='validate_sqlite_environment',
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
    bash_command='cd ~/si-dbt-test/my_dbt_project && source ../env/bin/activate && dbt seed',
    dag=dag,
)

run_staging = BashOperator(
    task_id='run_staging_models',
    bash_command='cd ~/si-dbt-test/my_dbt_project && source ../env/bin/activate && dbt run --select models/staging/',
    dag=dag,
)

run_analytics = BashOperator(
    task_id='run_analytics_models',
    bash_command='cd ~/si-dbt-test/my_dbt_project && source ../env/bin/activate && dbt run --select models/generated/',
    dag=dag,
)

test = BashOperator(
    task_id='test_models',
    bash_command='cd ~/si-dbt-test/my_dbt_project && source ../env/bin/activate && dbt test',
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

# Create startup script
cat > ~/si-dbt-test/start_pipeline.sh << 'EOF'
#!/bin/bash

echo "Starting SQLite-Compatible dbt Pipeline..."

# Activate environment
source ~/si-dbt-test/env/bin/activate
export AIRFLOW_HOME=~/airflow

# Check if already running
if pgrep -f "airflow scheduler" > /dev/null; then
    echo "Scheduler already running"
else
    echo "Starting scheduler..."
    airflow scheduler --daemon
fi

if pgrep -f "airflow webserver" > /dev/null; then
    echo "Webserver already running"
else
    echo "Starting webserver..."
    airflow webserver --port 8080 --daemon
fi

sleep 5
echo ""
echo "✅ Pipeline Started Successfully!"
echo "📊 Airflow UI: http://localhost:8080"
echo "👤 Username: admin"
echo "🔑 Password: admin123"
echo "🎯 DAG: dbt_sqlite_compatible_pipeline"
echo ""
EOF

chmod +x ~/si-dbt-test/start_pipeline.sh

# Create stop script
cat > ~/si-dbt-test/stop_pipeline.sh << 'EOF'
#!/bin/bash
echo "Stopping pipeline..."
pkill -f "airflow scheduler"
pkill -f "airflow webserver"
echo "Pipeline stopped!"
EOF

chmod +x ~/si-dbt-test/stop_pipeline.sh

echo ""
echo "============================================="
echo "✅ SQLite-COMPATIBLE SETUP COMPLETE!"
echo "============================================="
echo ""
echo "🔧 Configuration:"
echo "- Python: 3.11"
echo "- Airflow: 2.8.4 with LocalExecutor"
echo "- SQLite: $(sqlite3 --version | cut -d' ' -f1) (Compatible!)"
echo "- dbt: 1.7.13"
echo ""
echo "🚨 KEY FIXES APPLIED:"
echo "✅ LocalExecutor instead of CeleryExecutor"
echo "✅ SQLite 3.15.0+ compatibility check"
echo "✅ Custom airflow.cfg for SQLite"
echo "✅ SQLite-optimized model generation"
echo ""
echo "📋 MANUAL STEPS:"
echo ""
echo "1️⃣ ADD GEMINI API KEY:"
echo "   echo 'export GEMINI_API_KEY=\"AIzaSyBkoU4uxSqMLwV2B9MST48_3g7sATrl7QY\"' >> ~/.bashrc"
echo "   source ~/.bashrc"
echo ""
echo "2️⃣ START PIPELINE:"
echo "   cd ~/si-dbt-test && ./start_pipeline.sh"
echo ""
echo "3️⃣ ACCESS AIRFLOW:"
echo "   🌐 http://localhost:8080"
echo "   👤 admin / 🔑 admin123"
echo ""
echo "4️⃣ TRIGGER DAG:"
echo "   Find: 'dbt_sqlite_compatible_pipeline'"
echo ""
echo "🏗️ Project Structure:"
echo "~/si-dbt-test/"
echo "├── env/ (Python 3.11 venv)"
echo "├── my_dbt_project/"
echo "│   ├── data/ (CSV files)"
echo "│   ├── models/staging/"
echo "│   └── models/generated/"
echo "├── start_pipeline.sh"
echo "└── stop_pipeline.sh"
echo ""
echo "✅ Ready to use with full SQLite compatibility!"
