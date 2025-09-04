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
