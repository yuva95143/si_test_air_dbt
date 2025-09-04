#!/bin/bash
source ~/si-dbt-test/env/bin/activate
export AIRFLOW_HOME=~/airflow

echo "Starting Airflow..."
airflow scheduler --daemon
airflow webserver --port 8080 --daemon

echo "Services started!"
echo "Airflow UI: http://localhost:8080"
echo "Username: admin | Password: admin123"
