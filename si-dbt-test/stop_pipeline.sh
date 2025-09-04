#!/bin/bash

echo "Stopping SI-dbt Pipeline..."

# Stop Airflow processes
pkill -f "airflow scheduler"
pkill -f "airflow webserver"

echo "Pipeline stopped successfully!"
