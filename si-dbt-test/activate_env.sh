#!/bin/bash
source ~/si-dbt-test/env/bin/activate
export AIRFLOW_HOME=~/airflow
cd ~/si-dbt-test/my_dbt_project
echo "Environment activated! Python version: $(python --version)"
echo "Current directory: $(pwd)"
