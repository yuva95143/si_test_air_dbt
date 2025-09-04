#!/bin/bash
set -e

echo "=== Step 1: Update system and install dependencies ==="
sudo apt update
sudo apt install -y software-properties-common curl wget unzip gnupg lsb-release build-essential

echo "=== Step 2: Add deadsnakes PPA for newer Python versions ==="
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

echo "=== Step 3: Install Python 3.12 and required packages ==="
sudo apt install -y python3.12 python3.12-venv python3.12-dev

echo "=== Step 4: Create Python virtual environment ==="
if [ ! -d "airflow-venv" ]; then
  python3.12 -m venv airflow-venv
fi

echo "=== Step 5: Activate virtual environment ==="
source airflow-venv/bin/activate

echo "=== Step 6: Upgrade pip and wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== Step 7: Install Apache Airflow ==="
AIRFLOW_VERSION=2.8.2
PYTHON_VERSION=3.12
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

echo "=== Step 8: Initialize Airflow database ==="
export AIRFLOW_HOME=~/airflow
airflow db init

echo "=== Step 9: Create default Airflow user ==="
airflow users create \
    --username admin \
    --firstname First \
    --lastname Last \
    --role Admin \
    --email admin@example.com \
    --password admin

echo "=== Setup complete! ==="
echo "To start Airflow Webserver and Scheduler, run:"
echo "source airflow-venv/bin/activate"
echo "airflow webserver -p 8080 &"
echo "airflow scheduler &"
