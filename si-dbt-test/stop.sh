#!/bin/bash
echo "Stopping Airflow..."
pkill -f "airflow scheduler"
pkill -f "airflow webserver"
echo "Stopped!"
