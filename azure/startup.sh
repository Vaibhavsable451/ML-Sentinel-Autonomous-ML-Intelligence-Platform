#!/usr/bin/env bash

echo "============================================================"
echo "  ML SENTINEL — Azure App Service Startup Script"
echo "============================================================"

export PYTHONPATH=/home/site/wwwroot
cd /home/site/wwwroot

PYCMD=$(which python3 || which python)

# Ensure virtual environment or pip dependencies are installed
if [ -d "antenv" ]; then
    source antenv/bin/activate
    PYCMD="python"
elif [ -d "venv" ]; then
    source venv/bin/activate
    PYCMD="python"
else
    $PYCMD -m pip install --upgrade pip
    $PYCMD -m pip install --no-cache-dir -r requirements.txt
fi

# Run dataset generator & pipeline if artifacts do not exist
if [ ! -f "artifacts/champion_model.joblib" ]; then
    echo "Initializing datasets & champion model artifacts..."
    PYTHONPATH=. $PYCMD data/generate_data.py
    PYTHONPATH=. $PYCMD scripts/train_and_register.py
fi

# Start FastAPI governance API in background on port 8000
echo "Starting FastAPI backend service on port 8000..."
PYTHONPATH=. $PYCMD -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit dashboard on port 8080 or WEBSITES_PORT / PORT
PORT="${PORT:-${WEBSITES_PORT:-8080}}"
echo "Starting Streamlit dashboard on port ${PORT}..."
exec PYTHONPATH=. $PYCMD -m streamlit run app/streamlit_app.py --server.port "${PORT}" --server.address 0.0.0.0 --server.headless true


