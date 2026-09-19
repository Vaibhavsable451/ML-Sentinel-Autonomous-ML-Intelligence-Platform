#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  ML SENTINEL — Azure App Service Startup Script"
echo "============================================================"

# Ensure PYTHONPATH is set to application root
export PYTHONPATH=/home/site/wwwroot

cd /home/site/wwwroot

# Quick check if streamlit is installed before running pip
if ! python -c "import streamlit" &>/dev/null; then
    echo "Installing requirements..."
    pip install --no-cache-dir -r requirements.txt
fi

# Run dataset generator & pipeline if artifacts do not exist
if [ ! -f "artifacts/champion_model.joblib" ]; then
    echo "Initializing datasets & champion model artifacts..."
    PYTHONPATH=. python data/generate_data.py
    PYTHONPATH=. python scripts/train_and_register.py
fi

# Start FastAPI governance API in background on port 8000
echo "Starting FastAPI backend service on port 8000..."
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit dashboard on port 8080 or WEBSITES_PORT / PORT
PORT="${PORT:-${WEBSITES_PORT:-8080}}"
echo "Starting Streamlit dashboard on port ${PORT}..."
exec PYTHONPATH=. streamlit run app/streamlit_app.py --server.port "${PORT}" --server.address 0.0.0.0 --server.headless true

