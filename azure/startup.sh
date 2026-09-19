#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  ML SENTINEL — Azure App Service Startup Script"
echo "============================================================"

# Ensure PYTHONPATH is set to application root
export PYTHONPATH=/home/site/wwwroot

cd /home/site/wwwroot

# Install dependencies if needed
pip install -r requirements.txt

# Run dataset generator & pipeline if artifacts do not exist
if [ ! -f "artifacts/champion_model.joblib" ]; then
    echo "Initializing datasets & champion model artifacts..."
    PYTHONPATH=. python data/generate_data.py
    PYTHONPATH=. python scripts/train_and_register.py
fi

# Start FastAPI governance API in background on port 8000
echo "Starting FastAPI backend service on port 8000..."
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit dashboard on port 8080 (default Azure Web App PORT) or 8501
PORT="${PORT:-8080}"
echo "Starting Streamlit dashboard on port ${PORT}..."
exec PYTHONPATH=. streamlit run app/streamlit_app.py --server.port "${PORT}" --server.address 0.0.0.0 --server.headless true
