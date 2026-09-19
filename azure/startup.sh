#!/usr/bin/env bash

echo "============================================================"
echo "  ML SENTINEL — Azure App Service Startup Script"
echo "============================================================"

export PYTHONPATH=/home/site/wwwroot
cd /home/site/wwwroot

PYCMD=$(which python3 || which python)

# Activate virtual environment if available (created by SCM_DO_BUILD_DURING_DEPLOYMENT)
if [ -d "antenv" ]; then
    source antenv/bin/activate
    PYCMD="python"
elif [ -d "venv" ]; then
    source venv/bin/activate
    PYCMD="python"
fi

# Run data generation & training in BACKGROUND — do NOT block startup
# Services start immediately; artifacts will be ready within ~2 minutes
if [ ! -f "artifacts/champion_model.joblib" ]; then
    echo "[background] Generating dataset and training champion model..."
    (
        PYTHONPATH=. $PYCMD data/generate_data.py && \
        PYTHONPATH=. $PYCMD scripts/train_and_register.py && \
        echo "[background] Champion model ready."
    ) &
else
    echo "Champion model artifacts already exist — skipping training."
fi

# Start FastAPI governance API in background on port 8000
echo "Starting FastAPI backend on port 8000..."
PYTHONPATH=. $PYCMD -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit dashboard on WEBSITES_PORT (Azure routes 80/443 → this port)
PORT="${PORT:-${WEBSITES_PORT:-8080}}"
echo "Starting Streamlit dashboard on port ${PORT}..."
exec PYTHONPATH=. $PYCMD -m streamlit run app/streamlit_app.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true
