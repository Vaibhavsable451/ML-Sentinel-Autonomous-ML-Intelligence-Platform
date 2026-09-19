#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  ML SENTINEL — AWS EC2 Automated Provisioning & Setup"
echo "============================================================"

# Update OS packages
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git nginx curl

APP_DIR="/opt/ml-sentinel"
sudo mkdir -p $APP_DIR
sudo chown -R $USER:$USER $APP_DIR

# Copy or clone repo
if [ ! -f "$APP_DIR/requirements.txt" ]; then
    echo "Cloning repository..."
    git clone https://github.com/vibhav/ml-sentinel.git $APP_DIR || true
fi

cd $APP_DIR

# Create Virtual Environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run synthetic dataset generator & model training pipeline if artifacts are missing
if [ ! -f "artifacts/champion_model.joblib" ]; then
    echo "Training model pipeline & creating artifacts..."
    PYTHONPATH=. python3 data/generate_data.py
    PYTHONPATH=. python3 scripts/train_and_register.py
fi

# Configure Systemd Services
echo "Configuring Systemd services..."
sudo cp ec2/ml-sentinel-api.service /etc/systemd/system/
sudo cp ec2/ml-sentinel-app.service /etc/systemd/system/

# Configure Nginx Reverse Proxy
echo "Configuring Nginx reverse proxy..."
sudo cp ec2/nginx.conf /etc/nginx/sites-available/ml-sentinel
sudo ln -sf /etc/nginx/sites-available/ml-sentinel /etc/nginx/sites-enabled/default

# Enable and Start Services
sudo systemctl daemon-reload
sudo systemctl enable ml-sentinel-api ml-sentinel-app nginx
sudo systemctl restart ml-sentinel-api ml-sentinel-app nginx

PUBLIC_IP=$(curl -s http://checkip.amazonaws.com || echo "localhost")

echo "============================================================"
echo "  ✓ ML SENTINEL EC2 DEPLOYMENT COMPLETE!"
echo "  • Nginx Proxy Gateway:  http://${PUBLIC_IP}"
echo "  • Streamlit Dashboard:  http://${PUBLIC_IP}:8501"
echo "  • FastAPI API Docs:     http://${PUBLIC_IP}:8000/docs"
echo "============================================================"
