# ML Sentinel — Autonomous ML Intelligence Platform

A production-style ML platform for a customer-churn use case: it doesn't just
predict — it profiles data quality, benchmarks algorithms, scores model risk,
detects drift, explains predictions, checks fairness, and decides when to
retrain, with a human always in the loop before anything reaches production.

## Multi-Cloud Deployment Options

ML Sentinel supports production cloud deployments on **AWS EC2** and **Microsoft Azure App Service**:
1. **AWS EC2** (`ec2/`): Automated EC2 provisioning script (`ec2/deploy.py`), Systemd service management, Nginx reverse proxy gateway, and Docker Compose stack.
2. **Microsoft Azure App Service** (`azure/`): Azure Web App deployment scripts (`azure/deploy_azure.py` and `azure/deploy.sh`), Oryx Python startup handler (`azure/startup.sh`), and GitHub Actions workflow (`.github/workflows/azure_deployment.yml`).

## Architecture

```
Jupyter/EDA → Preprocessing → Feature Engineering → Model Benchmark (9 algos)
    → Champion Model → MLflow Tracking → Model Registry → Risk Engine
    → Cloud Deployment (AWS EC2 / Azure App Service) → FastAPI → Streamlit Dashboard
    → Monitoring (Drift + Risk) → Autonomous Retraining → Human Approval → Deploy
```

## Repo Layout

```
ml-sentinel/
├── data/                    synthetic churn dataset generator + generated CSVs
├── ml/
│   ├── preprocessing/       data quality profiling + cleaning/encoding pipeline
│   ├── features/            feature engineering + selection (MI, RFE, correlation)
│   ├── models/              9-algorithm benchmark, registry, champion/challenger, retraining
│   ├── evaluation/          metrics + calibration
│   ├── explainability/      SHAP global/local importance + counterfactuals
│   ├── drift/               PSI, KS, Jensen-Shannon, Wasserstein, schema/quality drift
│   ├── risk/                0-100 risk engine + security (PII, schema validation, checksums)
│   └── fairness/            demographic parity, equal opportunity, equalized odds
├── api/                     FastAPI service (prediction + governance endpoints, JWT/RBAC, rate limiting)
├── app/                     Streamlit dark command-center dashboard
├── ec2/                     AWS EC2 deployment script, systemd units, Nginx config, Dockerfile
├── azure/                   Microsoft Azure App Service deployment scripts, startup.sh, ARM config
├── notebooks/               Jupyter pipeline walkthrough notebook
├── scripts/                 train_and_register.py — the one-command pipeline runner
├── tests/                   pytest suite covering every core module
├── configs/                 IAM policies and Azure deployment role JSON
└── .github/workflows/       CI, scheduled training, EC2 deployment, Azure deployment
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset (train_raw.csv + production_traffic.csv)
PYTHONPATH=. python3 data/generate_data.py

# 2. Run the full pipeline: preprocess → benchmark 9 models → evaluate →
#    risk/drift/fairness → register champion → write artifacts/
PYTHONPATH=. python3 scripts/train_and_register.py

# 3. Serve predictions via FastAPI
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
# test: curl http://localhost:8000/governance/risk

# 4. Launch the Streamlit dashboard
PYTHONPATH=. streamlit run app/streamlit_app.py

# 5. Run tests
PYTHONPATH=. pytest tests/ -v
```

## 1. Deploying to AWS EC2

Deploy ML Sentinel to a dedicated AWS EC2 instance (Ubuntu 22.04 LTS):

```bash
python3 ec2/deploy.py \
    --key-name your-ec2-key-pair \
    --region us-east-1 \
    --instance-type m7i-flex.large
```

**What it provisions:**
- Creates an EC2 instance with security group ports open (`22`, `80`, `443`, `8000`, `8501`).
- Runs `ec2/setup_ec2.sh` to install Python 3.11, Nginx, and Systemd services (`ml-sentinel-api` and `ml-sentinel-app`).
- Configures Nginx reverse proxy mapping `/` to Streamlit (`8501`) and `/api/` to FastAPI (`8000`).

---

## 2. Deploying to Microsoft Azure App Service

Deploy ML Sentinel to a Microsoft Azure App Service (Linux Web App):

```bash
# Option A: Python deploy script
python3 azure/deploy_azure.py \
    --resource-group ml-sentinel-rg \
    --app-name ml-sentinel-app \
    --location eastus

# Option B: Shell script via Azure CLI
./azure/deploy.sh ml-sentinel-rg ml-sentinel-app eastus
```

**What it provisions:**
- Creates Azure Resource Group and Linux B1/P1v2 App Service Plan.
- Configures Python 3.11 runtime stack and sets startup script to `bash azure/startup.sh`.
- Bundles application code into deployment zip and provisions live `.azurewebsites.net` web app.

---

## Governance Model

- Every retrain produces a **candidate**, never a champion, until it passes the promotion gate in `ml/models/registry.py::evaluate_promotion` (must meet/beat the current champion's F1 and stay under the risk threshold).
- `ml/models/retraining.py` never auto-promotes — `requires_human_approval` is always `True` in its output.
- Cloud deployments enforce human-in-the-loop review before production candidate promotion.
