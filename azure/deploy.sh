#!/usr/bin/env bash
set -e

# Deploy ML Sentinel to Azure App Service using Azure CLI
# Usage: ./azure/deploy.sh <RESOURCE_GROUP> <APP_NAME> <LOCATION>

RESOURCE_GROUP="${1:-ml-sentinel-rg}"
APP_NAME="${2:-ml-sentinel-app-$RANDOM}"
LOCATION="${3:-eastus}"
PLAN_NAME="${APP_NAME}-plan"

echo "============================================================"
echo "  Deploying ML Sentinel to Microsoft Azure App Service"
echo "  Resource Group: ${RESOURCE_GROUP}"
echo "  App Name:       ${APP_NAME}"
echo "  Location:       ${LOCATION}"
echo "============================================================"

# 1. Create Resource Group
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

# 2. Create App Service Plan (B1 Basic Linux)
az appservice plan create \
    --name "$PLAN_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --sku B1 \
    --is-linux

# 3. Create Web App with Python 3.11 runtime
az webapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --plan "$PLAN_NAME" \
    --runtime "PYTHON|3.11" \
    --startup-file "bash azure/startup.sh"

# 4. Configure App Settings
az webapp config appsettings set \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --settings \
        SCM_DO_BUILD_DURING_DEPLOYMENT=true \
        PYTHONPATH=/home/site/wwwroot \
        WEBSITES_PORT=8080

# 5. Deploy Zip Package
echo "Zipping and deploying code package..."
zip -r deploy.zip . -x "*.git*" "venv/*" "__pycache__/*"

az webapp deployment source config-zip \
    --resource-group "$RESOURCE_GROUP" \
    --name "$APP_NAME" \
    --src deploy.zip

rm -f deploy.zip

echo "============================================================"
echo "  🎉 Azure App Service Deployment Complete!"
echo "  URL: https://${APP_NAME}.azurewebsites.net"
echo "============================================================"
