"""Deploy ML Sentinel to Microsoft Azure App Service.

Usage:
    python3 azure/deploy_azure.py \
        --resource-group ml-sentinel-rg \
        --app-name ml-sentinel-app \
        --location eastus
"""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Deploy ML Sentinel to Azure App Service")
    p.add_argument("--resource-group", default="ml-sentinel-rg", help="Azure Resource Group")
    p.add_argument("--app-name", required=True, help="Globally unique Azure Web App name")
    p.add_argument("--location", default="eastus", help="Azure Region (default: eastus)")
    p.add_argument("--sku", default="B1", help="App Service Plan SKU (default: B1)")
    return p.parse_args()


def run_cmd(cmd):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: {res.stderr}")
        sys.exit(res.returncode)
    print(res.stdout)
    return res.stdout.strip()


def create_zip():
    zip_path = Path("deploy.zip")
    print("Building application deployment zip archive...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in Path(".").rglob("*"):
            if any(part.startswith(".") or part in ("venv", "__pycache__", "artifacts", "deploy.zip") for part in file.parts):
                continue
            if file.is_file():
                zipf.write(file, file.relative_to("."))
    print(f"Zip package created: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    return zip_path


def main():
    args = parse_args()
    plan_name = f"{args.app_name}-plan"

    print("=== Deploying ML Sentinel to Microsoft Azure App Service ===")
    
    # Create Resource Group
    run_cmd(f"az group create --name {args.resource_group} --location {args.location}")

    # Create App Service Plan
    run_cmd(f"az appservice plan create --name {plan_name} --resource-group {args.resource_group} --sku {args.sku} --is-linux")

    # Create Web App
    run_cmd(f"az webapp create --name {args.app_name} --resource-group {args.resource_group} --plan {plan_name} --runtime 'PYTHON|3.11' --startup-file 'bash azure/startup.sh'")

    # Configure App Settings
    run_cmd(f"az webapp config appsettings set --name {args.app_name} --resource-group {args.resource_group} --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true PYTHONPATH=/home/site/wwwroot WEBSITES_PORT=8080")

    # Package & Deploy
    zip_file = create_zip()
    run_cmd(f"az webapp deployment source config-zip --resource-group {args.resource_group} --name {args.app_name} --src {zip_file}")

    zip_file.unlink()

    print("\n============================================================")
    print("  🎉 Azure App Service Deployment Successful!")
    print(f"  • App URL:   https://{args.app_name}.azurewebsites.net")
    print(f"  • Resource Group: {args.resource_group}")
    print("============================================================")


if __name__ == "__main__":
    main()
