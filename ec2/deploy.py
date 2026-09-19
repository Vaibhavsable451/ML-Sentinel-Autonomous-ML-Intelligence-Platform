"""Deploy ML Sentinel to AWS EC2 instance using boto3.

Usage (from your local machine or CI, with AWS credentials configured):

    python3 ec2/deploy.py \
        --key-name my-ec2-key \
        --region us-east-1 \
        --instance-type m7i-flex.large
"""
import argparse

import boto3


def parse_args():
    p = argparse.ArgumentParser(description="Deploy ML Sentinel to an AWS EC2 instance")
    p.add_argument("--key-name", required=True, help="AWS EC2 Key Pair name for SSH access")
    p.add_argument("--region", default="us-east-1", help="AWS Region (default: us-east-1)")
    p.add_argument("--instance-type", default="m7i-flex.large", help="EC2 Instance type (default: m7i-flex.large)")
    p.add_argument("--ami-id", default="", help="Optional Ubuntu AMI ID (auto-detects Ubuntu 22.04 LTS if empty)")
    p.add_argument("--security-group", default="ml-sentinel-sg", help="Security Group name")
    return p.parse_args()


USER_DATA_SCRIPT = """#!/bin/bash
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-venv git nginx curl

APP_DIR="/opt/ml-sentinel"
sudo mkdir -p $APP_DIR
sudo chown -R ubuntu:ubuntu $APP_DIR

cd /opt
git clone https://github.com/vibhav/ml-sentinel.git $APP_DIR || true
cd $APP_DIR

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

PYTHONPATH=. python3 data/generate_data.py
PYTHONPATH=. python3 scripts/train_and_register.py

sudo cp ec2/ml-sentinel-api.service /etc/systemd/system/
sudo cp ec2/ml-sentinel-app.service /etc/systemd/system/
sudo cp ec2/nginx.conf /etc/nginx/sites-available/ml-sentinel
sudo ln -sf /etc/nginx/sites-available/ml-sentinel /etc/nginx/sites-enabled/default

sudo systemctl daemon-reload
sudo systemctl enable ml-sentinel-api ml-sentinel-app nginx
sudo systemctl restart ml-sentinel-api ml-sentinel-app nginx
"""


def get_ubuntu_ami(ec2_client):
    """Find the latest Ubuntu 22.04 LTS AMI in the target region."""
    resp = ec2_client.describe_images(
        Owners=["099720109477"],  # Canonical
        Filters=[
            {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
            {"Name": "state", "Values": ["available"]},
        ],
    )
    images = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
    return images[0]["ImageId"]


def create_security_group(ec2_resource, ec2_client, sg_name):
    """Create or retrieve Security Group opening ports 22, 80, 443, 8000, 8501."""
    try:
        vpcs = list(ec2_resource.vpcs.all())
        vpc_id = vpcs[0].id
        sg = ec2_resource.create_security_group(
            GroupName=sg_name,
            Description="Security Group for ML Sentinel EC2 deployment",
            VpcId=vpc_id,
        )
        print(f"Created Security Group: {sg.id}")
        permissions = [
            {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            {"IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            {"IpProtocol": "tcp", "FromPort": 8000, "ToPort": 8000, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
            {"IpProtocol": "tcp", "FromPort": 8501, "ToPort": 8501, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
        ]
        sg.authorize_ingress(IpPermissions=permissions)
        return sg.id
    except Exception as e:
        if "already exists" in str(e):
            sgs = ec2_client.describe_security_groups(GroupNames=[sg_name])
            return sgs["SecurityGroups"][0]["GroupId"]
        raise


def main():
    args = parse_args()
    args.key_name = args.key_name.removesuffix(".pem")
    session = boto3.Session(region_name=args.region)
    ec2_client = session.client("ec2")
    ec2_resource = session.resource("ec2")

    ami_id = args.ami_id or get_ubuntu_ami(ec2_client)
    sg_id = create_security_group(ec2_resource, ec2_client, args.security_group)

    print(f"Launching EC2 instance using AMI {ami_id} ({args.instance_type})...")
    instances = ec2_resource.create_instances(
        ImageId=ami_id,
        InstanceType=args.instance_type,
        KeyName=args.key_name,
        SecurityGroupIds=[sg_id],
        MinCount=1,
        MaxCount=1,
        UserData=USER_DATA_SCRIPT,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": "ML-Sentinel-Platform"}],
        }],
    )
    instance = instances[0]
    print(f"Instance ID: {instance.id}. Waiting for running state...")
    instance.wait_until_running()
    instance.reload()

    public_ip = instance.public_ip_address
    public_dns = instance.public_dns_name

    print("\n🎉 AWS EC2 Instance Successfully Launched!")
    print(f"  • Instance ID: {instance.id}")
    print(f"  • Public IP:   {public_ip}")
    print(f"  • Public DNS:  {public_dns}")
    print("\nLive Endpoints (available once user-data script completes in ~2 mins):")
    print(f"  • Nginx Web Gateway:  http://{public_ip}")
    print(f"  • Streamlit Dashboard:  http://{public_ip}:8501")
    print(f"  • FastAPI Backend:    http://{public_ip}:8000/docs")
    print(f"\nSSH Access: ssh -i {args.key_name}.pem ubuntu@{public_ip}")


if __name__ == "__main__":
    main()
