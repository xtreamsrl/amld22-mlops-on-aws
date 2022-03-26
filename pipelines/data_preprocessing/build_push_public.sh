#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.
# Here we use public repo

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
region=$2

if [ "$image" == "" ] || [ "$region" == "" ]
then
    echo "Usage: $0 <image-name> <region>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# If the repository doesn't exist in ECR, create it.
repo_uri=$(aws ecr-public describe-repositories --repository-names "${image}" --region us-east-1 --query repositories[0].repositoryUri --output text)
if [ $? -ne 0 ]
then
    echo "Creating repository ${image}"
    aws ecr-public create-repository --repository-name "${image}" --region $region > /dev/null
    repo_uri=$(aws ecr-public describe-repositories --repository-names "${image}" --region us-east-1 --query repositories[0].repositoryUri --output text)
else
    echo "Repository ${image} already exists"
fi

fullname="${repo_uri}:latest"

# Get the login command from ECR and execute it directly
echo $repo_uri
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin $repo_uri

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

echo "Building docker image..."
docker build  -t ${image} .
docker tag ${image} ${fullname}

echo "Pushing docker image..."
docker push ${fullname}