# AWS Deployment Instructions

## Overview
This guide provides step-by-step instructions for deploying the Personalized Healthcare Recommendation System on AWS.

## Deployment Options

### Option 1: AWS EC2 Deployment

#### Prerequisites
- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Docker installed locally (for building images)

#### Step 1: Launch EC2 Instance
```bash
# Launch Ubuntu 20.04 LTS instance
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --user-data file://user-data.sh
```

#### Step 2: Security Group Configuration
```bash
# Allow HTTP traffic on ports 8000 and 8501
aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxxxxxx \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id sg-xxxxxxxxx \
    --protocol tcp \
    --port 8501 \
    --cidr 0.0.0.0/0
```

#### Step 3: Connect and Deploy
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone repository
git clone https://github.com/your-repo/healthcare-recommendation.git
cd healthcare-recommendation

# Install Docker
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ubuntu

# Build and run
docker-compose -f deployment/docker-compose.yml up -d
```

### Option 2: AWS Lambda + API Gateway (Serverless Backend)

#### Step 1: Prepare Lambda Package
```bash
# Create deployment package
mkdir lambda-package
cp -r backend/* lambda-package/
cd lambda-package

# Install dependencies
pip install -r requirements.txt -t .

# Create Lambda handler
cat > lambda_handler.py << 'EOF'
from mangum import Mangum
from app import app

handler = Mangum(app)
EOF

# Package for deployment
zip -r ../healthcare-lambda.zip .
```

#### Step 2: Deploy Lambda Function
```bash
# Create Lambda function
aws lambda create-function \
    --function-name healthcare-prediction \
    --runtime python3.9 \
    --role arn:aws:iam::account:role/lambda-execution-role \
    --handler lambda_handler.handler \
    --zip-file fileb://healthcare-lambda.zip \
    --timeout 30 \
    --memory-size 512
```

#### Step 3: Create API Gateway
```bash
# Create REST API
aws apigateway create-rest-api \
    --name healthcare-api \
    --description "Healthcare Recommendation API"

# Configure API Gateway integration with Lambda
# (Additional configuration steps required)
```

### Option 3: AWS ECS with Fargate

#### Step 1: Create ECR Repository
```bash
# Create ECR repository
aws ecr create-repository --repository-name healthcare-app

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin account.dkr.ecr.us-east-1.amazonaws.com
```

#### Step 2: Build and Push Docker Image
```bash
# Build image
docker build -f deployment/Dockerfile -t healthcare-app .

# Tag image
docker tag healthcare-app:latest account.dkr.ecr.us-east-1.amazonaws.com/healthcare-app:latest

# Push image
docker push account.dkr.ecr.us-east-1.amazonaws.com/healthcare-app:latest
```

#### Step 3: Create ECS Task Definition
```json
{
  "family": "healthcare-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "healthcare-container",
      "image": "account.dkr.ecr.us-east-1.amazonaws.com/healthcare-app:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        },
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/healthcare-app",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Step 4: Create ECS Service
```bash
# Create cluster
aws ecs create-cluster --cluster-name healthcare-cluster

# Create service
aws ecs create-service \
    --cluster healthcare-cluster \
    --service-name healthcare-service \
    --task-definition healthcare-app:1 \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxxxxxx],securityGroups=[sg-xxxxxxxxx],assignPublicIp=ENABLED}"
```

## Environment Variables

Set the following environment variables for production:

```bash
# Backend Configuration
export PYTHONPATH=/app
export MODEL_PATH=/app/backend/models/risk_model.pkl

# Database Configuration (if using external DB)
export DATABASE_URL=postgresql://user:pass@host:port/db

# Security
export SECRET_KEY=your-secret-key
export ALLOWED_HOSTS=your-domain.com

# AWS Configuration
export AWS_DEFAULT_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key
```

## Monitoring and Logging

### CloudWatch Setup
```bash
# Create log group
aws logs create-log-group --log-group-name /aws/ecs/healthcare-app

# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
    --dashboard-name healthcare-monitoring \
    --dashboard-body file://cloudwatch-dashboard.json
```

### Health Checks
- Backend health check: `GET /health`
- Frontend health check: Access Streamlit interface
- Model health check: `GET /api/model-info`

## Security Considerations

1. **Network Security**
   - Use VPC with private subnets
   - Configure security groups with minimal required access
   - Use Application Load Balancer with SSL/TLS

2. **Data Security**
   - Encrypt data at rest and in transit
   - Use AWS Secrets Manager for sensitive configuration
   - Implement proper IAM roles and policies

3. **Application Security**
   - Enable CORS with specific origins
   - Implement rate limiting
   - Use HTTPS only in production

## Cost Optimization

1. **EC2 Instances**
   - Use Reserved Instances for predictable workloads
   - Consider Spot Instances for development

2. **Lambda**
   - Optimize memory allocation
   - Use provisioned concurrency for consistent performance

3. **ECS Fargate**
   - Right-size CPU and memory
   - Use auto-scaling based on metrics

## Troubleshooting

### Common Issues

1. **Port Access Issues**
   ```bash
   # Check security group rules
   aws ec2 describe-security-groups --group-ids sg-xxxxxxxxx
   ```

2. **Container Startup Issues**
   ```bash
   # Check ECS logs
   aws logs get-log-events --log-group-name /ecs/healthcare-app --log-stream-name ecs/healthcare-container/task-id
   ```

3. **Model Loading Issues**
   ```bash
   # Verify model file exists
   docker exec -it container-id ls -la /app/backend/models/
   ```

## Scaling Considerations

- **Horizontal Scaling**: Use ECS auto-scaling or Lambda concurrency
- **Vertical Scaling**: Increase instance size or container resources
- **Database Scaling**: Consider RDS with read replicas
- **Caching**: Implement Redis for model predictions

## Backup and Recovery

1. **Model Backup**
   - Store trained models in S3
   - Version control model artifacts

2. **Data Backup**
   - Regular database backups
   - Cross-region replication for disaster recovery

3. **Configuration Backup**
   - Store infrastructure as code (Terraform/CloudFormation)
   - Version control deployment scripts
