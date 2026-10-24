> **Vigencia documental — 18/10/2026:** fuente histórica o de alcance parcial. Sus planes, comandos y afirmaciones no definen por sí solos el sistema actual. Consultar [contraste/índice](auditoria/CONTEXTO_DOCUMENTATION.md) y [estado vigente](auditoria/00_ESTADO.md) antes de proponer cambios. El contenido original se conserva debajo.

# Deployment Guide

## Overview
This guide provides instructions for deploying the Basketball O/U Prediction System in various environments.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM, recommended 8GB+
- **Storage**: 2GB free space for models and data
- **Network**: Stable internet connection for API access

### Dependencies
```bash
# Core dependencies
pip install pandas>=1.5.0 scikit-learn>=1.2.0 xgboost>=1.7.0 scipy>=1.10.0

# Web scraping
pip install playwright>=1.30.0

# Optional: Async support
pip install aiohttp>=3.8.0

# Optional: Monitoring
pip install prometheus-client>=0.16.0
```

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd basketball-ou-system
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Playwright Browsers
```bash
playwright install
```

### 5. Verify Installation
```bash
python -c "import pandas, sklearn, xgboost; print('Dependencies OK')"
python -c "from core.live_api import SofaScoreClient; print('System OK')"
```

## Configuration

### Environment Variables
```bash
# Create .env file
SOFASCORE_API_TIMEOUT=30
SOFASCORE_RATE_LIMIT=2.0
CACHE_DURATION=30
LOG_LEVEL=INFO
```

### League Data Setup
```bash
# Ensure league data files exist
ls leagues/
# Expected: NBA.json, WNBA.json, NBL.json
```

### Model Training
```bash
# Train models for each league
python -c "from core.training import train_league_model; train_league_model('NBA')"
python -c "from core.training import train_league_model; train_league_model('WNBA')"
python -c "from core.training import train_league_model; train_league_model('NBL')"
```

## Deployment Options

### Option 1: Local Development

#### Single Game Analysis
```bash
# Run interactive prediction
python analysis/live_system.py
```

#### Batch Processing
```bash
# Process multiple games
python analysis/live_system.py --batch --league NBA
```

#### Live Monitoring
```bash
# Monitor live games
python analysis/live_system.py --live --interval 60
```

### Option 2: Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Install Playwright
RUN playwright install-deps && playwright install chromium

EXPOSE 8000

CMD ["python", "analysis/live_system.py", "--serve"]
```

#### Docker Compose
```yaml
version: '3.8'
services:
  basketball-ou:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SOFASCORE_API_TIMEOUT=30
      - CACHE_DURATION=30
    volumes:
      - ./models:/app/models
      - ./leagues:/app/leagues
```

#### Build and Run
```bash
# Build image
docker build -t basketball-ou .

# Run container
docker run -p 8000:8000 basketball-ou

# Or use docker-compose
docker-compose up -d
```

### Option 3: Cloud Deployment

#### AWS EC2
```bash
# Provision EC2 instance
aws ec2 run-instances \
  --image-id ami-12345678 \
  --instance-type t3.medium \
  --key-name your-key-pair

# Install dependencies on instance
sudo yum update -y
sudo yum install python3 python3-pip -y

# Deploy application
git clone <repository-url>
cd basketball-ou-system
pip3 install -r requirements.txt
playwright install

# Run as service
sudo cp deployment/basketball-ou.service /etc/systemd/system/
sudo systemctl enable basketball-ou
sudo systemctl start basketball-ou
```

#### Google Cloud Run
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/basketball-ou', '.']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/basketball-ou']

  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'basketball-ou'
      - '--image'
      - 'gcr.io/$PROJECT_ID/basketball-ou'
      - '--platform'
      - 'managed'
      - '--port'
      - '8000'
```

### Option 4: Kubernetes Deployment

#### Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: basketball-ou
spec:
  replicas: 2
  selector:
    matchLabels:
      app: basketball-ou
  template:
    metadata:
      labels:
        app: basketball-ou
    spec:
      containers:
      - name: basketball-ou
        image: basketball-ou:latest
        ports:
        - containerPort: 8000
        env:
        - name: SOFASCORE_API_TIMEOUT
          value: "30"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

#### Service Manifest
```yaml
apiVersion: v1
kind: Service
metadata:
  name: basketball-ou-service
spec:
  selector:
    app: basketball-ou
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Monitoring and Maintenance

### Health Checks
```bash
# API health check
curl http://localhost:8000/health

# Model performance check
python -c "from core.training import validate_models; validate_models()"
```

### Logging
```python
# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/basketball_ou.log'),
        logging.StreamHandler()
    ]
)
```

### Automated Tasks

#### Cron Jobs for Model Retraining
```bash
# Weekly model retraining
0 2 * * 1 python -c "from core.training import retrain_all_models; retrain_all_models()"

# Daily data backup
0 1 * * * tar -czf backup_$(date +\%Y\%m\%d).tar.gz models/ leagues/
```

#### Log Rotation
```bash
# Configure logrotate
cat > /etc/logrotate.d/basketball-ou << EOF
/app/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF
```

## Performance Optimization

### Caching Strategy
```python
# Configure Redis cache (optional)
import redis

cache = redis.Redis(host='localhost', port=6379, db=0)

# Cache API responses
@cache.cached(timeout=30)
def get_live_games():
    return client.get_live_basketball_games()
```

### Database Integration (Optional)
```python
# PostgreSQL for historical data
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="basketball_ou",
    user="ou_user",
    password="secure_password"
)

# Store predictions and performance metrics
def store_prediction(prediction_data):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO predictions (game_id, prediction, actual, accuracy)
            VALUES (%s, %s, %s, %s)
        """, (prediction_data['game_id'], prediction_data['prediction'],
              prediction_data['actual'], prediction_data['accuracy']))
    conn.commit()
```

## Security Considerations

### API Key Management
```python
# Use environment variables for sensitive data
import os
api_key = os.getenv('SOFASCORE_API_KEY')

# Rotate API keys regularly
# Implement rate limiting
```

### Network Security
```bash
# Configure firewall
sudo ufw allow 8000/tcp
sudo ufw enable

# Use HTTPS in production
# Implement API authentication
```

## Troubleshooting

### Common Issues

#### API Connection Problems
```bash
# Check network connectivity
curl -I https://www.sofascore.com

# Verify Playwright installation
playwright --version

# Test API client
python -c "from core.live_api import SofaScoreClient; client = SofaScoreClient(); print(client.health_check())"
```

#### Memory Issues
```bash
# Monitor memory usage
ps aux --sort=-%mem | head

# Adjust Python memory limits
export PYTHONMALLOC=jemalloc

# Use smaller batch sizes for processing
```

#### Model Loading Errors
```bash
# Verify model files exist
ls -la models/

# Check model file integrity
python -c "import joblib; model = joblib.load('models/NBA.joblib'); print('Model loaded successfully')"

# Retrain models if corrupted
python -c "from core.training import train_league_model; train_league_model('NBA')"
```

### Performance Monitoring
```python
# Monitor prediction latency
import time

start_time = time.time()
prediction = get_live_prediction(team_data)
latency = time.time() - start_time

if latency > 5.0:
    logging.warning(f"High latency: {latency:.2f}s")
```

## Backup and Recovery

### Data Backup
```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup models and data
cp -r models/ $BACKUP_DIR/
cp -r leagues/ $BACKUP_DIR/
cp config.py $BACKUP_DIR/

# Compress backup
tar -czf ${BACKUP_DIR}.tar.gz $BACKUP_DIR
```

### Disaster Recovery
```bash
# Restore from backup
tar -xzf backup_20231201.tar.gz
cp -r backup_20231201/models/* models/
cp -r backup_20231201/leagues/* leagues/

# Verify system integrity
python -c "from core.training import validate_models; validate_models()"
```

## Scaling Considerations

### Horizontal Scaling
```python
# Implement load balancing
# Use multiple instances behind nginx
# Implement session affinity for live monitoring
```

### Vertical Scaling
```python
# Increase instance size for higher throughput
# Add more memory for larger models
# Use SSD storage for faster model loading
```

This deployment guide ensures reliable and scalable operation of the Basketball O/U Prediction System across different environments.