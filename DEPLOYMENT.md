# Deployment Guide

This guide covers different deployment options for the NVDL Stock Predictor system.

## Local Development Deployment

### Prerequisites
- Python 3.8 or higher
- Git
- Alpaca Markets API account

### Setup Steps

1. **Clone and setup:**
   ```bash
   git clone https://github.com/yourusername/nvdl-stock-predictor.git
   cd nvdl-stock-predictor
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the system:**
   ```bash
   python main.py
   ```

## Docker Deployment

### Build Docker Image

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY . .

   # Create directories for outputs
   RUN mkdir -p results checkpoints

   # Set environment variables
   ENV PYTHONPATH=/app
   ENV PYTHONUNBUFFERED=1

   # Expose port (if adding web interface)
   EXPOSE 8000

   # Run the application
   CMD ["python", "main.py"]
   ```

2. **Build and run:**
   ```bash
   docker build -t nvdl-predictor .
   docker run -e ALPACA_API_KEY=your_key -e ALPACA_SECRET_KEY=your_secret nvdl-predictor
   ```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  nvdl-predictor:
    build: .
    environment:
      - ALPACA_API_KEY=${ALPACA_API_KEY}
      - ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
    volumes:
      - ./results:/app/results
      - ./checkpoints:/app/checkpoints
    restart: unless-stopped

  # Optional: Add database for storing results
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=nvdl_predictor
      - POSTGRES_USER=predictor
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

## Cloud Deployment

### AWS Deployment

#### Using AWS Lambda (Serverless)

1. **Install AWS CLI and configure:**
   ```bash
   pip install awscli
   aws configure
   ```

2. **Create deployment package:**
   ```bash
   pip install -r requirements.txt -t ./package
   cp -r . ./package/
   cd package && zip -r ../deployment.zip .
   ```

3. **Deploy to Lambda:**
   ```bash
   aws lambda create-function \
     --function-name nvdl-predictor \
     --runtime python3.9 \
     --role arn:aws:iam::your-account:role/lambda-execution-role \
     --handler main.lambda_handler \
     --zip-file fileb://deployment.zip \
     --timeout 900 \
     --memory-size 3008
   ```

#### Using AWS EC2

1. **Launch EC2 instance:**
   - Choose Amazon Linux 2 AMI
   - Select appropriate instance type (t3.medium or larger)
   - Configure security groups

2. **Setup on EC2:**
   ```bash
   sudo yum update -y
   sudo yum install python3 python3-pip git -y
   git clone https://github.com/yourusername/nvdl-stock-predictor.git
   cd nvdl-stock-predictor
   pip3 install -r requirements.txt
   ```

3. **Configure as service:**
   Create `/etc/systemd/system/nvdl-predictor.service`:
   ```ini
   [Unit]
   Description=NVDL Stock Predictor
   After=network.target

   [Service]
   Type=simple
   User=ec2-user
   WorkingDirectory=/home/ec2-user/nvdl-stock-predictor
   Environment=ALPACA_API_KEY=your_key
   Environment=ALPACA_SECRET_KEY=your_secret
   ExecStart=/usr/bin/python3 main.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Enable and start:
   ```bash
   sudo systemctl enable nvdl-predictor
   sudo systemctl start nvdl-predictor
   ```

### Google Cloud Platform

#### Using Cloud Run

1. **Create Dockerfile** (see Docker section above)

2. **Build and deploy:**
   ```bash
   gcloud builds submit --tag gcr.io/your-project/nvdl-predictor
   gcloud run deploy --image gcr.io/your-project/nvdl-predictor \
     --platform managed \
     --set-env-vars ALPACA_API_KEY=your_key,ALPACA_SECRET_KEY=your_secret
   ```

#### Using Compute Engine

Similar to AWS EC2, but using GCP's Compute Engine instances.

### Azure Deployment

#### Using Azure Container Instances

1. **Create resource group:**
   ```bash
   az group create --name nvdl-predictor-rg --location eastus
   ```

2. **Deploy container:**
   ```bash
   az container create \
     --resource-group nvdl-predictor-rg \
     --name nvdl-predictor \
     --image your-registry/nvdl-predictor:latest \
     --environment-variables ALPACA_API_KEY=your_key ALPACA_SECRET_KEY=your_secret \
     --restart-policy OnFailure
   ```

## Scheduled Execution

### Using Cron (Linux/Mac)

Add to crontab for daily execution:
```bash
crontab -e
# Add line for daily execution at 6 PM
0 18 * * * cd /path/to/nvdl-predictor && python main.py
```

### Using Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (daily, weekly, etc.)
4. Set action to start program: `python.exe`
5. Add arguments: `main.py`
6. Set start in directory to project path

### Using GitHub Actions (CI/CD)

Create `.github/workflows/scheduled-run.yml`:
```yaml
name: Scheduled Prediction Run

on:
  schedule:
    - cron: '0 18 * * *'  # Daily at 6 PM UTC
  workflow_dispatch:  # Manual trigger

jobs:
  run-prediction:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run prediction
      env:
        ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
        ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
      run: |
        python main.py
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: prediction-results
        path: results/
```

## Monitoring and Logging

### Application Monitoring

1. **Add health check endpoint:**
   ```python
   # Add to main.py
   def health_check():
       return {"status": "healthy", "timestamp": datetime.now().isoformat()}
   ```

2. **Setup monitoring with Prometheus/Grafana:**
   - Install prometheus_client
   - Add metrics collection
   - Configure Grafana dashboards

### Log Management

1. **Centralized logging with ELK Stack:**
   - Elasticsearch for storage
   - Logstash for processing
   - Kibana for visualization

2. **Cloud logging:**
   - AWS CloudWatch
   - Google Cloud Logging
   - Azure Monitor

## Security Considerations

### API Key Management

1. **Use environment variables:**
   ```bash
   export ALPACA_API_KEY="your_key"
   export ALPACA_SECRET_KEY="your_secret"
   ```

2. **Use cloud secret managers:**
   - AWS Secrets Manager
   - Google Secret Manager
   - Azure Key Vault

3. **Use encrypted configuration files:**
   ```python
   from cryptography.fernet import Fernet
   # Encrypt sensitive configuration
   ```

### Network Security

1. **Use HTTPS for all API calls**
2. **Implement rate limiting**
3. **Use VPN for cloud deployments**
4. **Configure firewalls appropriately**

## Performance Optimization

### Resource Optimization

1. **Memory management:**
   - Use data streaming for large datasets
   - Implement garbage collection
   - Monitor memory usage

2. **CPU optimization:**
   - Use multiprocessing for parallel tasks
   - Optimize model training parameters
   - Use GPU acceleration when available

### Caching

1. **Data caching:**
   ```python
   import redis
   # Cache market data
   ```

2. **Model caching:**
   - Save trained models
   - Use model versioning
   - Implement model warm-up

## Backup and Recovery

### Data Backup

1. **Automated backups:**
   ```bash
   # Backup results and models
   tar -czf backup-$(date +%Y%m%d).tar.gz results/ checkpoints/
   ```

2. **Cloud storage:**
   - AWS S3
   - Google Cloud Storage
   - Azure Blob Storage

### Disaster Recovery

1. **Multi-region deployment**
2. **Database replication**
3. **Automated failover**
4. **Recovery testing**

## Troubleshooting

### Common Deployment Issues

1. **Dependency conflicts:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

2. **Memory issues:**
   - Increase instance size
   - Optimize batch sizes
   - Use data streaming

3. **API rate limits:**
   - Implement exponential backoff
   - Use multiple API keys
   - Cache data when possible

### Debugging

1. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use profiling tools:**
   ```python
   import cProfile
   cProfile.run('main()')
   ```

3. **Monitor system resources:**
   ```bash
   htop  # CPU and memory usage
   iotop # Disk I/O
   ```

## Maintenance

### Regular Tasks

1. **Update dependencies:**
   ```bash
   pip list --outdated
   pip install --upgrade package_name
   ```

2. **Clean up old files:**
   ```bash
   find results/ -name "*.html" -mtime +30 -delete
   ```

3. **Monitor logs:**
   ```bash
   tail -f logs/application.log
   ```

### Version Updates

1. **Test in staging environment**
2. **Backup current version**
3. **Deploy with rollback plan**
4. **Monitor post-deployment**

This deployment guide covers the major deployment scenarios. Choose the option that best fits your infrastructure and requirements.