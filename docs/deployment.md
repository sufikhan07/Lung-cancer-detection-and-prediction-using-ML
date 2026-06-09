# Deployment Guide

This guide covers deploying the Lung Cancer Diagnostic System to various environments.

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- PostgreSQL (optional, for data persistence)
- Redis (optional, for caching)
- SSL certificates (for production)

## Quick Start with Docker

### Development Deployment

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd lung-cancer-diagnostic-system
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services:**
   ```bash
   docker-compose up -d
   ```

4. **Access the application:**
   - Web Interface: http://localhost:5000
   - API Documentation: http://localhost:5000/api/docs
   - Health Check: http://localhost:5000/health

### Production Deployment

1. **Update environment variables:**
   ```bash
   # Set production values in .env
   FLASK_ENV=production
   SECRET_KEY=your-secure-secret-key
   DB_PASSWORD=your-db-password
   ```

2. **Use production Docker Compose:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

3. **Setup reverse proxy (Nginx):**
   ```bash
   docker-compose --profile production up -d nginx
   ```

## Manual Installation

### 1. System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3.9 python3.9-venv postgresql redis-server nginx
```

**CentOS/RHEL:**
```bash
sudo yum install python39 python39-devel postgresql-server redis nginx
```

### 2. Python Environment

```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup

```bash
# Create database
sudo -u postgres createdb lung_cancer_db
sudo -u postgres createuser --pwprompt lung_cancer_user

# Run migrations (if using SQLAlchemy)
flask db upgrade
```

### 4. Model Setup

```bash
# Create models directory
mkdir -p models

# Download or copy trained models
# lung_cancer_cnn_model.keras
# risk_assessment_model.h5
# scaler.pkl
```

### 5. Configuration

Create `.env` file:
```bash
# Flask Configuration
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key-here

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=lung_cancer_db
DB_USER=lung_cancer_user
DB_PASSWORD=your-db-password

# Redis
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
```

### 6. Start Application

**Development:**
```bash
python src/web/app.py
```

**Production with Gunicorn:**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 src.web.app:app
```

## Cloud Deployment

### AWS ECS

1. **Build and push Docker image:**
   ```bash
   # Build image
   docker build -t lung-cancer-system .

   # Tag for ECR
   docker tag lung-cancer-system:latest your-account.dkr.ecr.region.amazonaws.com/lung-cancer-system:latest

   # Push to ECR
   aws ecr get-login-password --region region | docker login --username AWS --password-stdin your-account.dkr.ecr.region.amazonaws.com
   docker push your-account.dkr.ecr.region.amazonaws.com/lung-cancer-system:latest
   ```

2. **Create ECS cluster and service:**
   ```bash
   # Create cluster
   aws ecs create-cluster --cluster-name lung-cancer-cluster

   # Register task definition
   aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

   # Create service
   aws ecs create-service \
     --cluster lung-cancer-cluster \
     --service-name lung-cancer-service \
     --task-definition lung-cancer-task \
     --desired-count 2 \
     --load-balancers ...
   ```

### Google Cloud Run

```bash
# Build and deploy
gcloud run deploy lung-cancer-system \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars FLASK_ENV=production
```

### Azure Container Instances

```bash
# Create container group
az container create \
  --resource-group your-resource-group \
  --name lung-cancer-container \
  --image your-registry.azurecr.io/lung-cancer-system:latest \
  --dns-name-label lung-cancer-system \
  --ports 80 \
  --environment-variables FLASK_ENV=production
```

## Nginx Configuration

Create `/etc/nginx/sites-available/lung-cancer-system`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeout settings
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }

    # Static files
    location /static/ {
        alias /path/to/your/app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/lung-cancer-system /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## SSL Configuration

### Let's Encrypt (Free SSL)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Automatic renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Manual SSL

Update Nginx configuration:
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/your/certificate.crt;
    ssl_certificate_key /path/to/your/private.key;

    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # ... rest of configuration
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

## Monitoring and Logging

### Application Monitoring

1. **Health Checks:**
   - Endpoint: `/health`
   - Returns system status and model availability

2. **Metrics Collection:**
   - Response times
   - Error rates
   - Model prediction statistics

### Log Management

```bash
# Create log directory
sudo mkdir -p /var/log/lung-cancer-system
sudo chown www-data:www-data /var/log/lung-cancer-system

# Log rotation
cat > /etc/logrotate.d/lung-cancer-system << EOF
/var/log/lung-cancer-system/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload lung-cancer-system
    endscript
}
EOF
```

### Monitoring Tools

- **Prometheus + Grafana:** For metrics collection and visualization
- **ELK Stack:** For log aggregation and analysis
- **Sentry:** For error tracking and alerting

## Backup Strategy

### Database Backup

```bash
# Daily database backup
crontab -e
# Add: 0 2 * * * pg_dump lung_cancer_db > /backup/lung_cancer_$(date +\%Y\%m\%d).sql

# Compress old backups
# Add: 0 3 * * * find /backup -name "*.sql" -mtime +7 -exec gzip {} \;
```

### Model Backup

```bash
# Backup models directory
rsync -av /path/to/models /backup/models_$(date +%Y%m%d)
```

## Scaling

### Horizontal Scaling

1. **Load Balancer:** Use Nginx or AWS ALB
2. **Multiple Application Instances:** Run multiple containers
3. **Database Connection Pooling:** Configure SQLAlchemy pool settings
4. **Redis Cluster:** For distributed caching

### Vertical Scaling

1. **Increase CPU/Memory:** Allocate more resources to containers
2. **Optimize Models:** Use model quantization for faster inference
3. **Database Indexing:** Add indexes for frequently queried data

## Security Checklist

- [ ] Change default SECRET_KEY
- [ ] Use HTTPS in production
- [ ] Configure firewall rules
- [ ] Regular security updates
- [ ] Monitor for vulnerabilities
- [ ] Implement rate limiting
- [ ] Secure database credentials
- [ ] Regular backup verification
- [ ] Log monitoring and alerting

## Troubleshooting

### Common Issues

1. **Model loading failures:**
   - Check model file paths
   - Verify TensorFlow version compatibility
   - Check file permissions

2. **Database connection errors:**
   - Verify connection string
   - Check firewall settings
   - Confirm database server is running

3. **Memory issues:**
   - Monitor memory usage
   - Adjust Gunicorn worker count
   - Consider model optimization

4. **Slow response times:**
   - Check system resources
   - Optimize database queries
   - Consider caching strategies

### Debug Commands

```bash
# Check application logs
docker-compose logs -f lung-cancer-app

# Check container resource usage
docker stats

# Test API endpoints
curl -X GET http://localhost:5000/health
curl -X POST http://localhost:5000/api/v1/models/status

# Database connectivity test
python -c "import psycopg2; psycopg2.connect('your-connection-string')"
```

## Support

For deployment issues:
1. Check the troubleshooting section
2. Review application logs
3. Verify system requirements
4. Contact the development team</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\docs\deployment.md