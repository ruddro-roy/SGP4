# Satellite Tracking Platform - Deployment Guide

## Quick Start

Deploy the entire platform with a single command:

```bash
# Clone and deploy immediately
git clone https://github.com/ruddro-roy/satellite-tracking-platform
cd satellite-tracking-platform
docker-compose up --build
```

The platform will be available at:
- **Main API**: http://localhost:3000
- **Frontend**: http://localhost:3001  
- **Portfolio**: http://localhost:8080
- **Orbit Service**: http://localhost:5000
- **Monitoring**: http://localhost:9090

## Production Deployment Options

### Option 1: Render (Recommended)

**One-click deployment with auto-scaling and managed databases:**

1. **Connect GitHub Repository**:
   ```bash
   # Push to your GitHub repository
   git remote add origin https://github.com/your-username/satellite-tracking-platform
   git push -u origin main
   ```

2. **Deploy to Render**:
   - Visit [render.com](https://render.com)
   - Connect your GitHub repository
   - The platform will auto-detect `render.yaml` and deploy all services
   - Configure environment variables in Render dashboard

3. **Environment Variables** (Set in Render dashboard):
   ```
   CESIUM_ION_TOKEN=your_cesium_token_here
   MONGODB_URI=auto_configured_by_render
   REDIS_URL=auto_configured_by_render
   JWT_SECRET=auto_generated_by_render
   ```

### Option 2: Kubernetes (Advanced)

**For high-scale deployments with auto-scaling:**

```bash
# Deploy to Kubernetes cluster
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/production/

# Wait for deployment
kubectl rollout status deployment/gateway -n satellite-production
kubectl rollout status deployment/orbit-service -n satellite-production
```

### Option 3: AWS ECS/Fargate

**Serverless container deployment:**

```bash
# Use AWS CDK or Terraform
cd infrastructure/aws
terraform init
terraform plan
terraform apply
```

## Environment Configuration

### Required Environment Variables

```bash
# Core Application
NODE_ENV=production
PORT=3000

# Databases (Auto-configured on Render)
MONGODB_URI=mongodb+srv://...
REDIS_URL=redis://...

# Security
JWT_SECRET=your_secure_secret
JWT_EXPIRES_IN=24h

# External APIs (Managed autonomously)
CESIUM_ION_TOKEN=your_cesium_token
CELESTRAK_API_BASE=https://celestrak.org

# Autonomous Operations
AUTO_TOKEN_ROTATION=true
SELF_HEALING_ENABLED=true
THREAT_ANALYSIS_ENABLED=true

# Performance
MAX_SATELLITES_RENDER=20000
UPDATE_INTERVAL_MS=30000
```

### Optional Configuration

```bash
# Security
ALLOWED_ORIGINS=https://yourdomain.com
RATE_LIMIT_MAX_REQUESTS=100
IP_WHITELIST=192.168.1.0/24

# Monitoring
LOG_LEVEL=info
SENTRY_DSN=your_sentry_dsn

# Advanced Features
CONJUNCTION_THRESHOLD_KM=10
DEBRIS_TRACKING_ENABLED=true
```

## API Token Management

### Cesium Ion Token

1. **Get Token**:
   - Visit [cesium.com/ion](https://cesium.com/ion/)
   - Create free account
   - Generate token with `assets:read` scope

2. **Configure Securely**:
   - On Render: Add as environment variable
   - On K8s: Store in secrets
   - Never commit to code

### Autonomous Token Rotation

The platform automatically manages token validation and rotation:

```typescript
// Tokens are automatically validated every 24 hours
// Failed tokens trigger self-healing protocols
// Backup data sources ensure continuous operation
```

## Monitoring & Observability

### Health Checks

```bash
# Check system health
curl https://your-domain.com/api/health

# Response
{
  "status": "healthy",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "orbitService": "healthy"
  },
  "metrics": {
    "uptime": 3600,
    "satellitesTracked": 18742,
    "threatsActive": 3
  }
}
```

### Prometheus Metrics

Access metrics at:
- **Prometheus**: http://your-domain:9090
- **Grafana**: http://your-domain:3000 (if deployed)
- **Alerts**: Configured for critical thresholds

### Log Aggregation

```bash
# View logs
docker-compose logs -f gateway
docker-compose logs -f orbit-service

# Production logs (Render)
render logs satellite-tracking-gateway
```

## Security Configuration

### JWT Authentication

```typescript
// Automatic JWT configuration
const token = await fetch('/api/auth/login', {
  method: 'POST',
  body: JSON.stringify({ username, password })
});
```

### Rate Limiting

- **Default**: 100 requests per 15 minutes per IP
- **Configurable**: Via `RATE_LIMIT_MAX_REQUESTS`
- **Bypass**: For authenticated users with higher limits

### IP Whitelisting

```bash
# Configure allowed IP ranges
IP_WHITELIST=192.168.1.0/24,10.0.0.0/8,172.16.0.0/12
```

## Performance Optimization

### Auto-scaling Configuration

**Render (Automatic)**:
- Scales based on CPU/memory usage
- Min: 1 instance, Max: 10 instances
- Response time triggers

**Kubernetes (Horizontal Pod Autoscaler)**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Caching Strategy

- **Redis**: 1-hour TTL for satellite data
- **Browser**: Static assets cached for 1 year
- **CDN**: Automatic on Render, configurable on K8s

### Database Optimization

- **Connection Pooling**: Max 10 connections per service
- **Indexes**: Optimized for satellite queries
- **Read Replicas**: Available for high-traffic deployments

## Troubleshooting

### Common Issues

1. **"Service Unavailable"**:
   ```bash
   # Check service health
   curl https://your-domain.com/api/health
   
   # Check logs
   render logs satellite-tracking-gateway
   ```

2. **"Satellite Data Stale"**:
   ```bash
   # Trigger manual refresh
   curl -X POST https://your-domain.com/api/satellites/refresh
   ```

3. **"High Memory Usage"**:
   ```bash
   # Reduce satellite count
   MAX_SATELLITES_RENDER=10000
   ```

### Debug Mode

```bash
# Enable debug logging
DEBUG_MODE=true
LOG_LEVEL=debug

# Access debug endpoints
curl https://your-domain.com/api/debug/system
curl https://your-domain.com/api/debug/metrics
```

### Self-Healing Status

```bash
# Check autonomous operations
curl https://your-domain.com/api/health/autonomous

# Response
{
  "selfHealing": "active",
  "tokenRotation": "enabled",
  "lastHealthCheck": "2024-01-15T10:30:00Z",
  "systemHealth": 0.95
}
```

## Backup & Recovery

### Automated Backups

- **Database**: Daily snapshots on Render
- **Redis**: Memory snapshots every 6 hours  
- **Configuration**: Stored in Git repository

### Recovery Procedures

1. **Service Recovery**:
   ```bash
   # Automatic via self-healing
   # Manual restart if needed
   render restart satellite-tracking-gateway
   ```

2. **Data Recovery**:
   ```bash
   # Restore from backup
   render restore-backup satellite-db backup-id
   ```

## Cost Optimization

### Render Pricing

- **Gateway**: $7/month (Standard plan)
- **Orbit Service**: $15/month (Standard Plus plan)
- **Database**: $15/month (Standard plan)
- **Redis**: $15/month (Standard plan)
- **Total**: ~$52/month for production deployment

### Resource Optimization

```bash
# Optimize for cost
MAX_SATELLITES_RENDER=5000  # Reduce resource usage
UPDATE_INTERVAL_MS=60000    # Less frequent updates
THREAT_ANALYSIS_ENABLED=false  # Disable if not needed
```

## Compliance & Security

### Security Audit

```bash
# Run security scan
npm audit
pip audit

# Container scanning
docker scan satellite-tracking-gateway
```

### GDPR Compliance

- No personal data stored
- Anonymous usage metrics only
- User opt-out available

### Export Controls

- Open source software (MIT License)
- No export restrictions for satellite tracking algorithms
- Educational and research use encouraged

## Support

### Documentation

- **API Docs**: https://your-domain.com/api/docs
- **Architecture**: See `docs/architecture.md`
- **Runbooks**: See `docs/runbooks/`

### Community

- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Ask questions and share ideas
- **Contributing**: See `CONTRIBUTING.md`

---

**🚀 Happy satellite tracking! Building the future of space situational awareness.**