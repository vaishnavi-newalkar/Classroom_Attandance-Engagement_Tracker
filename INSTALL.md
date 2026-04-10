# Installation Guide

## System Requirements

### Minimum
- CPU: Intel i5 / AMD Ryzen 5 (4 cores)
- RAM: 8 GB
- Storage: 20 GB
- OS: Ubuntu 20.04+ / Windows 10+ / macOS 11+

### Recommended
- CPU: Intel i7 / AMD Ryzen 7 (8 cores)
- RAM: 16 GB
- GPU: NVIDIA RTX 3060 (12 GB VRAM)
- Storage: 50 GB SSD
- OS: Ubuntu 22.04

## Prerequisites

### Docker Method
- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (for GPU)

### Manual Method
- Python 3.10+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+
- CUDA 11.8+ (for GPU)

## Installation Steps

### Docker Installation

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 2. Install NVIDIA Docker (for GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 3. Start Services
cd classroom-ai-complete
docker-compose up -d

# 4. Verify
docker-compose ps
curl http://localhost:8000/health
```

### Manual Installation

**1. Install Python Dependencies**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Install Node Dependencies**
```bash
cd frontend
npm install
```

**3. Setup PostgreSQL**
```bash
sudo apt-get install postgresql-14
sudo -u postgres createuser classroom_user -P
sudo -u postgres createdb classroom_ai
psql classroom_ai < database/schemas/init.sql
```

**4. Install Redis**
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

**5. Download Models**
```bash
cd backend
python scripts/download_models.py
```

**6. Configure Environment**
```bash
cp backend/.env.example backend/.env
# Edit .env with your settings
```

**7. Start Services**
```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate
python main.py

# Terminal 2: Frontend
cd frontend
npm start
```

## Verification

```bash
# Check Backend
curl http://localhost:8000/health

# Check Frontend
curl http://localhost:3000

# Check Database
psql postgresql://classroom_user:password123@localhost:5432/classroom_ai -c "SELECT version();"

# Check Redis
redis-cli ping
```

## Next Steps

1. Access dashboard: http://localhost:3000
2. Read [QUICKSTART.md](QUICKSTART.md)
3. Enroll students
4. Create first session

## Troubleshooting

See [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
