# 🚀 Quick Start Guide
## Get Classroom AI Running in 5 Minutes

### Option 1: Docker (Recommended)

```bash
# Step 1: Clone/Extract
cd classroom-ai-complete

# Step 2: Start Services
docker-compose up -d

# Step 3: Access
# Frontend: http://localhost:3000
# API: http://localhost:8000/docs
```

### Option 2: Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

### First Steps

1. **Access Dashboard**: http://localhost:3000
2. **Enroll Students**: Upload face images
3. **Create Session**: Set course details
4. **Start Monitoring**: Select camera
5. **View Analytics**: Check engagement

### Troubleshooting

**Docker Issues:**
```bash
docker-compose logs backend
docker-compose restart
```

**Port Conflicts:**
```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Backend
  - "3001:3000"  # Frontend
```

