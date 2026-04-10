# 🧠 Classroom AI: Intelligent Attendance & Cognitive Engagement

**Classroom AI** is an advanced, hybrid-environment learning analytics system. It is designed to bridge the gap between physical classrooms and online learning environments by employing state-of-the-art Computer Vision, Facial Recognition, and Large Language Models (LLMs) to automatically track attendance and dynamically monitor student engagement.

---

## ✨ Core Highlights & Architecture

### 1. Offline Mode: Dual-Stream Computer Vision Engine
For physical classrooms, the system processes live CCTV/webcam feeds via asynchronous WebSockets or processes pre-recorded batch videos.
* **Identity Recognition (ArcFace / InsightFace):** The system uses the high-fidelity `buffalo_l` multi-task model. Instead of storing images, it extracts rich facial embeddings and maps them into a robust **FAISS (Facebook AI Similarity Search)** index. This permits lightning-fast attendance marking across 30+ frames per second.
* **Spatial Temporal Tracking (ByteTrack):** Ensuring students who briefly look away or get obscured aren't dropped, temporal tracking maps their spatial coordinates across time, linking Face Embeddings consistently.
* **Cognitive Engagement & Posture Estimation (YOLOv8-Pose):** We deployed a highly optimized `yolov8n-pose` model to map skeletal telemetry. By continuously calculating the mathematical vectors of the eyes, nose, and shoulders (Yaw/Pitch geometric relationships), the system achieves deterministic mapping of whether a student is "Attentive", "Distracted", "Drowsy", or actively "Sleeping".
* **Manual Resolution Grid:** If an unregistered student sits in the class, the system silently captures a high-resolution base64 facial crop and places them in the "Unidentified Feed" on the dashboard, allowing the teacher to manually map them to a database entry seamlessly.

### 2. Online Mode: The "Classroom AI Connect" Extension
Traditional cameras are privacy-invasive and computationally heavy for students' personal laptops. We designed a lightweight, zero-latency **Chrome Extension** to monitor online classes (e.g. Google Meet).
* **Instant Auto-Attendance:** Students login securely with their Enrollment ID through the Chrome extension popup. The split-second they connect to the teacher's session, their attendance is instantly validated via an HTTP tunnel, bypassing Google Meet's strict CSP policies.
* **Tab-Switching Penalties:** Built using Chrome's native `tabs.onActivated` API, the extension registers if a student clicks away from the online lecture (e.g. browsing Instagram) and deducts engagement points dynamically.
* **LLM Cognitive Pop-up Quizzes:** A custom integration using the blazing fast **Llama 3** (via the Groq API). The teacher can input the current topic being discussed. Directly inside the extension popup, students can trigger a "Cognitive Check" which utilizes Meta's Llama-3 model to dynamically generate an academic multiple-choice question on the spot. Answering correctly grants engagement points, failing issues a penalty.

### 3. The Centralized Hub
* **Frontend:** Built natively in React & Vite. It constructs elegant and fluid Dark-Mode Dashboards utilizing Raw Modular CSS, Recharts for granular visual telemetry data, and optimistic un-blocking UI to show frames and attendance grids without delay.
* **Backend:** Built on Python `FastAPI` to handle the immense multiprocessing concurrency scaling required for video stream analytics alongside parallel API REST routes.
* **Database:** Robust multi-relational `SQLite` architecture capturing enrollment profiles, faiss dictionaries, live tracking caches, temporal histories, and consolidated analytic summaries. 

---

## 🛠️ Technology Stack & "Why We Used It"

| Technology / Component | Purpose & Rationale |
| :--- | :--- |
| **FastAPI (Python)** | Selected over Flask/Django for its native async/await capabilities, which are strictly mandatory for hosting high-throughput WebSocket video feeds alongside deep-learning inference. |
| **InsightFace (ArcFace)** | Chosen due to its massive superiority in partial-face recognition and varying light environments compared to standard `face_recognition` libraries (dlib). |
| **YOLOv8-Pose (Ultralytics)** | The industry standard for real-time skeleton tracking. Used to extract facial geometric landmarks to calculate head-tilt angles (distraction mapping) instantly without relying on a slow secondary classifier model. |
| **FAISS** | Facebook's incredibly fast vector-search library structure ensures that even with hundreds of enrolled student facial embeddings, comparing a webcam face to the database takes fractions of milliseconds. |
| **Groq API & Llama-3** | For the Extension Quizzes. Groq uses Local Processing Units (LPUs) achieving ~800 tokens per second. In a live classroom, waiting 10 seconds for GPT-4 to make a quiz is unacceptable; Llama-3 via Groq does it in <1 second. |
| **React + Vite** | For the dashboard. Provides reactive DOM management to update the live stream canvases and attendance tables smoothly. Utilized custom raw CSS/Glassmorphism to give a highly premium, modern, distraction-free analytical environment. |
| **Chrome Manifest V3** | Built to securely tunnel API requirements from restricted sandbox environments (Google Meet) into our local backend using Background Service Workers. |

---

## 🚀 How to Run the Ecosystem

### 1. Backend AI Server
```bash
cd backend
python -m venv venv
venv\Scripts\activate   # (On Windows)
pip install -r requirements.txt

# IMPORTANT: Create a .env file and add your Groq Key: 
# GROQ_API_KEY=gsk_your_actual_key

python -m app.main
```

### 2. Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```

### 3. Chrome Extension (Online Mode)
1. Go to `chrome://extensions/` in your Chrome browser.
2. Enable **Developer mode** in the top right.
3. Click **Load unpacked** and select the `/chrome-extension/` directory.
4. Click the extension icon to sign in with an Enrollment ID. 

---

*This full-stack system entirely modernizes the pipeline from identity establishment to behavioral analysis, empowering educators with complete transparency whether they are in the hall or across the globe.*
