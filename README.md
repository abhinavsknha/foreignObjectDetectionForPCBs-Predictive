# Dual-Track PCB Defect and Foreign Object Detection System

A decoupled inspection system for printed circuit board (PCB) surface defects and foreign object debris. The system pairs a fast convolutional model with a vision transformer in an asynchronous dual-track pipeline, combines their bounding boxes using Weighted Box Fusion (WBF), stores vector embeddings in PostgreSQL via pgvector, and renders results interactively in a client-side Next.js dashboard.

---

## Architecture Overview

Industrial automated optical inspection (AOI) requires both low inference latency for production line throughput and high recall for microscopic defects. This system separates ingestion, heavy computation, persistence, and visualization across four isolated services:

```
[ Next.js 16 Client ]
       │
       ▼ (HTTP POST / Upload)
[ FastAPI Backend ] ──────► [ Local / Object Storage ]
       │
       ▼ (Task Dispatch via Redis)
[ Celery Inference Worker ]
       ├── Track 1: YOLO
       ├── Track 2: RT-DETR
       └── Fusion: Weighted Box Fusion (IoU >= 0.40)
       │
       ▼ (SQLAlchemy / pg8000)
[ PostgreSQL + pgvector ]
  ├── 512-D Defect Feature Vectors
  └── Bounding Box JSON Payloads
```

1. **Ingestion Layer (FastAPI)**: Accepts high-resolution PCB scans, stores original binaries in dedicated storage, and returns static access keys.
2. **Message Broker (Redis)**: Queues asynchronous inspection tasks, decoupling API response times from heavy neural network execution.
3. **Inference Engine (Celery Worker)**: Loads YOLO and RT-DETR model weights simultaneously in memory. Both detectors process the scan concurrently. Their outputs are merged through a custom Weighted Box Fusion algorithm.
4. **Vector Database (PostgreSQL + pgvector)**: Stores inspection records, detected box coordinates, and 512-dimensional defect space embeddings for future similarity retrieval and incremental model training.
5. **Interactive UI (Next.js)**: Reads coordinate payloads from the API and dynamically draws bounding boxes directly on the client GPU using an HTML5 Canvas element.

---

## Defect Taxonomy

The models are trained to classify defects across nine industrial categories defined in PCB surface inspection standards:

| Code | Defect Class | Description | Physical Action Triggered |
| :--- | :--- | :--- | :--- |
| **SH** | Short | Unintended connection between conductors | Verify wave soldering temperature and flux density |
| **SP** | Spur | Irregular protrusion along the edge of a trace | Check etching chemical concentration and wash timing |
| **SC** | Spurious Copper | Unwanted copper residue on base material or holes | Review etching exposure timing and bath agitation |
| **OP** | Open | Interrupted conductor path breaking circuit continuity | Halt line and inspect photoresist coating application |
| **MB** | Mouse Bite | Localized depression or gap at the conductor edge | Inspect mechanical routing and milling tools for wear |
| **HB** | Hole Breakout | Hole center deviation causing missing pad copper | Recalibrate CNC drilling drill-head alignment coordinates |
| **CS** | Conductor Scratch | Surface scratches across copper traces | Inspect conveyor transfer belts and automated handling arms |
| **CFO** | Conductor Foreign Object | Particles, dust, or contamination on conductive traces | Clean lamination rollers and inspect solder mask environment |
| **BMFO** | Base Material Foreign Object | Contamination on the non-conductive substrate | Verify cleanroom HEPA filtration and panel storage hygiene |

---

## Core Features

### Asynchronous Dual-Track Ensemble
Instead of relying on a single detector, the Celery worker passes each PCB image through both YOLO and RT-DETR models. 
* Predictions with overlap (IoU >= 0.40) are merged using confidence-weighted coordinate averaging.
* Independent high-confidence detections from either detector are preserved.
* The API returns the raw counts from each model alongside the final fused count to verify ensemble contribution.

### Zero-Latency Client-Side Thresholding
The backend worker returns all detected anomalies meeting a low baseline threshold (20%). The Next.js frontend provides an interactive slider (20% to 95%):
* Moving the slider filters the bounding box array client-side.
* The HTML5 Canvas clears and redraws boxes in real time without sending additional HTTP requests to the backend server.

### Crosshair Defect Inspection
Hovering over any item in the detected anomaly list dynamically highlights the corresponding bounding box on the PCB canvas in high-contrast blue, allowing operators to locate tiny flaws on densely packed circuit boards immediately.

### Automated Manufacturing Recommendations
Based on the detected defect classes, the system evaluates the root cause in the PCB fabrication line (etching, drilling, wave soldering, or cleanroom handling) and displays prioritized physical remediation steps.

---

## Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── celery_worker.py     # Dual-track YOLO + RT-DETR inference & WBF math
│   │   ├── database.py          # SQLAlchemy connection engine (pg8000 driver)
│   │   ├── main.py              # FastAPI endpoints and static file mounts
│   │   ├── models.py            # PostgreSQL schema with pgvector embeddings
│   │   └── training_cron.py     # Scheduled task skeleton for incremental retraining
│   ├── test_db.py               # Database table initialization script
│   └── requirements.txt         # Python package dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx       # Root Next.js layout configuration
│   │   │   └── page.tsx         # Primary application container and view router
│   │   └── components/ui/
│   │       ├── FileUploader.tsx # Drag-and-drop upload handler with status polling
│   │       └── ResultsView.tsx  # Canvas defect renderer, confidence slider, hover states
│   ├── package.json
│   └── tsconfig.json
├── docker-compose.yml           # PostgreSQL (pgvector) and Redis services
├── start_app.bat                # Windows one-click launcher for frontend, backend, and worker
├── pcb_data.yaml                # YOLO dataset configuration
└── .gitignore                   # Excludes raw datasets, weights, and caches
```

---

## Getting Started

### Prerequisites
* Python 3.10 to 3.12 (or 3.14 with pg8000 driver)
* Node.js 18+ and npm
* Docker Desktop (running)

### 1. Start Infrastructure Services
Start PostgreSQL with the pgvector extension and Redis:

```bash
docker compose up -d
```

Verify that both containers are running:
* PostgreSQL on port `5432`
* Redis on port `6379`

### 2. Configure the Backend

Navigate to the `backend` directory, create a virtual environment, and install dependencies:

```bash
cd backend
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Linux/macOS

pip install -r requirements.txt
```

Initialize the database schema:

```bash
python test_db.py
```

### 3. Configure the Frontend

Navigate to the `frontend` directory and install JavaScript dependencies:

```bash
cd ../frontend
npm install
```

### 4. Run the Full Stack

#### Option A: One-Click Startup (Windows)
From the project root directory, run the batch script:

```bash
start_app.bat
```

This opens three separate terminal windows:
* Window 1: Next.js dev server at `http://localhost:3000`
* Window 2: FastAPI API server at `http://localhost:8000`
* Window 3: Celery worker processing tasks from Redis

#### Option B: Manual Startup

**Terminal 1 (FastAPI Backend):**
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 (Celery AI Worker):**
```bash
cd backend
celery -A app.celery_worker.celery_app worker --loglevel=info --pool=solo
```
*(Note: `--pool=solo` is required on Windows platforms)*

**Terminal 3 (Next.js Frontend):**
```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` in your web browser.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/upload/local` | Ingests PCB image binary and saves to local storage |
| `POST` | `/api/inference/trigger` | Dispatches image path to the Celery queue for asynchronous inference |
| `GET` | `/api/inference/status/{task_id}` | Polls Celery task state (`PENDING`, `SUCCESS`, or `FAILURE`) |
| `GET` | `/{file_path}` | Static file endpoint serving raw PCB scans to the canvas |

---

## Technical Specifications

* **Ensemble Fusion**: Weighted Box Fusion (WBF) with an IoU matching threshold of `0.40`.
* **Feature Embeddings**: 512-dimensional vectors generated per scan and stored using PostgreSQL `vector(512)` type.
* **Canvas Rendering**: Dynamic 2D context scaling mapped to native image dimensions.
* **Database Driver**: Pure-Python `pg8000` to maintain compatibility with modern Python versions without requiring local C compilers.
