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

### Resolution Check and Super-Resolution
Every upload is first measured by the Adaptive Resolution Router (ARR): Canny edge density, local contrast and the share of high-frequency energy in the 2D FFT. If the check flags the image, and the image is smaller than the detector input (long side under 640 px), it is enlarged 2x with bicubic interpolation, sharpened with an unsharp mask (alpha 1.5, sigma 3) and averaged back to its size. Larger images are never enhanced. The results screen shows which path the image took and the three measurements behind the decision.
* With the thresholds used in the paper (high-frequency ratio below 0.7), the check flags almost every image. On 400 validation images enhancement did not improve detection (F1 0.770 without it, 0.749 with it), so it can be switched off with `SR_POLICY=off`.

### Asynchronous Dual-Track Ensemble
The Celery worker passes each PCB image through both YOLO26n and RT-DETR-L.
* Boxes are merged with asymmetric Weighted Box Fusion: track weights 1.0 (YOLO26n) and 1.5 (RT-DETR-L), IoU 0.5, minimum fused score 0.25.
* Only the fused output is shown to the operator, with each detection's source (both detectors, or one) and per-detector confidences.
* The API also returns the raw per-track counts so the ensemble contribution can be checked.

### Client-Side Thresholding
The worker returns every fused detection with a score of 0.25 or more. The Next.js slider (20% to 95%, default 40%) filters them in the browser:
* 40% gave the best balance on validation images (precision 0.82, recall 0.70, 0.3 false alarms per image, against 0.68 / 0.81 / 0.73 at 25%).
* Moving the slider redraws the canvas without any request to the backend.
* A notice appears when the scan looks unlike the training data (for example green solder-mask boards) or when no detection reaches 50%.
* **Export CSV** downloads the detections currently shown.

### Defect Inspection
Hovering a box on the image shows a detail card (class, fused score, which detectors agree, size and position) and scrolls the catalogue to its entry. Hovering a catalogue entry highlights its box.

### Automated Manufacturing Recommendations
Based on the detected defect classes, the system evaluates the root cause in the PCB fabrication line (etching, drilling, wave soldering, or cleanroom handling) and displays prioritized physical remediation steps.

---

## Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── pipeline.py          # Resolution check, super-resolution, dual-track inference, WBF
│   │   ├── celery_worker.py     # Celery task: runs the pipeline, saves the record in the background
│   │   ├── database.py          # SQLAlchemy connection engine (pg8000 driver)
│   │   ├── main.py              # FastAPI endpoints, health check, upload validation
│   │   ├── models.py            # PostgreSQL schema with pgvector embeddings
│   │   └── training_cron.py     # Scheduled task skeleton for incremental retraining
│   ├── test_db.py               # Database table initialization script
│   └── requirements.txt         # Python package dependencies
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx       # Root Next.js layout configuration
│   │   │   └── page.tsx         # Primary application container and view router
│   │   ├── lib/                 # API address resolution, shared types
│   │   └── components/ui/
│   │       ├── FileUploader.tsx # Upload, stage stepper, status polling, error state
│   │       ├── ResultsView.tsx  # Canvas renderer, slider, hover detail, CSV export
│   │       ├── PipelineStatus.tsx # Header status from /api/health
│   │       └── ThemeToggle.tsx  # Light / dark theme
│   ├── public/config.json       # Backend address used by the deployed site
│   ├── package.json
│   └── tsconfig.json
├── docker-compose.yml           # PostgreSQL (pgvector) and Redis services
├── start_app.bat                # Developer launcher: dev server, API and worker
├── start_demo.bat               # Demo: API + worker + public tunnel (for the Vercel site)
├── start_demo_local.bat         # Demo: everything on this laptop, no tunnel
├── run_tunnel.py                # Keeps the tunnel on its fixed address, restarts it if it drops
├── pcb_data.yaml                # YOLO dataset configuration
└── .gitignore                   # Excludes raw datasets, weights, and caches
```

---

## Running the demo (Windows)

Both launchers need the model weights (`MODEL RESULTS/MORE TRAINING/YOLO/best.pt` and `.../DETR/best.pt`, or set `YOLO_WEIGHTS` / `DETR_WEIGHTS`) and a `backend/.env` with `REDIS_URL` and `DATABASE_URL`.

| Launcher | What it runs | Use it when |
| :--- | :--- | :--- |
| `start_demo.bat` | API, Celery worker and a public tunnel at `https://fodpcb-predictive-api.loca.lt`; the Vercel site (`fodpcb-predictive.vercel.app`) talks to it | Showing the deployed site |
| `start_demo_local.bat` | The same backend plus the website at `http://localhost:3000`, no tunnel | The tunnel is slow or down |

Wait for the Celery Worker window to print `ready` (about 30 seconds). The header shows "Pipeline Online" once the worker answers. Keep the laptop awake and online. The first inspection after a start is slower.

Settings (environment variables for the backend):

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `SR_POLICY` | `gate` | `off` disables super-resolution |
| `SR_MAX_SIDE` | `640` | Images with a long side at or above this are never enhanced |
| `INFERENCE_MODE` | `celery` | `inline` runs inference inside the API, no Redis needed (development) |
| `YOLO_WEIGHTS`, `DETR_WEIGHTS` | repo paths | Weight file locations |

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
| `GET` | `/api/health` | Whether an inference worker is available (drives the header status) |
| `POST` | `/api/inspect` | Uploads the image and queues the inspection in one request |
| `POST` | `/api/upload/local` | Ingests a PCB image and saves it to local storage |
| `POST` | `/api/inference/trigger` | Queues a stored image for inspection |
| `GET` | `/api/inference/status/{task_id}` | Task state: `processing` with its current stage, `completed` with the result, or `failed`. `?wait=15&since=<stage>` holds the request until the stage changes |
| `GET` | `/{file_path}` | Static file endpoint serving raw PCB scans |

---

## Technical Specifications

* **Ensemble fusion**: asymmetric Weighted Box Fusion (weights 1.0 and 1.5, IoU 0.5, minimum score 0.25). When one detector reports nothing, the other's boxes are kept with their confidence halved.
* **Training data and limits**: both detectors were trained on DsPCBSD+ (nine classes, 226 px close-ups of copper-toned boards). Scans of other board types, such as green solder-mask boards, are outside the training data and give unreliable results.
* **Validation results** (200 to 400 DsPCBSD+ validation images, this pipeline): recall 0.81 to 0.85 and precision 0.68 to 0.73 at the 25% reporting threshold, depending on the sample.
* **Records**: each inspection is saved to PostgreSQL in the background, so a slow or unreachable database never delays a result. The 512-dimensional `embedding` column is a placeholder: the vectors currently stored are random, not model embeddings, so similarity search is not yet meaningful.
* **Database driver**: pure-Python `pg8000`, with connection timeouts and health checks.
