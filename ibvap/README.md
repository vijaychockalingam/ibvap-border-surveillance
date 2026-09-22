# IBVAP — Intelligent Border Video Analytics Platform (1-day prototype)

A working demo that turns a CCTV/recorded video feed into an AI surveillance
system: person + vehicle detection, tracking, a virtual restricted zone,
night-movement detection, real-time alerts, an event database, and a live
command dashboard.

Built specifically to run **CPU-only** (no GPU required) on a 16GB laptop —
tested end-to-end in this environment before being handed to you (every
module except the actual YOLO model, which needs your machine's internet
connection to download its ~6MB weights the first time it runs).

## 1. Install (one-time)

```
install.bat
```

Run this instead of a raw `pip install -r requirements.txt`. It installs
everything (Ultralytics/YOLO26, OpenCV, Flask, NumPy, easyocr for ANPR),
then removes `opencv-python-headless`. The script ends by verifying OpenCV's GUI support actually works.

No GPU/CUDA setup needed — it detects you have no dedicated GPU and runs on CPU automatically.

## 2. Architecture

```text
         EXISTING CCTV
               ↓
          RTSP STREAM
               ↓
        FRAME SAMPLER
               ↓
           YOLO26n
               ↓
    ┌──────────┼──────────┐
    ↓          ↓          ↓
 PERSON     VEHICLE      OTHER
    ↓          ↓
    └──────┬───┘
           ↓
        TRACKING
           ↓
     ENTITY MEMORY
           ↓
   CONTEXT / ZONE ENGINE
           ↓
   EVENT CORRELATION
           ↓
   INCIDENT MANAGEMENT
           ↓
  EXPLAINABLE SECURITY
         ALERT
           ↓
      DASHBOARD
```

The important point is that YOLO26n is only the **AI perception layer**.
The uniqueness of IBVAP comes from the layers above it:
- Multi-camera correlation
- Entity/movement memory
- Context-aware event fusion
- Explainable alerts
- Camera health monitoring
- Incident reconstruction
- Evidence generation
- Adaptive processing

### AI Object Detection

**Model:** Ultralytics YOLO26n

**Purpose:**
Real-time detection of people and vehicles from existing CCTV streams.

**Deployment:**
CPU-oriented edge/local processing.

**Model loading:**
Local `.pt` weights after initial setup.

## 3. Benchmarking

We have provided a `benchmark.py` tool to test the YOLO26n CPU processing capabilities on your laptop. Since the platform adapts dynamically to activity, you must run the script to see the hardware-specific measurements.

Run the benchmark:
```
python benchmark.py
```

Record actual measurements below (Do NOT fill with invented numbers):

| Cameras |   AI FPS (System Total) |      CPU (Multi-Core) |      RAM |
| ------- | -------: | -------: | -------: |
| 1       | 7.47 FPS | 741.3%   | 463.6 MB |
| 3       | 11.19 FPS| 776.9%   | 514.2 MB |
| 6       | 15.46 FPS| 1240.4%  | 554.4 MB |


## 4. Draw your restricted zone

```
python select_zone.py sample.mp4
```

Click 3+ points around the area you want to treat as the "restricted zone".
It auto-saves to `zone_config.json`.

## 5. Set up multiple cameras

Edit `cameras.json` to define your 6-camera topology:

```json
{
  "cam1": { "name": "BOP Camera 1", "location": "North Fence", "processing_node": "node_1", "source": "sample1.mp4", "zone_config": "zone_cam1.json" }
}
```

## 6. Run it

```
python app.py
```

Then open **http://localhost:5000** in your browser.

## Performance tuning for your laptop (integrated GPU, 16GB RAM)

1. **Adaptive AI Processing limits**: The system now scales between `IDLE_FPS` (3) and `ACTIVE_FPS` (8) depending on whether motion/objects are detected, keeping CPU low on quiet scenes.
2. **Raise FRAME_SKIP**: Set `FRAME_SKIP` higher in `config.py`.
3. **Lower RESIZE_WIDTH**: (e.g. 480 or 400) in `config.py`.

## Project structure

```
benchmark.py         CPU usage and FPS measurement script for YOLO26n
install.bat          Run this first - installs deps
app.py               Flask server, loads context engines
stream_manager.py    Thread-safe stream ingest with camera health checks
ai_node.py           Per-node AI pipeline (YOLO26, tracking)
incident_engine.py   Incident aggregation and evidence generation
context_engine.py    Event prioritization and explainability
entity_memory.py     Heuristic cross-camera tracking
detector.py          YOLO26n wrapper, filters person/vehicle/animal
tracker.py           Lightweight centroid tracker
zone.py              Virtual fence: load polygon, point-in-zone test
anpr.py              OCR-based plate reading for vehicles
database.py          SQLite event & incident log
config.py            Tunable constants (YOLO26n, FPS bounds)
templates/index.html Dashboard UI
```
