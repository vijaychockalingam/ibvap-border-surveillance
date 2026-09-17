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
everything (Ultralytics/YOLOv8, OpenCV, Flask, NumPy, easyocr for ANPR),
then removes `opencv-python-headless` - a dependency `easyocr` silently
pulls in that breaks `cv2.namedWindow` (used by `select_zone.py`) if left
installed alongside the regular `opencv-python`. This has bitten every
fresh install so far, so it's now handled automatically. The script ends by
verifying OpenCV's GUI support actually works - look for
`OK - OpenCV GUI window support is working.` If you ever see the
`cvNamedWindow`/"Rebuild the library" error again after this, run:
```
pip uninstall opencv-python-headless -y
```

No GPU/CUDA setup needed — it detects you have no dedicated GPU and runs on
CPU automatically.

## 2. Get a demo video

Any CCTV-style clip works — a static or slightly moving overhead/street view
with people and/or vehicles walking through. A few free stock-footage sites
have "CCTV footage" or "street surveillance" clips you can download for a
demo. Save it as `sample.mp4` in this folder (or point `IBVAP_VIDEO` at any
path — see step 4).

## 3. Draw your restricted zone

```
python select_zone.py sample.mp4
```

Click 3+ points around the area you want to treat as the "restricted zone"
(e.g. a border boundary, a fenced area). It **auto-saves after every point**
you add, move, or delete - no save key to remember, and it's safe to close
the window with the X button. Drag an existing point to move it, right-click
to delete it, `R` clears everything, `Q` quits. This writes
`zone_config.json`, which the live system reads on startup.

You can skip this step — the system still runs and detects/tracks people and
vehicles — but intrusion alerts won't fire without a zone defined.

## 3b. (Optional) Set up multiple cameras

By default IBVAP runs one camera. To run 4+ (or any number), edit
`cameras.json` — one entry per camera, each with its own video file and its
own zone config:

```json
{
  "cam1": { "name": "BOP Camera 1 - Main Gate", "source": "sample1.mp4", "zone_config": "zone_cam1.json" },
  "cam2": { "name": "BOP Camera 2 - East Fence", "source": "sample2.mp4", "zone_config": "zone_cam2.json" }
}
```

If you only have one demo clip, it's fine to point every camera entry at the
same file — it still shows up as N distinct labeled tiles on the dashboard.

Draw a zone for each camera (repeat step 3 with the matching output file):
```
python select_zone.py sample1.mp4 zone_cam1.json
python select_zone.py sample2.mp4 zone_cam2.json
python select_zone.py sample3.mp4 zone_cam3.json
python select_zone.py sample4.mp4 zone_cam4.json
```

**All cameras run simultaneously, all the time.** Each one gets its own
background thread doing its own full detect → track → zone-check → alert
pipeline, continuously, independent of whether you're currently looking at
it. The dashboard shows every camera as a tile in a live grid at once - no
switching, no "only one active" limitation. A violation lights up red/amber
on the exact tile it happened on, in addition to being burned into that
camera's video frame and logged to the shared event feed.

This is heavier on your CPU than the earlier single-active-camera version -
see the performance section below for tuning if 4 simultaneous streams
feels sluggish on your laptop.

If a camera's video file is missing or won't open, it's skipped at startup
(shown as an "OFFLINE" tile) instead of crashing the other cameras - check
the terminal log for which one and why.

If a camera crashes mid-session (some CCTV clips can trigger rare
low-level decoder errors), it automatically reopens its video source and
keeps going - it only gives up and marks itself offline after 5 crashes in
a row with no successful frame in between. The dashboard reflects this
live: the camera's status dot goes gray and its tile shows a "FEED
STOPPED" overlay over its last frame if it does eventually give up.

## 4. Run it

```
python app.py
```

Then open **http://localhost:5000** in your browser. You'll see:
- every camera as a live tile in a grid, all streaming and detecting at once
- a system status chip showing how many cameras are online
- live stat counters (person events / vehicle events / high-severity alerts)
- an alarm bar + audible beep + screen flash on new HIGH/CRITICAL events
- a scrolling event log pulled from the shared database, tagged per-camera

To use a different video file for the single-camera fallback (no
`cameras.json` present):
```
# Windows PowerShell
$env:IBVAP_VIDEO="path\to\your\video.mp4"; python app.py
```
Or set `IBVAP_VIDEO=0` to use your laptop's webcam instead of a file.

## How it satisfies the SIH problem statement

**On the "without requiring dedicated FRS, ANPR, or smart-camera hardware"
constraint:** that line restricts *hardware*, not the features themselves —
the very next sentence in the problem statement lists ANPR and face
detection as required capabilities, and asks for them "through software."
That's exactly what this build does: `anpr.py` and `face.py` both operate
on a plain cropped image pulled out of an ordinary `cv2.VideoCapture` frame
- there's no dedicated ANPR camera, no FRS appliance, no special sensor
anywhere in the pipeline. Any standard IP/CCTV camera feed works. This is
worth stating plainly if an evaluator asks about it.

| Problem statement requirement | Where it's implemented |
|---|---|
| Works on existing CCTV video, no special hardware | `processor.py` reads any video file/stream via OpenCV |
| Human detection & tracking | `detector.py` (YOLOv8n) + `tracker.py` (centroid tracker) |
| Vehicle detection & classification | `detector.py` (car/motorcycle/bus/truck classes) |
| Face detection | `face.py` — Haar-cascade detection only, see note below on why it stops there |
| Automatic Number Plate Recognition (ANPR) | `anpr.py` — OCR on the vehicle crop, filtered to plate-shaped strings |
| Virtual fence intrusion detection | `zone.py` + `select_zone.py`, point-in-polygon check per tracked object |
| Suspicious activity detection | `processor.py._check_loitering()` — a person who dwells in the zone past `LOITERING_SECONDS` triggers a separate escalated event |
| Night-time movement detection | `processor.py._is_night()` — average frame brightness threshold |
| Real-time alert generation and event logging | `database.py` (SQLite audit trail) + `templates/index.html` (alarm bar, beep, screen flash, mute toggle) |
| Multi-camera analytics, all simultaneous | `app.py` — every camera runs continuously in its own thread; dashboard shows all as live tiles at once |
| Watchlist comparison (vehicle identification) | `watchlist.json` + `processor.py` — plate match escalates to a CRITICAL `WATCHLIST_MATCH` event |
| Event-based processing (not storing every frame) | Only alert-worthy detections are written to DB/disk, not every frame |
| Support integration with command/control systems | REST endpoints (`/api/events`, `/api/stats`, `/api/cameras`) return structured JSON any external system could poll |

Deliberately **left out** of this prototype: face *recognition/identification*
(matching a face to a known identity — see below), and edge/distributed
deployment across multiple physical machines. Both are natural "future
work" slide bullets — the architecture (event-based, modular
detector/tracker/zone/anpr/face) is built so either could be added without
restructuring anything.

## Face detection vs. face recognition — read this before your pitch

This prototype does **face detection** (drawing a box when a face is
visible) — it does **not** do face *recognition/identification* (matching
that face to a known person's identity). That's a deliberate choice, not a
missing feature:

- Identification needs an enrolled photo database, an embedding/matching
  model, and careful accuracy tuning - a materially bigger project than
  detection, and not something that can be made reliable in a 1-day build.
- Even production border-security systems struggle with false matches on
  real CCTV-quality footage; on a laptop demo the false-positive rate would
  undermine the pitch rather than strengthen it.
- It also crosses from "detecting a person is present" into "identifying
  who someone is", which raises real privacy/governance questions the
  moment you're matching individuals rather than just detecting them.

If asked about this in front of evaluators, the honest and stronger answer
is: *"face detection is implemented and working; face recognition against a
watchlist is a natural next phase that needs an enrolled database and more
compute than a laptop prototype - we scoped it out deliberately rather than
ship something unreliable."* That's a better answer than an identification
system that misfires on stage.

Plate-watchlist matching, by contrast, **is** implemented (`watchlist.json`)
- text matching against OCR output is a much more tractable problem than
  face matching, and is explicitly called for in the problem statement.

Edit `watchlist.json` to add/change plates you want flagged:
```json
{ "plates": ["TN38XY9988", "KA05AB1234"] }
```
Plates are matched uppercase with spaces stripped, so `"tn 38 xy 9988"` and
`"TN38XY9988"` are treated as the same plate.

## ANPR (number plate reading)

Vehicles get a basic plate read attempt using OCR directly on the cropped
vehicle box (no separate plate-detector model - matches the problem
statement's "demo it on sample footage, don't claim production-grade ANPR"
guidance). It retries a few frames, then gives up and logs the vehicle
either way (with the plate if found, or marked unreadable). If the plate
matches an entry in `watchlist.json`, the event escalates to a `CRITICAL`
`WATCHLIST_MATCH` alert (red pulsing alarm bar + triple-beep on the
dashboard, vs. a single beep for a normal HIGH alert).

To enable it:
```
pip install easyocr
```
It's already in `requirements.txt`. If you skip this, the app still runs
fine - you'll just see a one-time "ANPR disabled" message in the terminal
and vehicles get logged without plate numbers.

Note: `easyocr` pulls in its own model weights on first use (small
download) and adds noticeably more CPU work per vehicle than the rest of
the pipeline. If it feels slow, lower `config.ANPR_MAX_ATTEMPTS` (e.g. to
3), or set `config.ENABLE_ANPR = False` to turn it off entirely for a
smoother demo.

## Alarm system

The dashboard has an alarm bar at the top. When a new HIGH-severity event
(intrusion) or CRITICAL event (watchlist match) comes in, it:
- Plays an audible beep (triple-beep, higher pitch for CRITICAL)
- Flashes a red border around the whole page briefly
- Shows a red pulsing "ALERT ACTIVE" bar with the alert type

There's a mute button if the beeping gets in the way while you're setting
up - it only affects sound, the visual flash and alert log keep working.
Browsers require a user click before they'll allow audio to play at all,
so click anywhere on the page (e.g. a camera tab) once after loading it,
or the first alarm might be silent.

## Real-time footage (webcam or live IP camera)

Any camera's `"source"` field in `cameras.json` accepts:
- **A video file path** — `"sample1.mp4"` (what you've been using)
- **A webcam index** — `"0"` for your laptop's built-in camera, `"1"` for a
  second USB camera, etc.
- **A live stream URL** — `"rtsp://192.168.1.50:554/stream1"` for a real
  IP/CCTV camera on your network, or an HTTP MJPEG URL. OpenCV connects to
  these the same way it opens a file.

If you don't have access to an actual CCTV/IP camera for the hackathon, a
free phone app like "IP Webcam" (Android) turns your phone into an RTSP/HTTP
camera on your WiFi network in a couple of taps - a quick way to get a real
live feed into the demo instead of only prerecorded clips.

## Performance tuning for your laptop (integrated GPU, 16GB RAM)

Since all cameras now run **simultaneously**, each in its own full detection
pipeline, CPU load scales roughly linearly with camera count - 4 cameras is
genuinely ~4x the work of 1. If playback feels sluggish:

1. Raise `FRAME_SKIP` (e.g. 4–6 for 4 cameras) — runs YOLO less often per
   camera, the tracker fills in the gaps between detections.
2. Lower `RESIZE_WIDTH` (e.g. 480 or 400) — smaller frames = faster inference,
   applies to every camera at once.
3. Turn off the heavier extras if you don't need them for every camera:
   `config.ENABLE_ANPR = False` and/or `config.ENABLE_FACE_DETECTION = False`
   cut real CPU cost per vehicle/person across all 4 streams.
4. Lower `config.STREAM_FPS` (e.g. 8) — this only throttles how often the
   browser is sent a new frame, it doesn't touch detection speed, but it
   reduces encoding/network overhead if that's part of the slowdown.
5. Keep demo clips short (30–90 sec) and looped rather than long videos —
   doesn't affect speed, but keeps memory/disk use predictable over a long
   demo session.

**Safest option for judging day:** run it once beforehand to confirm all 4
cameras start cleanly and alerts fire, then just re-run `app.py` fresh right
before your demo slot — every camera loops its video automatically, so
they'll keep generating fresh alerts the whole time you're presenting
without you needing to touch anything.

## Project structure

```
install.bat          Run this first - installs deps and fixes the OpenCV conflict
app.py            Flask server: starts every camera's background thread at
                    boot, dashboard route, video streams, event/stat/camera APIs
processor.py       Per-camera pipeline, runs continuously in its own thread:
                    detect -> track -> zone check -> loitering -> ANPR -> face
                    -> alert -> draw -> encode into a shared JPEG buffer
detector.py         YOLOv8n wrapper, filtered to person/vehicle classes
tracker.py          Lightweight centroid tracker (assigns stable IDs)
zone.py             Virtual fence: load polygon, point-in-zone test
select_zone.py       CLI tool to click/drag-edit the zone on a video, per-camera
anpr.py              OCR-based plate reading for vehicles
face.py              Face detection only (see note above on why not recognition)
watchlist.json         Plate numbers that escalate a vehicle event to CRITICAL
cameras.py            Loads cameras.json (falls back to single camera)
cameras.json           Your camera list - edit this to add/change cameras
database.py          SQLite event log, tagged per-camera, includes plate_number
config.py            All tunable constants in one place
templates/index.html Dashboard UI: camera tabs, alarm bar, alert feed
snapshots/            Saved JPEG snapshots for each logged alert
```
