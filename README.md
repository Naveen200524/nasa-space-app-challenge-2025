## SeismoGuard — Planetary Seismology Explorer

Desktop-first web app to explore seismic activity on Earth, Mars, and the Moon.

- Frontend: Vanilla HTML/CSS/JS with Three.js (GLTF), Plotly, and a static server.
- Backend: Flask API with ObsPy-based processing, classical detection (STA/LTA, Z-score), optional ML, and real data from IRIS and NASA PDS.
- Works offline with graceful fallbacks; auto-upgrades to live data when the backend is running.

## Features

- 3D planet viewers (GLB) with aligned, static poses for clean comparison.
- Waveform visualizer with real-time and historical modes, and event markers.
- Timeline synced with detections returned by the backend.
- Data sources: file upload, IRIS (FDSN), URL fetch, and PDS search (InSight).
- Optional wind-noise masking for Mars (pressure channel).
- API surfaces health, algorithms, presets, and enhanced endpoints (earthquakes feed, IRIS utilities, ML stubs, satellite correlation stubs).

## Repository structure

```
.
├─ index.html                # App entry (desktop-only UI)
├─ css/                      # Styles (desktop-first layout, sticky headers)
├─ js/                       # Frontend modules (Three.js viewers, Plotly charts)
├─ public/                   # GLB models and static assets
├─ backend/                  # Flask API + processing pipeline
│  ├─ app/                   # API, fetchers, preprocess, detectors, etc.
│  ├─ run_server.py          # Start script with startup checks
│  ├─ requirements.txt       # Python deps (Windows-friendly pins)
│  └─ README.md              # Backend‑specific docs
├─ start_all.bat             # One-click (Windows): venv + backend + static server
├─ start_all.ps1             # Orchestrator invoked by the batch file
└─ README.md                 # This file
```

## Prerequisites

- Windows 10/11 (PowerShell 5.1+)
- Python 3.11 (64‑bit)

Optional (development):
- Git, VS Code

## Quick start (Windows)

1) From the repo root, run the one-click starter:

```powershell
.\n+start_all.bat
```

This will:
- Create and use `backend\venv` (isolated deps)
- Install backend requirements
- Start the backend on http://127.0.0.1:5000
- Start a static server on http://127.0.0.1:8080
- Open the app in your browser

If PowerShell blocks scripts, the `.bat` already uses ExecutionPolicy Bypass for its child process.

## Manual run (alternative)

Backend:

```powershell
cd backend
python -m venv venv
.\n+venv\Scripts\python.exe -m pip install --upgrade pip
.
venv\Scripts\python.exe -m pip install -r requirements.txt
.
venv\Scripts\python.exe run_server.py --host 0.0.0.0 --port 5000
```

Frontend (new terminal, repo root):

```powershell
python -m http.server 8080
```

Open http://127.0.0.1:8080 in a desktop browser.

## Using the app

- Home: select a planet and click “Explore Seismic Activity”.
- Dashboard: the waveform fetches real data (when backend is up) and overlays detected events; the timeline syncs automatically.
- Compare: side‑by‑side planet viewers and activity charts.

Notes:
- The app also works without the backend (mock data); it switches to real data once the API is reachable.
- The frontend targets desktop browsers; mobile isn’t supported.

## API overview (backend)

Base URL: `http://127.0.0.1:5000`

- GET `/health` — Health check and algorithms list.
- POST `/detect` — Main detection endpoint.
	- Sources: `seismic` file upload or `source=iris|url|pds_search`
	- Common params: `planet=earth|mars|moon`
	- IRIS: `network, station, channel, starttime, endtime`
	- URL: `url`
	- PDS: `mission=insight`, `instrument=SEIS`
	- Returns: `events`, `diagnostics`, and compact `timeseries` (starttime, sampling_rate, samples[])
- GET `/earthquakes/recent` — Recent earthquakes (USGS/EMSC).
- GET `/algorithms` — Available detection algorithms.
- GET `/planet-presets` — Per‑planet processing presets.

Enhanced (best‑effort, optional): IRIS station/event utilities, ML stubs, satellite correlation.

## Configuration

Environment variables (optional):
- `FLASK_ENV=production` — Production mode.
- `SEISMO_MODEL_PATH` — Custom model directory.
- `SEISMO_DATA_PATH` — Data storage path.

## Troubleshooting

- “Backend not reachable” in console:
	- Ensure http://127.0.0.1:5000/health returns JSON.
	- Use the batch starter or start the backend manually in `backend\venv`.
- CORS errors (Origin null):
	- Don’t open `index.html` directly. Serve via `python -m http.server` and use http://127.0.0.1:8080.
- Port conflicts:
	- Change ports via `start_all.ps1` parameters or switch the http.server port.
- TensorFlow Addons on Windows/Python 3.11:
	- It’s gated out by markers in `requirements.txt`. No action needed.
- Slow first response:
	- TensorFlow/ObsPy imports can be slow on cold start; the frontend retries with backoff.

## Development

- Code style: Vanilla JS modules and Python 3.11.
- Tests (backend):

```powershell
cd backend
python test_backend.py
```

- Helpful scripts:
	- `start_all.bat` — orchestrates venv + backend + static server.
	- `backend\start_backend.bat` — backend‑only with venv bootstrap.

## Contributing

Issues and PRs are welcome. Please include clear reproduction steps and environment details (OS, Python version).

## License

See the repository’s license file if present.
