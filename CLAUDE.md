# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Fluxia — a research project that solves the 2D TM Maxwell equations (PEC cavity, TM₁₁ mode) two ways and compares them cell-by-cell on the same mesh and time steps:

- **PINN** (`model.py`, `physics.py`, `train.py`): a fully-connected net `(x,y,t) → (Ez,Bx,By)` with Fourier Feature Encoding, trained by minimizing Maxwell PDE residuals + PEC boundary conditions + initial condition.
- **FDTD** (`fdtd.py`): a 2nd-order Yee-scheme solver with CFL condition, heterogeneous materials (ε_r, μ_r, σ), PEC, graded PML, Gaussian pulse sources, and verified energy conservation.
- **Comparison** (`comparison.py`): evaluates the PINN on the FDTD's exact mesh/times and computes L2/RMS/max error, error maps, time series, compute time/memory/iterations.

All source lives under `pinn-maxwell/`. The frontend (`pinn-maxwell/demo/`) is a framework-free CAD-style UI (Three.js viewport + vanilla JS) served by the FastAPI backend.

## Running

```bash
pip install -r requirements.txt
cd pinn-maxwell
uvicorn server:app --port 8000
# → http://localhost:8000/demo
```

On Windows, `start_demo.bat` does the same (tries `py` then `python`, opens `http://localhost:8000/`).

Docker: `Dockerfile` runs from `/app/pinn-maxwell` on port 7860 (`ENV PORT=7860`), used for the Hugging Face Space deployment (see `README.md` frontmatter: `sdk: docker`, `app_port: 7860`).

CPU-only environment — no GPU assumed by default; GPU config is exposed via `/api/gpu` but not required.

## Verifying components without the server

Each core module is independently runnable as a self-check/smoke test:

```bash
cd pinn-maxwell
python fdtd.py          # O(dx²) convergence vs analytical solution + energy conservation
python comparison.py    # comparison pipeline validated against the exact solution
python exporters.py     # exercises all export formats
python main.py --train  # full PINN training run (Adam + L-BFGS)
```

There is no separate test suite (no `pytest`/`tests/` dir) — these module-level `__main__` checks are the verification mechanism. Run the relevant one after touching its module.

## Architecture

### Physics/solver layer (`pinn-maxwell/*.py`)

- `model.py` — FCN + Fourier Feature Encoding architecture for the PINN.
- `physics.py` — Maxwell PDE residuals (via autograd) and the PINN loss function (PDE + BC + IC terms).
- `sampling.py` — collocation / boundary / initial-condition point sampling for training.
- `train.py` — training loop: Adam then L-BFGS.
- `analytical.py` — exact TM_mn analytical solution, used to independently validate both the FDTD solver and the PINN.
- `fdtd.py` — Yee-scheme 2D TM FDTD solver (materials, PML, sources, energy tracking). `FDTDConfig` is the config object also built from the frontend's project JSON (see `server.py:project_to_fdtd_config`).
- `comparison.py` — runs PINN and FDTD on identical mesh/time grid and computes error metrics.
- `metrics.py`, `sensitivity.py`, `validation.py`, `higher_mode.py`, `visualization.py` — supporting analysis/plotting utilities.
- `exporters.py` — CSV, VTK (ParaView), MATLAB (.mat), NumPy (.npz), HDF5, PNG, GIF, PDF, STL export.
- `main.py` — CLI entry point for a full train/evaluate pipeline (`--train` flag).

### Backend (`pinn-maxwell/server.py`)

FastAPI app. Key concepts:
- **Jobs**: `/api/run` (`mode: pinn|fdtd|both`) and `/api/train` launch background jobs via `_launch`/`_run_job`; poll status with `/api/jobs/{id}`.
- **Live events**: `/ws/events` WebSocket streams console/progress/loss events (`emit`/`log` helpers); `/api/events` is a polling fallback.
- **Results**: persisted via `store.py` (`_get_result`/`_store_result` delegate to it) — job/result metadata in SQLite, snapshot arrays as `.npz` files, so both survive a server restart. Read back through `/api/results/{id}/meta` (metrics/series/times) and `/api/results/{id}/frames/{field}` (binary float32 frames).
- **Project → config**: the frontend's project JSON (geometry, materials, sources, domain) is translated to `FDTDConfig` by `project_to_fdtd_config`.
- **Export/tools**: `/api/export` (10 formats via `exporters.py`), `/api/tools/mesh`, `/api/tools/refine` (Richardson extrapolation), `/api/gpu`, `/api/slice`, `/api/residuals`.

### Persistence (`pinn-maxwell/store.py`)

SQLite (stdlib) + `.npz` files, not a project library — see `store.py` module docstring. Data directory resolution order: `FLUXIA_DATA_DIR` env var → `/data` if it exists and is writable (Hugging Face Spaces "Persistent Storage" mount point) → `pinn-maxwell/data/` inside the container (ephemeral on HF Spaces without the paid addon). Result retention is a size-based sweep (`FLUXIA_MAX_RESULTS`, default 200), not a hard cap — old results are deleted (row + `.npz`) once the threshold is exceeded, not silently evicted on every write. A small in-process LRU cache in `store.py` avoids re-reading `.npz` files on every animation-frame scrub. On startup, jobs left `queued`/`running` from a previous crash are reconciled to `interrupted`.

### Frontend (`pinn-maxwell/demo/`)

No build step — plain HTML/CSS/JS. Loaded at `/demo` (or `/`).
- `js/app.js` — startup/orchestration.
- `js/api.js` — REST + WebSocket client.
- `js/state.js` — central state (project, selection, frames).
- `js/viewport.js` — Three.js 3D scene (domain, mesh, materials, field surface, PML, isolines, XY/XZ/YZ slices).
- `js/ui.js` — project tree, properties panel, console, menus.
- `js/panels.js` — training panel, 2×2 PINN/FDTD/error/metrics comparison view, plots, playback controls.
- `js/charts.js` — canvas-only line charts, heatmaps, FFT (no chart library dependency).

The core interaction: run PINN and/or FDTD via the menu (Simulación → Ejecutar ambos), then the comparison tab shows 4 synchronized panels (PINN | FDTD | error map | metrics) driven by a shared time slider.

## Conventions

- User-facing strings (UI, README, comments in some modules) are in Spanish; keep this consistent when editing `demo/` or docs.
- Physical units are natural units (c = ε₀ = μ₀ = 1).
- When changing `FDTDConfig` fields or the project JSON schema, update both `project_to_fdtd_config` in `server.py` and the frontend's project model (`demo/js/state.js`) together — they must stay in sync.
