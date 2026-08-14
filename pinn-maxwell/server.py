"""
server.py — Backend FastAPI del entorno profesional de simulación
             "Fluxia" — Maxwell 2D TM: PINN vs FDTD.

Arquitectura:
    - Carga del PINN entrenado (model.py + results/maxwell_pinn.pth)
    - Solver FDTD 2D TM (fdtd.py) — mismo problema físico que el PINN
    - Sistema de jobs en hilos con consola en streaming (WS + polling)
    - Entrenamiento PINN en vivo con métricas por época
    - Resultados servidos como binario float32 (frames de animación)
    - Exportación: CSV, VTK, MATLAB, NumPy, HDF5, PNG, GIF, PDF, STL

Ejecución:
    cd pinn-maxwell
    uvicorn server:app --reload --port 8000
    → http://localhost:8000/demo
"""
from __future__ import annotations

import os, sys, math, json, time, uuid, threading, platform
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FCN
from analytical import Ez_exact, Bx_exact, By_exact, cavity_mode
from fdtd import FDTDConfig, FDTDSolver
import comparison as cmp_mod
import exporters
import store

# ─── Parámetros de arquitectura del PINN (coinciden con el entrenamiento) ────
ARCH    = [3, 128, 128, 128, 128, 3]
FOURIER = True
N_F     = 32
SIGMA   = 1.5
L       = 1.0
T_MAX   = 2.828427            # 2 periodos TM₁₁

APP_VERSION = "1.0.0"
APP_NAME    = "Fluxia — Maxwell PINN vs FDTD"

# ═════════════════════════════════════════════════════════════════════════════
# BUS DE EVENTOS (consola / progreso / entrenamiento)
# ═════════════════════════════════════════════════════════════════════════════
_EVENTS: deque = deque(maxlen=5000)
_EVENT_SEQ = {"n": 0}
_EVENT_LOCK = threading.Lock()

def emit(kind: str, **payload):
    """Publica un evento para la consola / paneles de la UI."""
    with _EVENT_LOCK:
        _EVENT_SEQ["n"] += 1
        _EVENTS.append({"seq": _EVENT_SEQ["n"], "kind": kind,
                        "ts": time.time(), **payload})

def log(msg: str, level: str = "info", job: str | None = None):
    emit("log", level=level, msg=msg, job=job)
    print(f"[{level.upper()}] {msg}")


# ═════════════════════════════════════════════════════════════════════════════
# MODELO PINN
# ═════════════════════════════════════════════════════════════════════════════
_model = None
_device = "cpu"
_model_lock = threading.Lock()

def _locate_model():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, "results", "maxwell_pinn.pth"),
              os.path.join(base, "..", "results", "maxwell_pinn.pth")]:
        if os.path.exists(p):
            return os.path.normpath(p)
    return None

def _load_model():
    global _model, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)   # reproduce la matriz B del Fourier Encoding
    m = FCN(ARCH, fourier_features=FOURIER, n_fourier=N_F, sigma=SIGMA)
    pth = _locate_model()
    if pth:
        m.load_state_dict(torch.load(pth, map_location=_device, weights_only=True))
        log(f"PINN cargado: {pth}")
    else:
        log("maxwell_pinn.pth no encontrado — PINN sin entrenar (usa Entrenar)", "warn")
    m.to(_device).eval()
    _model = m

@torch.no_grad()
def _infer(x_np, y_np, t_np):
    """Evaluación batch del PINN. Retorna (Ez, Bx, By) 1D float64."""
    if _model is None:
        raise HTTPException(status_code=503, detail="PINN no cargado")
    to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32).view(-1, 1).to(_device)
    with _model_lock:
        out = _model(to(x_np), to(y_np), to(t_np))
    return tuple(o.cpu().numpy().astype(np.float64).flatten() for o in out)

def _pinn_meta():
    n_params = sum(p.numel() for p in _model.parameters() if p.requires_grad) if _model else 0
    return {"epochs": 12000, "params": int(n_params),
            "memory_bytes": int(n_params * 4)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    store.reconcile_startup()
    log(f"Persistencia en: {store.data_dir()}")
    log(f"CORS — orígenes cruzados permitidos: {_CORS_ORIGINS or '(ninguno; solo same-origin)'}")
    try:
        _load_model()
    except Exception as exc:
        log(f"No se pudo cargar el PINN: {exc}", "error")
    log(f"{APP_NAME} v{APP_VERSION} — dispositivo: {_device.upper()}")
    log("UI disponible en http://localhost:8000/demo")
    yield

# El frontend (demo/js/api.js: BASE = "") es same-origin — no depende de CORS
# para funcionar. Un allow_origins=["*"] sin auth deja que CUALQUIER sitio web
# que un visitante tenga abierto haga que su navegador dispare requests contra
# este servidor (lanzar entrenamientos, cancelar jobs, cambiar hilos de
# PyTorch) sin que se entere. Por defecto no se permite ningún origen cruzado;
# FLUXIA_CORS_ORIGINS permite habilitar orígenes específicos (coma-separados)
# si se quiere consumir la API desde un dashboard/frontend separado.
_CORS_ORIGINS = [o.strip() for o in
                 os.environ.get("FLUXIA_CORS_ORIGINS", "").split(",") if o.strip()]

app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=_CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])

_demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
if os.path.isdir(_demo_path):
    app.mount("/demo", StaticFiles(directory=_demo_path, html=True), name="demo")


# ═════════════════════════════════════════════════════════════════════════════
# PROYECTO → CONFIGURACIÓN FDTD
# ═════════════════════════════════════════════════════════════════════════════
DEFAULT_PROJECT = {
    "name": "Cavidad TM11",
    "version": APP_VERSION,
    "domain": {"L": 1.0, "Nx": 101, "Ny": 101, "T_max": T_MAX,
               "courant": 0.5, "n_snapshots": 60},
    "materials": [
        {"id": "air",     "name": "Aire",    "eps_r": 1.0,  "mu_r": 1.0, "sigma": 0.0,
         "color": "#9aa3ad", "dispersive": "none"},
        {"id": "pec",     "name": "PEC",     "eps_r": 1.0,  "mu_r": 1.0, "sigma": 1e7,
         "color": "#e8833a", "dispersive": "none", "pec": True},
        {"id": "silicon", "name": "Silicio", "eps_r": 11.7, "mu_r": 1.0, "sigma": 0.0,
         "color": "#3a76e8", "dispersive": "none"},
    ],
    "geometry": [
        {"id": "domain-box", "name": "Caja (dominio)", "shape": "domain",
         "material": "air", "enabled": True},
        {"id": "sphere-1", "name": "Esfera", "shape": "sphere", "material": "silicon",
         "cx": 0.68, "cy": 0.32, "r": 0.12, "enabled": False},
        {"id": "antenna-1", "name": "Antena", "shape": "box", "material": "pec",
         "x0": 0.47, "y0": 0.15, "x1": 0.53, "y1": 0.45, "enabled": False},
    ],
    "sources": [
        {"id": "mode-tm11", "name": "Modo TM11 (IC)", "type": "mode",
         "enabled": True, "E0": 1.0, "m": 1, "n": 1},
        {"id": "pulse-1", "name": "Pulso Gaussiano", "type": "pulse",
         "enabled": False, "x": 0.5, "y": 0.5, "amplitude": 5.0,
         "freq": 1.5, "t0": 0.5, "tau": 0.15, "polarization": "Ez",
         "kind": "pulse"},
    ],
    "monitors": [
        {"id": "mon-xy", "name": "Plano XY", "plane": "xy", "position": 0.5, "enabled": True},
        {"id": "mon-xz", "name": "Plano XZ", "plane": "xz", "position": 0.5, "enabled": True},
        {"id": "probe-1", "name": "Sonda centro", "plane": "point", "x": 0.5, "y": 0.5,
         "enabled": True},
    ],
    "pml": {"enabled": False, "cells": 8, "order": 3},
    "solver_pinn": {"arch": ARCH, "fourier": FOURIER, "n_fourier": N_F,
                    "sigma": SIGMA, "epochs": 12000, "lr": 1e-3},
    "solver_fdtd": {"scheme": "Yee 2D TM", "boundary": "PEC"},
}

def project_to_fdtd_config(project: dict) -> FDTDConfig:
    dom = project.get("domain", {})
    mats = {m["id"]: m for m in project.get("materials", [])}
    objects = []
    for g in project.get("geometry", []):
        if g.get("shape") in ("sphere", "circle", "box") and g.get("enabled", True):
            mat = mats.get(g.get("material", "air"), {})
            o = {k: g[k] for k in ("shape", "cx", "cy", "r", "x0", "y0", "x1", "y1")
                 if k in g}
            o.update({"eps_r": float(mat.get("eps_r", 1.0)),
                      "mu_r": float(mat.get("mu_r", 1.0)),
                      "sigma": float(mat.get("sigma", 0.0)),
                      "pec": bool(mat.get("pec", False)),
                      "enabled": True})
            objects.append(o)

    init_mode, E0, source = "zero", 1.0, None
    for s in project.get("sources", []):
        if not s.get("enabled", True):
            continue
        if s.get("type") == "mode":
            init_mode = "tm11"
            E0 = float(s.get("E0", 1.0))
        elif s.get("type") == "pulse":
            source = {"enabled": True, "x": s.get("x", .5), "y": s.get("y", .5),
                      "amplitude": s.get("amplitude", 1.0), "freq": s.get("freq", 1.5),
                      "t0": s.get("t0", .5), "tau": s.get("tau", .15),
                      "kind": s.get("kind", "pulse")}

    probes = [{"x": m.get("x", .5), "y": m.get("y", .5), "name": m.get("name", "sonda")}
              for m in project.get("monitors", [])
              if m.get("plane") == "point" and m.get("enabled", True)]
    if not probes:
        probes = [{"x": 0.5, "y": 0.5, "name": "centro"}]

    pml_conf = project.get("pml", {})
    pml = ({"cells": int(pml_conf.get("cells", 8)),
            "order": float(pml_conf.get("order", 3))}
           if pml_conf.get("enabled") else None)

    return FDTDConfig(
        L=float(dom.get("L", 1.0)),
        Nx=int(dom.get("Nx", 101)), Ny=int(dom.get("Ny", 101)),
        T_max=float(dom.get("T_max", T_MAX)),
        courant=float(dom.get("courant", 0.5)),
        objects=objects, source=source, init_mode=init_mode, E0=E0,
        pml=pml, probes=probes,
        n_snapshots=int(dom.get("n_snapshots", 60)))


# ═════════════════════════════════════════════════════════════════════════════
# SISTEMA DE JOBS Y ALMACÉN DE RESULTADOS (persistidos vía store.py)
# ═════════════════════════════════════════════════════════════════════════════
# Fast-path de cancelación: los callbacks de progreso de FDTD/PINN/entrenamiento
# se llaman cientos/miles de veces por simulación, así que no pueden pagar un
# round-trip a SQLite por paso. No necesita sobrevivir a un reinicio: un job
# interrumpido por caída del proceso ya queda marcado "interrupted" al arrancar
# (store.reconcile_startup).
_cancel_flags: dict[str, bool] = {}

def _store_result(res: dict) -> str:
    return store.result_store(res)

def _run_job(job_id: str, jtype: str, target, *args):
    try:
        store.job_set_status(job_id, "running")
        emit("job", job=job_id, status="running", jtype=jtype)
        rid = target(job_id, *args)
        store.job_set_status(job_id, "done", result_id=rid)
        emit("job", job=job_id, status="done", jtype=jtype, result_id=rid)
    except Exception as exc:
        import traceback; traceback.print_exc()
        store.job_set_status(job_id, "error", error=str(exc))
        log(f"Job {jtype} falló: {exc}", "error", job=job_id)
        emit("job", job=job_id, status="error", jtype=jtype, error=str(exc))
    finally:
        _cancel_flags.pop(job_id, None)

def _launch(jtype: str, target, *args) -> str:
    job_id = uuid.uuid4().hex[:10]
    store.job_create(job_id, jtype)
    _cancel_flags[job_id] = False
    threading.Thread(target=_run_job, args=(job_id, jtype, target) + args,
                     daemon=True).start()
    return job_id


# ─── Trabajos de simulación ──────────────────────────────────────────────────
def _job_log(job_id):
    return lambda m: log(m, "info", job=job_id)

def _pinn_probe_series(cfg: FDTDConfig, n=400):
    """Serie temporal fina Ez(t) del PINN en las sondas."""
    ts = np.linspace(0.0, cfg.T_max, n).astype(np.float32)
    out = {}
    for p in cfg.probes:
        xs = np.full(n, p["x"], dtype=np.float32)
        ys = np.full(n, p["y"], dtype=np.float32)
        Ez, _, _ = _infer(xs, ys, ts)
        out[p.get("name", "sonda")] = Ez.tolist()
    return ts.tolist(), out

def _target_fdtd(job_id: str, project: dict) -> str:
    cfg = project_to_fdtd_config(project)
    lg = _job_log(job_id)
    lg("Compilando geometría y materiales…")
    lg(f"Generando malla {cfg.Nx}×{cfg.Ny}  (dx={cfg.dx:.4f}, CFL dt={cfg.dt:.5f})")
    res = cmp_mod.compare(cfg, None, log=lg,
                          cancel_cb=lambda: _cancel_flags.get(job_id, False))
    lg("Guardando resultados FDTD…")
    res["mode"] = "fdtd"
    res["project"] = project
    return _store_result(res)

def _target_pinn(job_id: str, project: dict) -> str:
    cfg = project_to_fdtd_config(project)
    lg = _job_log(job_id)
    lg("Inicializando PINN…")
    times = np.linspace(0.0, cfg.T_max, cfg.n_snapshots)
    lg(f"Evaluando PINN en {len(times)} instantes × {cfg.Nx*cfg.Ny} nodos…")
    Ez_p, Bx_p, By_p, wall = cmp_mod.eval_pinn_on_grid(
        _infer, cfg.Nx, cfg.Ny, cfg.L, times,
        eps_r=cfg.eps_r_bg, mu_r=cfg.mu_r_bg,
        progress_cb=lambda k, K: (_cancel_flags.get(job_id, False) or
                                  (k % max(1, K // 10) == 0 and lg(f"[PINN] instante {k}/{K}"))))
    energy = cmp_mod.energy_series(Ez_p, Bx_p, By_p, cfg.dx, cfg.dy)
    probe_t, probes = _pinn_probe_series(cfg)
    lg(f"PINN completado en {wall:.2f}s")
    res = {
        "mode": "pinn",
        "times": times.tolist(),
        "grid": {"Nx": cfg.Nx, "Ny": cfg.Ny, "L": cfg.L,
                 "dx": cfg.dx, "dy": cfg.dy, "dt": cfg.dt},
        "snapshots": {"Ez_pinn": Ez_p, "Bx_pinn": Bx_p, "By_pinn": By_p},
        "pinn": {"wall_time": wall, "energy_t": times.tolist(),
                 "energy": energy.tolist(), "probe_t": probe_t, "probes": probes},
        "metrics": {"pinn_wall_time": wall, **{f"pinn_{k}": v
                    for k, v in _pinn_meta().items()}},
        "project": project,
    }
    return _store_result(res)

def _target_both(job_id: str, project: dict) -> str:
    cfg = project_to_fdtd_config(project)
    lg = _job_log(job_id)
    lg("Compilando geometría y materiales…")
    lg(f"Generando malla {cfg.Nx}×{cfg.Ny}…")
    lg("Inicializando PINN…")
    res = cmp_mod.compare(cfg, _infer, pinn_meta=_pinn_meta(), log=lg,
                          cancel_cb=lambda: _cancel_flags.get(job_id, False))
    lg("Comparando con FDTD…")
    probe_t, probes = _pinn_probe_series(cfg)
    res.setdefault("pinn", {})["probe_t"] = probe_t
    res["pinn"]["probes"] = probes
    res["mode"] = "both"
    res["project"] = project
    lg("Guardando resultados…")
    return _store_result(res)


# ─── Entrenamiento PINN en vivo ──────────────────────────────────────────────
# _train_lock evita que dos POST /api/train concurrentes pasen ambos el chequeo
# "active" antes de que el hilo en background alcance a marcarlo (TOCTOU): el
# endpoint marca active=True de forma sincrónica, con el lock, antes de lanzar
# el job — ver api_train().
_TRAIN = {"active": False, "cancel": False}
_train_lock = threading.Lock()

def _target_train(job_id: str, opts: dict) -> str:
    from physics import total_loss
    from sampling import sample_interior, sample_boundary, sample_initial

    epochs   = int(opts.get("epochs", 1000))
    lr       = float(opts.get("lr", 1e-3))
    n_col    = int(opts.get("n_collocation", 3000))
    n_bc     = int(opts.get("n_boundary", 150))
    n_ic     = int(opts.get("n_initial", 1500))
    resample = int(opts.get("resample_every", 200))
    reset    = bool(opts.get("reset", False))
    save     = bool(opts.get("save", True))

    lg = _job_log(job_id)
    # _TRAIN["active"]/["cancel"] ya fueron marcados de forma atómica por
    # api_train() antes de lanzar este job — no se tocan acá salvo para
    # liberarlos en el finally, así un crash a mitad de entrenamiento no deja
    # el flag trabado en True para siempre.
    try:
        global _model
        with _model_lock:
            if reset or _model is None:
                torch.manual_seed(42)
                model = FCN(ARCH, fourier_features=FOURIER, n_fourier=N_F, sigma=SIGMA).to(_device)
                lg("Modelo reinicializado (Xavier)")
            else:
                model = _model
            model.train()

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9998)

        lg(f"Entrenando: {epochs} épocas, lr={lr}, colocación={n_col}, BC={n_bc}×4, IC={n_ic}")
        x_c, y_c, t_c = sample_interior(N=n_col, device=_device)
        bp = sample_boundary(N_per_side=n_bc, device=_device)
        x_i, y_i, t_i = sample_initial(N=n_ic, device=_device)

        t0 = time.perf_counter()
        hist = {"epoch": [], "loss_total": [], "loss_pde": [], "loss_bc": [],
                "loss_ic": [], "lr": [], "elapsed": []}

        for ep in range(1, epochs + 1):
            if _TRAIN["cancel"] or _cancel_flags.get(job_id, False):
                lg(f"Entrenamiento cancelado en época {ep}")
                break
            if ep % resample == 0:
                x_c, y_c, t_c = sample_interior(N=n_col, device=_device)
                bp = sample_boundary(N_per_side=n_bc, device=_device)
                x_i, y_i, t_i = sample_initial(N=n_ic, device=_device)

            opt.zero_grad()
            L_t, L_p, L_b, L_i, _ = total_loss(model, x_c, y_c, t_c, bp, x_i, y_i, t_i)
            L_t.backward()
            opt.step()
            sched.step()

            if ep % max(1, min(20, epochs // 100 or 1)) == 0 or ep == 1 or ep == epochs:
                el = time.perf_counter() - t0
                cur_lr = opt.param_groups[0]["lr"]
                emit("train", job=job_id, epoch=ep, epochs=epochs,
                     loss_total=float(L_t), loss_pde=float(L_p),
                     loss_bc=float(L_b), loss_ic=float(L_i),
                     lr=float(cur_lr), elapsed=el)
                hist["epoch"].append(ep)
                hist["loss_total"].append(float(L_t))
                hist["loss_pde"].append(float(L_p))
                hist["loss_bc"].append(float(L_b))
                hist["loss_ic"].append(float(L_i))
                hist["lr"].append(float(cur_lr))
                hist["elapsed"].append(el)
            if ep % 100 == 0:
                lg(f"Epoch {ep}/{epochs}  Total={float(L_t):.3e}  PDE={float(L_p):.3e}  "
                   f"BC={float(L_b):.3e}  IC={float(L_i):.3e}")

        model.eval()
        with _model_lock:
            _model = model

        if save:
            base = os.path.dirname(os.path.abspath(__file__))
            outp = os.path.join(base, "results", "maxwell_pinn_live.pth")
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            torch.save(model.state_dict(), outp)
            lg(f"Modelo guardado: {outp}")

        return _store_result({"mode": "training", "history": hist,
                              "wall_time": time.perf_counter() - t0})
    finally:
        _TRAIN["active"] = False


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — básicos
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/")
def home():
    return RedirectResponse(url="/demo/")

@app.get("/api/health")
def health():
    return {"status": "online", "app": APP_NAME, "version": APP_VERSION,
            "model_loaded": _model is not None, "device": _device,
            "T_MAX": T_MAX, "training_active": _TRAIN["active"]}

@app.get("/api/version")
def version():
    return {"app": APP_NAME, "version": APP_VERSION,
            "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "solver_fdtd": "Yee 2D TM leapfrog, orden 2",
            "solver_pinn": f"FCN {ARCH} + Fourier(n={N_F}, σ={SIGMA})"}

@app.get("/api/project/default")
def project_default():
    return DEFAULT_PROJECT

class ProjectReq(BaseModel):
    project: dict

@app.post("/api/project/validate")
def project_validate(r: ProjectReq):
    """Valida un proyecto y retorna la configuración FDTD derivada (CFL, pasos…)."""
    try:
        cfg = project_to_fdtd_config(r.project)
        return {"ok": True, "config": cfg.to_dict()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — eventos y jobs
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/api/events")
def events(since: int = 0):
    with _EVENT_LOCK:
        evs = [e for e in _EVENTS if e["seq"] > since]
    return {"events": evs, "last": _EVENT_SEQ["n"]}

@app.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    import asyncio
    await ws.accept()
    cursor = _EVENT_SEQ["n"]
    try:
        while True:
            with _EVENT_LOCK:
                evs = [e for e in _EVENTS if e["seq"] > cursor]
            if evs:
                cursor = evs[-1]["seq"]
                await ws.send_text(json.dumps({"events": evs}))
            await asyncio.sleep(0.2)
    except (WebSocketDisconnect, RuntimeError):
        pass

class RunReq(BaseModel):
    mode: str = "both"          # "pinn" | "fdtd" | "both"
    project: dict | None = None

@app.post("/api/run")
def api_run(r: RunReq):
    project = r.project or DEFAULT_PROJECT
    if r.mode == "fdtd":
        jid = _launch("fdtd", _target_fdtd, project)
    elif r.mode == "pinn":
        if _model is None:
            raise HTTPException(503, "PINN no cargado")
        jid = _launch("pinn", _target_pinn, project)
    else:
        if _model is None:
            raise HTTPException(503, "PINN no cargado")
        jid = _launch("both", _target_both, project)
    log(f"Simulación '{r.mode}' lanzada (job {jid})")
    return {"job_id": jid}

class TrainReq(BaseModel):
    epochs: int = 1000
    lr: float = 1e-3
    n_collocation: int = 3000
    n_boundary: int = 150
    n_initial: int = 1500
    resample_every: int = 200
    reset: bool = False
    save: bool = True

@app.post("/api/train")
def api_train(r: TrainReq):
    with _train_lock:
        if _TRAIN["active"]:
            raise HTTPException(409, "Ya hay un entrenamiento activo — esperá a que termine "
                                      "antes de lanzar otro.")
        _TRAIN["active"] = True
        _TRAIN["cancel"] = False
    try:
        jid = _launch("train", _target_train, r.model_dump())
    except Exception:
        _TRAIN["active"] = False
        raise
    log(f"Entrenamiento PINN lanzado (job {jid}, {r.epochs} épocas)")
    return {"job_id": jid}

@app.post("/api/train/stop")
def api_train_stop():
    _TRAIN["cancel"] = True
    return {"ok": True}

@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = store.job_get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    return {k: job[k] for k in ("id", "type", "status", "result_id", "error")}

@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str):
    job = store.job_get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado")
    store.job_set_cancel(job_id)
    _cancel_flags[job_id] = True
    return {"ok": True}


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — resultados
# ═════════════════════════════════════════════════════════════════════════════
def _get_result(rid: str) -> dict:
    res = store.result_get_full(rid)
    if res is None:
        raise HTTPException(404, "Resultado no encontrado (puede haber expirado)")
    return res

@app.get("/api/results/{rid}/meta")
def result_meta(rid: str):
    meta = store.result_get_meta(rid)
    if meta is None:
        raise HTTPException(404, "Resultado no encontrado (puede haber expirado)")
    return meta

@app.get("/api/results/{rid}/frames/{field}")
def result_frames(rid: str, field: str):
    """Frames como binario float32 little-endian, shape (K, Nx, Ny)."""
    arr = store.result_get_frames(rid, field)
    if arr is None:
        raise HTTPException(404, f"Campo '{field}' no disponible para el resultado '{rid}'")
    arr = np.ascontiguousarray(arr, dtype="<f4")
    K, Nx, Ny = arr.shape
    return Response(content=arr.tobytes(),
                    media_type="application/octet-stream",
                    headers={"X-Shape": f"{K},{Nx},{Ny}",
                             "Access-Control-Expose-Headers": "X-Shape"})

@app.get("/api/results")
def results_list():
    return {"results": store.result_list()}


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — exportación
# ═════════════════════════════════════════════════════════════════════════════
class ExportReq(BaseModel):
    result_id: str
    format: str = "csv"          # csv|vtk|mat|npz|hdf5|png|gif|pdf|stl|csv-series
    field: str = "Ez_fdtd"
    frame: int = 0
    fps: int = 12

def _dl(content: bytes, filename: str, mime: str):
    return Response(content=content, media_type=mime,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})

@app.post("/api/export")
def api_export(r: ExportReq):
    res = _get_result(r.result_id)
    snaps = res.get("snapshots", {})
    times = res.get("times", [0.0])
    Lg = res.get("grid", {}).get("L", 1.0)
    fmt = r.format.lower()
    field = r.field if r.field in snaps else (next(iter(snaps)) if snaps else None)
    k = min(max(r.frame, 0), len(times) - 1)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if fmt == "csv":
        if field is None: raise HTTPException(400, "Sin campos")
        return _dl(exporters.field_to_csv(snaps[field][k], Lg, field, times[k]),
                   f"{field}_t{times[k]:.3f}_{stamp}.csv", "text/csv")

    if fmt == "csv-series":
        cols = {"t": res.get("error_series", {}).get("t", times)}
        es = res.get("error_series")
        if es:
            cols.update({"err_l2_pct": es["l2"], "err_rms_pct": es["rms"],
                         "err_max_pct": es["max"]})
        fd = res.get("fdtd", {})
        if fd.get("energy"):
            cols_e = {"t": fd["energy_t"], "energia_fdtd": fd["energy"]}
            if not es:
                cols = cols_e
            else:
                # exportar series de error; energía va aparte si difieren longitudes
                pass
        return _dl(exporters.series_to_csv(cols), f"series_{stamp}.csv", "text/csv")

    if fmt == "vtk":
        fields = {name: snaps[name][k] for name in snaps}
        return _dl(exporters.to_vtk(fields, Lg, t=times[k]),
                   f"maxwell_t{times[k]:.3f}_{stamp}.vtk", "application/octet-stream")

    if fmt in ("mat", "npz", "hdf5"):
        data = {name: snaps[name] for name in snaps}
        data["times"] = np.asarray(times)
        fd = res.get("fdtd", {})
        if fd.get("energy"):
            data["energy_t"] = np.asarray(fd["energy_t"])
            data["energy_fdtd"] = np.asarray(fd["energy"])
        es = res.get("error_series")
        if es:
            data["err_l2_pct"] = np.asarray(es["l2"])
        if fmt == "mat":
            return _dl(exporters.to_mat(data), f"maxwell_{stamp}.mat",
                       "application/octet-stream")
        if fmt == "npz":
            return _dl(exporters.to_npz(data), f"maxwell_{stamp}.npz",
                       "application/octet-stream")
        return _dl(exporters.to_hdf5(data, {"app": APP_NAME, "version": APP_VERSION}),
                   f"maxwell_{stamp}.h5", "application/octet-stream")

    if fmt == "png":
        if field is None: raise HTTPException(400, "Sin campos")
        if "Ez_pinn" in snaps and "Ez_fdtd" in snaps:
            png = exporters.comparison_png(snaps["Ez_pinn"][k], snaps["Ez_fdtd"][k],
                                           Lg, times[k])
        else:
            png = exporters.field_png(snaps[field][k], Lg, field, times[k])
        return _dl(png, f"maxwell_t{times[k]:.3f}_{stamp}.png", "image/png")

    if fmt == "gif":
        if field is None: raise HTTPException(400, "Sin campos")
        return _dl(exporters.animation_gif(np.asarray(snaps[field]), times, Lg,
                                           field, fps=r.fps),
                   f"{field}_{stamp}.gif", "image/gif")

    if fmt == "pdf":
        return _dl(exporters.technical_pdf(res, res.get("project")),
                   f"informe_maxwell_{stamp}.pdf", "application/pdf")

    if fmt == "stl":
        proj = res.get("project") or DEFAULT_PROJECT
        objs = [g for g in proj.get("geometry", [])
                if g.get("shape") in ("sphere", "circle", "box") and g.get("enabled")]
        return _dl(exporters.geometry_stl(objs, Lg), f"geometria_{stamp}.stl",
                   "application/octet-stream")

    raise HTTPException(400, f"Formato no soportado: {fmt}")

class StlReq(BaseModel):
    project: dict

@app.post("/api/export/stl")
def api_export_stl(r: StlReq):
    """STL directo desde el proyecto actual (sin necesidad de resultados)."""
    objs = [g for g in r.project.get("geometry", [])
            if g.get("shape") in ("sphere", "circle", "box") and g.get("enabled")]
    Lg = r.project.get("domain", {}).get("L", 1.0)
    return _dl(exporters.geometry_stl(objs, Lg),
               f"geometria_{time.strftime('%Y%m%d_%H%M%S')}.stl",
               "application/octet-stream")


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — herramientas
# ═════════════════════════════════════════════════════════════════════════════
@app.get("/api/gpu")
def gpu_info():
    info = {"device": _device,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "cpu": platform.processor() or platform.machine()}
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info

class GpuReq(BaseModel):
    num_threads: int | None = None

@app.post("/api/gpu")
def gpu_config(r: GpuReq):
    # torch.set_num_threads(...) es un ajuste global del proceso — afecta a
    # TODAS las requests en curso, no solo a quien lo pide. No se puede aislar
    # por request sin un pool de threads por job (fuera de alcance), así que
    # al menos evitamos que pise una simulación/entrenamiento activo de otra
    # pestaña/usuario.
    if r.num_threads and 1 <= r.num_threads <= 64:
        if _cancel_flags:
            raise HTTPException(409, "Hay una simulación o entrenamiento en curso — "
                                      "esperá a que termine antes de cambiar los hilos "
                                      "de PyTorch (es una configuración global del proceso).")
        torch.set_num_threads(int(r.num_threads))
        log(f"Hilos de PyTorch: {r.num_threads}")
    return gpu_info()

class MeshReq(BaseModel):
    m: int = 1
    n: int = 1
    ppw: int = 20            # puntos por longitud de onda
    L: float = 1.0
    courant: float = 0.5
    T_max: float = T_MAX

@app.post("/api/tools/mesh")
def tool_mesh(r: MeshReq):
    """Generador de malla: Nx sugerido según puntos-por-longitud-de-onda."""
    mode = cavity_mode(max(r.m, 1), max(r.n, 1))
    lam = 2 * math.pi / mode["k"]
    Nx = max(11, int(math.ceil(r.ppw * r.L / lam)) + 1)
    if Nx % 2 == 0:
        Nx += 1  # impar → nodo central exacto
    dx = r.L / (Nx - 1)
    dt = r.courant * dx  # c=1, dx=dy
    n_steps = int(math.ceil(r.T_max / dt))
    return {"lambda": lam, "Nx": Nx, "Ny": Nx, "dx": dx, "dt": dt,
            "n_steps": n_steps, "omega": mode["omega"],
            "T_period": mode["T_period"], "ppw_effective": lam / dx}

class RefineReq(BaseModel):
    project: dict | None = None
    target_error_pct: float = 0.05
    T_test: float = 1.0

@app.post("/api/tools/refine")
def tool_refine(r: RefineReq):
    """
    Refinamiento adaptativo por extrapolación de Richardson:
    corre FDTD a N y 2N−1 en una ventana corta; el esquema es O(dx²),
    así que err(N) ≈ ‖u_N − u_2N‖/(1−1/4). Duplica hasta alcanzar el objetivo.
    """
    project = r.project or DEFAULT_PROJECT
    base_cfg = project_to_fdtd_config(project)
    study = []
    Nx = 41
    rec = None
    for _ in range(4):
        N2 = 2 * Nx - 1
        c1 = FDTDConfig(L=base_cfg.L, Nx=Nx, Ny=Nx, T_max=r.T_test,
                        courant=base_cfg.courant, objects=base_cfg.objects,
                        init_mode=base_cfg.init_mode, n_snapshots=5)
        c2 = FDTDConfig(L=base_cfg.L, Nx=N2, Ny=N2, T_max=r.T_test,
                        courant=base_cfg.courant, objects=base_cfg.objects,
                        init_mode=base_cfg.init_mode, n_snapshots=5)
        r1 = FDTDSolver(c1).run()
        r2 = FDTDSolver(c2).run()
        u1 = r1.Ez[-1]
        u2 = r2.Ez[-1][::2, ::2]      # submuestreo a la malla gruesa
        denom = np.linalg.norm(u2.ravel()) or 1.0
        err = float(np.linalg.norm((u1 - u2).ravel()) / denom / (1 - 0.25) * 100)
        study.append({"Nx": Nx, "err_est_pct": err,
                      "wall_time": r1.wall_time})
        if err <= r.target_error_pct and rec is None:
            rec = Nx
            break
        Nx = N2
    if rec is None:
        rec = 2 * study[-1]["Nx"] - 1
    return {"study": study, "recommended_Nx": rec,
            "target_error_pct": r.target_error_pct,
            "note": "Estimación por extrapolación de Richardson (esquema O(dx²))"}


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS — inferencia PINN en vivo (viewport) y diagnóstico
# ═════════════════════════════════════════════════════════════════════════════
class SliceReq(BaseModel):
    axis: str = "xy"
    position: float = 0.5
    t: float = 0.0
    N: int = 60
    eps_r: float = 1.0
    mu_r: float = 1.0

def _t_eff(t, eps_r, mu_r):
    return float(t) / math.sqrt(max(float(eps_r) * float(mu_r), 1e-9))

@app.post("/slice")
def api_slice(r: SliceReq):
    """Corte del campo PINN: XY (plano espacial), XZ/YZ (espacio-tiempo)."""
    if _model is None:
        raise HTTPException(503, "PINN no cargado")
    N = int(min(max(r.N, 10), 256))
    te = _t_eff(r.t, r.eps_r, r.mu_r)
    lin = np.linspace(0.0, L, N, dtype=np.float32)
    t_lin = np.linspace(0.0, T_MAX, N, dtype=np.float32)
    sp = float(r.position)

    if r.axis == "xy":
        Xg, Yg = np.meshgrid(lin, lin)
        xf, yf = Xg.flatten(), Yg.flatten()
        tf = np.full_like(xf, te)
    elif r.axis == "xz":
        Xg, Tg = np.meshgrid(lin, t_lin)
        xf = Xg.flatten()
        yf = np.full(N * N, sp, dtype=np.float32)
        tf = Tg.flatten()
    else:
        Tg, Yg = np.meshgrid(t_lin, lin)
        xf = np.full(N * N, sp, dtype=np.float32)
        yf = Yg.flatten()
        tf = Tg.flatten()

    Ez, Bx, By = _infer(xf, yf, tf)
    return {"Ez": Ez.reshape(N, N).tolist(),
            "Bx": Bx.reshape(N, N).tolist(),
            "By": By.reshape(N, N).tolist(), "t_eff": te}

class ResidualReq(BaseModel):
    t: float = 0.0
    N: int = 20

@app.post("/residuals")
def api_residuals(r: ResidualReq):
    """Residuales de Maxwell del PINN vía autograd (medias absolutas)."""
    if _model is None:
        raise HTTPException(503, "PINN no cargado")
    N = int(min(max(r.N, 8), 30))
    lin = np.linspace(0.05, 0.95, N, dtype=np.float32)
    Xg, Yg = np.meshgrid(lin, lin)
    x = torch.tensor(Xg.flatten(), dtype=torch.float32).view(-1, 1).to(_device).requires_grad_(True)
    y = torch.tensor(Yg.flatten(), dtype=torch.float32).view(-1, 1).to(_device).requires_grad_(True)
    t = torch.tensor(np.full(N * N, r.t, dtype=np.float32)).view(-1, 1).to(_device).requires_grad_(True)

    with _model_lock:
        Ez, Bx, By = _model(x, y, t)
    ones = torch.ones_like(Ez)
    ag = lambda o, i: torch.autograd.grad(o, i, ones, create_graph=False,
                                          retain_graph=True)[0]
    r_fx = (ag(Bx, t) + ag(Ez, y)).detach().cpu().numpy()
    r_fy = (ag(By, t) - ag(Ez, x)).detach().cpu().numpy()
    r_amp = (ag(Ez, t) - ag(By, x) + ag(Bx, y)).detach().cpu().numpy()
    r_gauss = (ag(Bx, x) + ag(By, y)).detach().cpu().numpy()
    return {"faraday_x": float(np.mean(np.abs(r_fx))),
            "faraday_y": float(np.mean(np.abs(r_fy))),
            "ampere": float(np.mean(np.abs(r_amp))),
            "gauss_b": float(np.mean(np.abs(r_gauss))),
            "total_mse": float(np.mean(r_fx**2 + r_fy**2 + r_amp**2 + r_gauss**2))}

class ResidualSeriesReq(BaseModel):
    n_t: int = 24
    N: int = 16

@app.post("/api/graphs/residual_series")
def api_residual_series(r: ResidualSeriesReq):
    """Residual PDE del PINN vs tiempo (para la pestaña Residual)."""
    if _model is None:
        raise HTTPException(503, "PINN no cargado")
    n_t = int(min(max(r.n_t, 4), 60))
    ts = np.linspace(0.0, T_MAX, n_t)
    out = {"t": ts.tolist(), "faraday_x": [], "faraday_y": [],
           "ampere": [], "gauss_b": [], "total": []}
    for tv in ts:
        res = api_residuals(ResidualReq(t=float(tv), N=r.N))
        out["faraday_x"].append(res["faraday_x"])
        out["faraday_y"].append(res["faraday_y"])
        out["ampere"].append(res["ampere"])
        out["gauss_b"].append(res["gauss_b"])
        out["total"].append(res["total_mse"])
    return out

@app.get("/api/training/history")
def api_training_history():
    """Historial de entrenamiento desde results/training_log.txt (si existe)."""
    import re
    base = os.path.dirname(os.path.abspath(__file__))
    history = []
    for lp in [os.path.join(base, "..", "results", "training_log.txt"),
               os.path.join(base, "results", "training_log.txt")]:
        lp = os.path.normpath(lp)
        if not os.path.exists(lp):
            continue
        try:
            with open(lp, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("[") and "Total:" in line:
                        parts = line.split("|")
                        m = re.search(r"(\d+)", parts[0])
                        history.append({
                            "epoch": int(m.group(1)) if m else 0,
                            "loss_total": float(parts[1].replace("Total:", "")),
                            "loss_pde": float(parts[2].replace("PDE:", "")),
                            "loss_bc": float(parts[3].replace("BC:", "")),
                            "loss_ic": float(parts[4].replace("IC:", "")),
                        })
        except Exception:
            pass
        break
    n_params = sum(p.numel() for p in _model.parameters()
                   if p.requires_grad) if _model else 0
    return {"history": history,
            "params": {"arch": str(ARCH), "fourier": FOURIER, "n_fourier": N_F,
                       "sigma": SIGMA, "n_params": int(n_params)}}
