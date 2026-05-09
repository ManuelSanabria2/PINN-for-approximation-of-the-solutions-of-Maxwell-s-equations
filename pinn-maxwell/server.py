"""
server.py — Backend FastAPI para la Cámara de Faraday Digital
Sirve predicciones del PINN Maxwell TM₁₁ en tiempo real al frontend Three.js.

Ejecución:
    cd pinn-maxwell
    pip install fastapi uvicorn
    uvicorn server:app --reload --port 8000

Frontend: http://localhost:8000/demo
"""
from __future__ import annotations

import os, sys, math
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    import torch
    import numpy as np
except Exception as exc:
    torch = None
    np = None
    _import_error = exc
else:
    _import_error = None

# Asegura importabilidad de model.py desde cualquier CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from model import FCN
except Exception as exc:
    FCN = None
    _model_import_error = exc
else:
    _model_import_error = None

# ─── Parámetros de arquitectura (deben coincidir con el entrenamiento) ────────
ARCH    = [3, 128, 128, 128, 128, 3]
FOURIER = True
N_F     = 32
SIGMA   = 1.5
L       = 1.0
T_MAX   = 2.828427          # 2 periodos TM₁₁ en unidades naturales

# ─── Localización del modelo entrenado ───────────────────────────────────────
def _locate_model():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [
        os.path.join(base, 'results', 'maxwell_pinn.pth'),
        os.path.join(base, '..', 'results', 'maxwell_pinn.pth'),
    ]:
        if os.path.exists(p):
            return os.path.normpath(p)
    raise FileNotFoundError(
        "maxwell_pinn.pth no encontrado. Ejecuta primero: python main.py --train")

# ─── Estado global del modelo ─────────────────────────────────────────────────
_model = _device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device
    if torch is None or FCN is None:
        _model = None
        _device = 'unavailable'
        reason = _import_error or _model_import_error
        print(f"\n[WARN] Dependencias PINN no disponibles. La demo usara modo analitico: {reason}")
        print("[OK] Demo disponible en       : http://localhost:8000/demo\n")
        yield
        return

    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        torch.manual_seed(42)       # reproduce la matriz B del Fourier Encoding
        _model = FCN(ARCH, fourier_features=FOURIER, n_fourier=N_F, sigma=SIGMA)
        pth = _locate_model()
        _model.load_state_dict(torch.load(pth, map_location=_device, weights_only=True))
        _model.to(_device).eval()
        print(f"\n[OK] PINN Maxwell TM11 cargado: {pth}")
    except Exception as exc:
        _model = None
        print(f"\n[WARN] No se pudo cargar el PINN. La demo usara modo analitico: {exc}")
    print(f"[OK] Dispositivo de inferencia: {_device.upper()}")
    print(f"[OK] Demo disponible en       : http://localhost:8000/demo\n")
    yield

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cámara de Faraday Digital — PINN Maxwell TM₁₁",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Servir el directorio demo/ como archivos estáticos en /demo
_demo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo')
if os.path.isdir(_demo_path):
    app.mount("/demo", StaticFiles(directory=_demo_path, html=True), name="demo")

# ─── Schemas Pydantic ─────────────────────────────────────────────────────────
class SliceReq(BaseModel):
    axis:     str   = "xy"   # "xy" | "xz" | "yz"
    position: float = 0.5    # posición del corte en el eje normal [0, 1]
    t:        float = 0.0    # tiempo actual [0, T_MAX]
    N:        int   = 60     # resolución de la malla (NxN)
    eps_r:    float = 1.0    # permitividad relativa del medio
    mu_r:     float = 1.0    # permeabilidad relativa del medio
    freq_scale: float = 1.0
    sigma_wall: float = 0.0

class VecReq(BaseModel):
    t:     float = 0.0
    N:     int   = 10        # rejilla NxN de vectores (N²  flechas)
    eps_r: float = 1.0
    mu_r:  float = 1.0
    freq_scale: float = 1.0
    sigma_wall: float = 0.0

# ─── Utilidades de Inferencia ─────────────────────────────────────────────────
def _t_eff(t: float, eps_r: float, mu_r: float, freq_scale: float = 1.0) -> float:
    """Escala el tiempo para simular cambio de ε_r y μ_r.
    La velocidad de fase es c/√(ε_r·μ_r), por tanto la onda 've'
    un tiempo efectivo t_eff = t / √(ε_r·μ_r).
    """
    freq = max(float(freq_scale), 0.05)
    return float(t) * freq / math.sqrt(max(float(eps_r) * float(mu_r), 1e-9))

def _damping(t: float, sigma_wall: float) -> float:
    sigma = max(float(sigma_wall), 0.0)
    return math.exp(-sigma * max(float(t), 0.0))

@torch.no_grad()
def _infer(x_np: np.ndarray, y_np: np.ndarray, t_np: np.ndarray):
    """Evaluación batch sin autograd. Retorna (Ez, Bx, By) como arrays 1D."""
    if _model is None or torch is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")
    to = lambda a: torch.tensor(a, dtype=torch.float32).view(-1, 1).to(_device)
    out = _model(to(x_np), to(y_np), to(t_np))
    return tuple(o.cpu().numpy().flatten() for o in out)

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return RedirectResponse(url="/demo/")


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "model": "Maxwell PINN TM₁₁",
        "model_loaded": _model is not None,
        "device": _device,
        "T_MAX": T_MAX,
        "sigma_fourier": SIGMA,
    }


@app.post("/slice")
def api_slice(r: SliceReq):
    """
    Evalúa los campos (Ez, Bx, By) en un plano de corte NxN.

    Convención de Textura (Three.js DataTexture, flipY=false):
      - row j=0  → V=0 (borde inferior del plano)
      - col i=0  → U=0 (borde izquierdo del plano)

    Pour cada modo:
      XY: U=x, V=y         → data[j][i] = campo(x=lin[i], y=lin[j], t=t_eff)
      XZ: U=x, V=t         → data[j][i] = campo(x=lin[i], y=pos,    t=t_lin[j])
      YZ: U=t, V=y         → data[j][i] = campo(x=pos,    y=lin[j], t=t_lin[i])
    """
    if _model is None or np is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")
    N    = int(min(max(r.N, 10), 100))
    te   = _t_eff(r.t, r.eps_r, r.mu_r, r.freq_scale)
    sp   = float(r.position)
    lin  = np.linspace(0.0, L, N, dtype=np.float32)
    t_lin = np.linspace(0.0, T_MAX, N, dtype=np.float32)
    fac  = math.sqrt(max(float(r.eps_r) * float(r.mu_r), 1e-9)) / max(float(r.freq_scale), 0.05)

    if r.axis == "xy":
        # meshgrid(lin_x, lin_y): Xg[j,i]=x, Yg[j,i]=y
        Xg, Yg = np.meshgrid(lin, lin)
        xf = Xg.flatten(); yf = Yg.flatten()
        tf = np.full_like(xf, te)

    elif r.axis == "xz":            # U=x (cols), V=t (rows); y fijo en sp
        # meshgrid(lin_x, t_lin): Xg[j,i]=x, Tg[j,i]=t
        Xg, Tg = np.meshgrid(lin, t_lin)
        xf = Xg.flatten()
        yf = np.full(N * N, sp, dtype=np.float32)
        tf = Tg.flatten() / fac         # escalar por sqrt(ε_r·μ_r)

    else:                            # "yz": U=t (cols), V=y (rows); x fijo en sp
        # meshgrid(t_lin, lin_y): Tg[j,i]=t, Yg[j,i]=y
        Tg, Yg = np.meshgrid(t_lin, lin)
        xf = np.full(N * N, sp, dtype=np.float32)
        yf = Yg.flatten()
        tf = Tg.flatten() / fac

    Ez, Bx, By = _infer(xf, yf, tf)
    damp = _damping(r.t, r.sigma_wall)
    Ez, Bx, By = Ez * damp, Bx * damp, By * damp
    return {
        "Ez": Ez.reshape(N, N).tolist(),
        "Bx": Bx.reshape(N, N).tolist(),
        "By": By.reshape(N, N).tolist(),
        "t_eff": float(te),
    }


@app.post("/vectors")
def api_vectors(r: VecReq):
    """Retorna vectores B en una rejilla NxN del plano XY, en el tiempo t."""
    if _model is None or np is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")
    N  = int(min(max(r.N, 5), 20))
    te = _t_eff(r.t, r.eps_r, r.mu_r, r.freq_scale)
    lin = np.linspace(0.05, 0.95, N, dtype=np.float32)
    Xg, Yg = np.meshgrid(lin, lin)
    xf = Xg.flatten(); yf = Yg.flatten()
    tf = np.full_like(xf, te)
    Ez, Bx, By = _infer(xf, yf, tf)
    damp = _damping(r.t, r.sigma_wall)
    Ez, Bx, By = Ez * damp, Bx * damp, By * damp
    return {
        "x": xf.tolist(), "y": yf.tolist(),
        "Ez": Ez.tolist(), "Bx": Bx.tolist(), "By": By.tolist(),
    }


class FieldGridReq(BaseModel):
    t:     float = 0.0
    N:     int   = 30
    eps_r: float = 1.0
    mu_r:  float = 1.0
    freq_scale: float = 1.0
    sigma_wall: float = 0.0


@app.post("/field_grid")
def api_field_grid(r: FieldGridReq):
    """Retorna campos en rejilla densa NxN para streamlines y heatmap |B|."""
    if _model is None or np is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")
    N = int(min(max(r.N, 10), 60))
    te = _t_eff(r.t, r.eps_r, r.mu_r, r.freq_scale)
    lin = np.linspace(0.0, 1.0, N, dtype=np.float32)
    Xg, Yg = np.meshgrid(lin, lin)
    xf = Xg.flatten(); yf = Yg.flatten()
    tf = np.full_like(xf, te)
    Ez, Bx, By = _infer(xf, yf, tf)
    damp = _damping(r.t, r.sigma_wall)
    Ez, Bx, By = Ez * damp, Bx * damp, By * damp
    Bmag = np.sqrt(Bx**2 + By**2)
    return {
        "N": N,
        "Bx": Bx.reshape(N, N).tolist(),
        "By": By.reshape(N, N).tolist(),
        "Bmag": Bmag.reshape(N, N).tolist(),
        "Ez": Ez.reshape(N, N).tolist(),
    }


class ResidualReq(BaseModel):
    t:     float = 0.0
    N:     int   = 20
    eps_r: float = 1.0
    mu_r:  float = 1.0
    freq_scale: float = 1.0
    sigma_wall: float = 0.0


@app.post("/residuals")
def api_residuals(r: ResidualReq):
    """Calcula residuales de Maxwell usando autograd del PINN."""
    if _model is None or torch is None or np is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")
    N = int(min(max(r.N, 8), 30))
    te = _t_eff(r.t, r.eps_r, r.mu_r, r.freq_scale)
    lin = np.linspace(0.05, 0.95, N, dtype=np.float32)
    Xg, Yg = np.meshgrid(lin, lin)

    x = torch.tensor(Xg.flatten(), dtype=torch.float32).view(-1, 1).to(_device).requires_grad_(True)
    y = torch.tensor(Yg.flatten(), dtype=torch.float32).view(-1, 1).to(_device).requires_grad_(True)
    t = torch.tensor(np.full(N * N, te, dtype=np.float32)).view(-1, 1).to(_device).requires_grad_(True)

    Ez, Bx, By = _model(x, y, t)
    ones = torch.ones_like(Ez)

    ag = lambda o, i: torch.autograd.grad(o, i, ones, create_graph=False, retain_graph=True)[0]

    dEz_dy = ag(Ez, y); dEz_dx = ag(Ez, x); dEz_dt = ag(Ez, t)
    dBx_dt = ag(Bx, t); dBy_dt = ag(By, t)
    dBy_dx = ag(By, x); dBx_dy = ag(Bx, y)
    dBx_dx = ag(Bx, x); dBy_dy = ag(By, y)

    r_fx = (dBx_dt + dEz_dy).detach().cpu().numpy().flatten()
    r_fy = (dBy_dt - dEz_dx).detach().cpu().numpy().flatten()
    r_amp = (dEz_dt - dBy_dx + dBx_dy).detach().cpu().numpy().flatten()
    r_gauss = (dBx_dx + dBy_dy).detach().cpu().numpy().flatten()

    return {
        "faraday_x": float(np.mean(np.abs(r_fx))),
        "faraday_y": float(np.mean(np.abs(r_fy))),
        "ampere":    float(np.mean(np.abs(r_amp))),
        "gauss_b":   float(np.mean(np.abs(r_gauss))),
        "total_mse": float(np.mean(r_fx**2 + r_fy**2 + r_amp**2 + r_gauss**2)),
    }


class ResidualMapReq(BaseModel):
    t:     float = 0.0
    N:     int   = 15
    eps_r: float = 1.0
    mu_r:  float = 1.0
    freq_scale: float = 1.0
    sigma_wall: float = 0.0


@app.post("/residual_map")
def api_residual_map(r: ResidualMapReq):
    """Retorna residuales de Maxwell resueltos espacialmente para overlay heatmap."""
    if _model is None or torch is None or np is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")
    N = int(min(max(r.N, 8), 25))
    te = _t_eff(r.t, r.eps_r, r.mu_r, r.freq_scale)
    lin = np.linspace(0.05, 0.95, N, dtype=np.float32)
    Xg, Yg = np.meshgrid(lin, lin)

    x = torch.tensor(Xg.flatten(), dtype=torch.float32).view(-1, 1).to(_device).requires_grad_(True)
    y = torch.tensor(Yg.flatten(), dtype=torch.float32).view(-1, 1).to(_device).requires_grad_(True)
    t = torch.tensor(np.full(N * N, te, dtype=np.float32)).view(-1, 1).to(_device).requires_grad_(True)

    Ez, Bx, By = _model(x, y, t)
    ones = torch.ones_like(Ez)
    ag = lambda o, i: torch.autograd.grad(o, i, ones, create_graph=False, retain_graph=True)[0]

    r_fx = (ag(Bx, t) + ag(Ez, y)).detach().cpu().numpy().flatten()
    r_fy = (ag(By, t) - ag(Ez, x)).detach().cpu().numpy().flatten()
    r_amp = (ag(Ez, t) - ag(By, x) + ag(Bx, y)).detach().cpu().numpy().flatten()
    r_gauss = (ag(Bx, x) + ag(By, y)).detach().cpu().numpy().flatten()

    total = np.sqrt(r_fx**2 + r_fy**2 + r_amp**2 + r_gauss**2)

    return {
        "N": N,
        "total": total.reshape(N, N).tolist(),
        "faraday_x": np.abs(r_fx).reshape(N, N).tolist(),
        "faraday_y": np.abs(r_fy).reshape(N, N).tolist(),
        "ampere": np.abs(r_amp).reshape(N, N).tolist(),
        "gauss_b": np.abs(r_gauss).reshape(N, N).tolist(),
    }
