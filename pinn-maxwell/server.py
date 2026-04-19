"""
server.py — Backend FastAPI para la Cámara de Faraday Digital
Sirve predicciones del PINN Maxwell TM₁₁ en tiempo real al frontend Three.js.

Ejecución:
    cd pinn-maxwell
    pip install fastapi uvicorn
    uvicorn server:app --reload --port 8000

Frontend: http://localhost:8000/demo
"""
import os, sys, math
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Asegura importabilidad de model.py desde cualquier CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FCN

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
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)       # reproduce la matriz B del Fourier Encoding
    _model = FCN(ARCH, fourier_features=FOURIER, n_fourier=N_F, sigma=SIGMA)
    pth = _locate_model()
    _model.load_state_dict(torch.load(pth, map_location=_device, weights_only=True))
    _model.to(_device).eval()
    print(f"\n[OK] PINN Maxwell TM11 cargado: {pth}")
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

class VecReq(BaseModel):
    t:     float = 0.0
    N:     int   = 10        # rejilla NxN de vectores (N²  flechas)
    eps_r: float = 1.0
    mu_r:  float = 1.0

# ─── Utilidades de Inferencia ─────────────────────────────────────────────────
def _t_eff(t: float, eps_r: float, mu_r: float) -> float:
    """Escala el tiempo para simular cambio de ε_r y μ_r.
    La velocidad de fase es c/√(ε_r·μ_r), por tanto la onda 've'
    un tiempo efectivo t_eff = t / √(ε_r·μ_r).
    """
    return float(t) / math.sqrt(max(float(eps_r) * float(mu_r), 1e-9))

@torch.no_grad()
def _infer(x_np: np.ndarray, y_np: np.ndarray, t_np: np.ndarray):
    """Evaluación batch sin autograd. Retorna (Ez, Bx, By) como arrays 1D."""
    to = lambda a: torch.tensor(a, dtype=torch.float32).view(-1, 1).to(_device)
    out = _model(to(x_np), to(y_np), to(t_np))
    return tuple(o.cpu().numpy().flatten() for o in out)

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status": "online",
        "model": "Maxwell PINN TM₁₁",
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
    N    = int(min(max(r.N, 10), 100))
    te   = _t_eff(r.t, r.eps_r, r.mu_r)
    sp   = float(r.position)
    lin  = np.linspace(0.0, L, N, dtype=np.float32)
    t_lin = np.linspace(0.0, T_MAX, N, dtype=np.float32)
    fac  = math.sqrt(max(float(r.eps_r) * float(r.mu_r), 1e-9))

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
    return {
        "Ez": Ez.reshape(N, N).tolist(),
        "Bx": Bx.reshape(N, N).tolist(),
        "By": By.reshape(N, N).tolist(),
        "t_eff": float(te),
    }


@app.post("/vectors")
def api_vectors(r: VecReq):
    """Retorna vectores B en una rejilla NxN del plano XY, en el tiempo t."""
    N  = int(min(max(r.N, 5), 20))
    te = _t_eff(r.t, r.eps_r, r.mu_r)
    lin = np.linspace(0.05, 0.95, N, dtype=np.float32)
    Xg, Yg = np.meshgrid(lin, lin)
    xf = Xg.flatten(); yf = Yg.flatten()
    tf = np.full_like(xf, te)
    Ez, Bx, By = _infer(xf, yf, tf)
    return {
        "x": xf.tolist(), "y": yf.tolist(),
        "Ez": Ez.tolist(), "Bx": Bx.tolist(), "By": By.tolist(),
    }
