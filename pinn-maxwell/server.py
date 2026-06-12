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

import os, sys, math, json, csv, io, time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
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
    from analytical import Ez_exact, Bx_exact, By_exact, cavity_mode, L, eps0, mu0
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


# ════════════════════════════════════════════════════════════════
# FASE 3 — COMPARACIÓN, DIAGNÓSTICO Y EXPORTACIÓN
# ════════════════════════════════════════════════════════════════

class CompareReq(BaseModel):
    N:     int   = 60
    eps_r: float = 1.0
    mu_r:  float = 1.0
    t:     float = 0.0

@app.post("/api/compare/slice")
def api_compare_slice(r: CompareReq):
    """
    Retorna solución PINN + analítica + mapa de error en una sola respuesta.
    Permite mostrar side-by-side PINN vs Analytical en el frontend.
    """
    if _model is None or np is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")
    N = int(min(max(r.N, 20), 100))
    lin = np.linspace(0.0, L, N, dtype=np.float32)

    # Escalar tiempo por permitividad/permeabilidad
    te = _t_eff(r.t, r.eps_r, r.mu_r, 1.0)

    Xg, Yg = np.meshgrid(lin, lin)
    xf = Xg.flatten()
    yf = Yg.flatten()
    tf = np.full_like(xf, te)

    # PINN prediction
    Ez_p, Bx_p, By_p = _infer(xf, yf, tf)

    # Analytical solution
    Ez_a = Ez_exact(xf, yf, te, 1, 1).flatten().astype(np.float64)
    Bx_a = Bx_exact(xf, yf, te, 1, 1).flatten().astype(np.float64)
    By_a = By_exact(xf, yf, te, 1, 1).flatten().astype(np.float64)

    # Error maps
    err_Ez = np.abs(Ez_p - Ez_a)
    err_Bx = np.abs(Bx_p - Bx_a)
    err_By = np.abs(By_p - By_a)

    # Relative L2 errors
    def rel_l2(pred, exact):
        denom = np.linalg.norm(exact, 2)
        return float(np.linalg.norm(pred - exact, 2) / denom * 100.0) if denom > 1e-12 else 0.0

    # MSE
    def mse(pred, exact):
        return float(np.mean((pred - exact) ** 2))

    return {
        "Ez_pinn": Ez_p.reshape(N, N).tolist(),
        "Bx_pinn": Bx_p.reshape(N, N).tolist(),
        "By_pinn": By_p.reshape(N, N).tolist(),
        "Ez_analytical": Ez_a.reshape(N, N).tolist(),
        "Bx_analytical": Bx_a.reshape(N, N).tolist(),
        "By_analytical": By_a.reshape(N, N).tolist(),
        "err_Ez": err_Ez.reshape(N, N).tolist(),
        "err_Bx": err_Bx.reshape(N, N).tolist(),
        "err_By": err_By.reshape(N, N).tolist(),
        "N": N,
        "t_eff": float(te),
        "metrics": {
            "l2_Ez": rel_l2(Ez_p, Ez_a),
            "l2_Bx": rel_l2(Bx_p, Bx_a),
            "l2_By": rel_l2(By_p, By_a),
            "mse_Ez": mse(Ez_p, Ez_a),
            "mse_Bx": mse(Bx_p, Bx_a),
            "mse_By": mse(By_p, By_a),
        }
    }


@app.post("/api/compare/metrics")
def api_compare_metrics(r: CompareReq):
    """
    Retorna métricas completas de error: L2, MSE, PDE residuals, BC, energía.
    """
    if _model is None or torch is None or np is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")

    N = int(min(max(r.N, 20), 80))
    te = _t_eff(r.t, r.eps_r, r.mu_r, 1.0)
    lin = np.linspace(0.05, 0.95, N, dtype=np.float32)
    Xg, Yg = np.meshgrid(lin, lin)
    xf = Xg.flatten(); yf = Yg.flatten()
    tf = np.full_like(xf, te)

    # PINN
    Ez_p, Bx_p, By_p = _infer(xf, yf, tf)

    # Analytical
    Ez_a = Ez_exact(xf, yf, te, 1, 1).flatten().astype(np.float64)
    Bx_a = Bx_exact(xf, yf, te, 1, 1).flatten().astype(np.float64)
    By_a = By_exact(xf, yf, te, 1, 1).flatten().astype(np.float64)

    def rel_l2(p, e):
        denom = np.linalg.norm(e, 2)
        return float(np.linalg.norm(p - e, 2) / denom * 100.0) if denom > 1e-12 else 0.0

    def mse_v(p, e):
        return float(np.mean((p - e) ** 2))

    # PDE residuals via autograd
    x = torch.tensor(xf, dtype=torch.float32).view(-1, 1).to(_device).requires_grad_(True)
    y = torch.tensor(yf, dtype=torch.float32).view(-1, 1).to(_device).requires_grad_(True)
    t = torch.tensor(tf, dtype=torch.float32).view(-1, 1).to(_device).requires_grad_(True)

    Ez, Bx, By = _model(x, y, t)
    ones = torch.ones_like(Ez)
    ag = lambda o, i: torch.autograd.grad(o, i, ones, create_graph=False, retain_graph=True)[0]

    r_fx = (ag(Bx, t) + ag(Ez, y)).detach().cpu().numpy()
    r_fy = (ag(By, t) - ag(Ez, x)).detach().cpu().numpy()
    r_amp = (ag(Ez, t) - ag(By, x) + ag(Bx, y)).detach().cpu().numpy()
    r_gauss = (ag(Bx, x) + ag(By, y)).detach().cpu().numpy()

    # Energy conservation — integración temporal multi-paso
    # Conservación verdadera: U(t) = ∬ u(x,y,t) dx dy debe ser constante.
    # Medimos (U_max - U_min) / U_mean a lo largo del tiempo, no dispersión espacial.
    eps_r_val = max(float(r.eps_r), 1e-9)
    mu_r_val = max(float(r.mu_r), 1e-9)
    Nt = 20
    t_energy = np.linspace(0.0, T_MAX, Nt, dtype=np.float32)
    x_2d = np.linspace(0.05, 0.95, 30, dtype=np.float32)
    dx = x_2d[1] - x_2d[0]
    Xg2, Yg2 = np.meshgrid(x_2d, x_2d)
    xf2 = Xg2.flatten(); yf2 = Yg2.flatten()
    U_arr = []
    for t_val in t_energy:
        tf2 = np.full_like(xf2, t_val)
        Ez_t, Bx_t, By_t = _infer(xf2, yf2, tf2)
        u_t = 0.5 * eps0 * eps_r_val * (Ez_t ** 2) + (0.5 / (mu0 * mu_r_val)) * (Bx_t ** 2 + By_t ** 2)
        U_arr.append(np.sum(u_t) * dx * dx)
    U_arr = np.array(U_arr)
    U_mean_t = float(np.mean(U_arr))
    conservation_pct = float((np.max(U_arr) - np.min(U_arr)) / U_mean_t * 100.0) if U_mean_t > 1e-12 else 0.0

    return {
        "l2": {
            "Ez": rel_l2(Ez_p, Ez_a),
            "Bx": rel_l2(Bx_p, Bx_a),
            "By": rel_l2(By_p, By_a),
            "average": (rel_l2(Ez_p, Ez_a) + rel_l2(Bx_p, Bx_a) + rel_l2(By_p, By_a)) / 3.0
        },
        "mse": {
            "Ez": mse_v(Ez_p, Ez_a),
            "Bx": mse_v(Bx_p, Bx_a),
            "By": mse_v(By_p, By_a)
        },
        "pde_residuals": {
            "faraday_x": float(np.mean(np.abs(r_fx))),
            "faraday_y": float(np.mean(np.abs(r_fy))),
            "ampere": float(np.mean(np.abs(r_amp))),
            "gauss_b": float(np.mean(np.abs(r_gauss))),
            "total_mse": float(np.mean(r_fx**2 + r_fy**2 + r_amp**2 + r_gauss**2))
        },
        "energy": {
            "mean": U_mean_t,
            "n_steps": Nt,
            "conservation_pct": conservation_pct
        }
    }


@app.get("/api/training/history")
def api_training_history():
    """
    Retorna el historial de entrenamiento desde training_log.txt.
    Si no existe, retorna datos mock de demostración.
    """
    log_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'training_log.txt'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'training_log.txt'),
    ]
    history = []
    headers = None

    for lp in log_paths:
        lp = os.path.normpath(lp)
        if os.path.exists(lp):
            try:
                with open(lp, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith('[') and 'Total:' in line:
                        parts = line.split('|')
                        epoch_part = parts[0].strip()
                        # Extract epoch number
                        import re
                        m = re.search(r'(\d+)', epoch_part)
                        epoch = int(m.group(1)) if m else 0
                        total = float(parts[1].replace('Total:', '').strip())
                        pde = float(parts[2].replace('PDE:', '').strip())
                        bc = float(parts[3].replace('BC:', '').strip())
                        ic = float(parts[4].replace('IC:', '').strip())
                        history.append({
                            'epoch': epoch, 'loss_total': total,
                            'loss_pde': pde, 'loss_bc': bc, 'loss_ic': ic
                        })
            except:
                pass
            break

    # Return fallback mock data if no log found
    if not history:
        import math
        for i in range(1, 301):
            e = i * 100
            history.append({
                'epoch': e,
                'loss_total': 124.0 * math.exp(-0.015 * e**0.6) + 1e-3,
                'loss_pde': 30.0 * math.exp(-0.012 * e**0.6) + 1e-3,
                'loss_bc': 1.0 * math.exp(-0.02 * e**0.5) + 1e-4,
                'loss_ic': 0.81 * math.exp(-0.025 * e**0.5) + 1e-4,
            })

    return {
        "history": history,
        "params": {
            "arch": str(ARCH),
            "fourier": FOURIER,
            "n_fourier": N_F,
            "sigma": SIGMA,
            "n_params": sum(p.numel() for p in _model.parameters() if p.requires_grad) if _model is not None else 0
        }
    }


@app.post("/api/validate/boundary")
def api_validate_boundary(N_per_side: int = 100):
    """
    Valida la condición PEC: Ez = 0 en los 4 bordes.
    Retorna error medio y máximo en cada pared.
    """
    if _model is None or np is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")

    sides = {}
    all_pts_list = []
    for side_name, x_fixed, y_fixed in [
        ('left', 0.0, None), ('right', L, None),
        ('bottom', None, 0.0), ('top', None, L)
    ]:
        xs = np.full(N_per_side, x_fixed if x_fixed is not None else 0.0, dtype=np.float32)
        ys = np.full(N_per_side, y_fixed if y_fixed is not None else 0.0, dtype=np.float32)

        if x_fixed is None:
            xs = np.random.uniform(0, L, N_per_side).astype(np.float32)
        if y_fixed is None:
            ys = np.random.uniform(0, L, N_per_side).astype(np.float32)

        ts = np.random.uniform(0, T_MAX, N_per_side).astype(np.float32)

        Ez, _, _ = _infer(xs, ys, ts)
        abs_err = np.abs(Ez)
        all_pts_list.append(abs_err)

        side_errors = {
            'mean': float(np.mean(abs_err)),
            'max': float(np.max(abs_err)),
            'std': float(np.std(abs_err))
        }
        sides[side_name] = side_errors

    all_pts = np.concatenate(all_pts_list)
    return {
        "sides": sides,
        "global_mean": float(np.mean(all_pts)),
        "global_max": float(np.max(all_pts)),
        "pec_violation_pct": float(np.mean(all_pts > 1e-3) * 100.0)
    }


class ExportReq(BaseModel):
    format: str = "json"       # "json" | "csv"
    t:      float = 0.0
    N:      int   = 50
    eps_r:  float = 1.0
    mu_r:   float = 1.0

@app.post("/api/export/field")
def api_export_field(r: ExportReq):
    """
    Exporta los campos en formato JSON o CSV para post-procesamiento.
    """
    if _model is None or np is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")

    N = int(min(max(r.N, 10), 100))
    te = _t_eff(r.t, r.eps_r, r.mu_r, 1.0)
    lin = np.linspace(0.0, L, N, dtype=np.float32)
    Xg, Yg = np.meshgrid(lin, lin)
    xf = Xg.flatten(); yf = Yg.flatten()
    tf = np.full_like(xf, te)

    Ez_p, Bx_p, By_p = _infer(xf, yf, tf)
    Ez_a = Ez_exact(xf, yf, te, 1, 1).flatten().astype(np.float64)
    Bx_a = Bx_exact(xf, yf, te, 1, 1).flatten().astype(np.float64)
    By_a = By_exact(xf, yf, te, 1, 1).flatten().astype(np.float64)

    if r.format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['x', 'y', 't', 'Ez_pinn', 'Bx_pinn', 'By_pinn', 'Ez_analytical', 'Bx_analytical', 'By_analytical'])
        for i in range(len(xf)):
            writer.writerow([xf[i], yf[i], te, Ez_p[i], Bx_p[i], By_p[i], Ez_a[i], Bx_a[i], By_a[i]])
        return Response(content=output.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=maxwell_fields_t{te:.3f}.csv"})

    # JSON format (default)
    data = []
    for i in range(len(xf)):
        data.append({
            "x": float(xf[i]), "y": float(yf[i]), "t": float(te),
            "Ez_pinn": float(Ez_p[i]), "Bx_pinn": float(Bx_p[i]), "By_pinn": float(By_p[i]),
            "Ez_analytical": float(Ez_a[i]), "Bx_analytical": float(Bx_a[i]), "By_analytical": float(By_a[i])
        })

    return {
        "fields": data,
        "metadata": {
            "N": N, "t": float(te), "eps_r": float(r.eps_r), "mu_r": float(r.mu_r),
            "mode": "TM11", "model": "PINN Maxwell"
        }
    }


@app.get("/api/export/config")
def api_export_config():
    """Exporta la configuración actual de simulación como JSON."""
    return {
        "config": {
            "arch": str(ARCH),
            "fourier_features": FOURIER,
            "n_fourier": N_F,
            "sigma_fourier": SIGMA,
            "domain": {"L": L, "T_MAX": T_MAX},
            "physics": {"mode": "TM11", "mu0": 1.0, "eps0": 1.0, "c": 1.0},
            "model_loaded": _model is not None,
            "device": str(_device)
        }
    }


class BenchmarkReq(BaseModel):
    N: int = 40

@app.post("/api/benchmark")
def api_benchmark(r: BenchmarkReq):
    """
    Benchmark conceptual: compara tiempo de inferencia PINN vs
    estimación de lo que tomaría FDTD/FEM (mock).
    """
    if _model is None or torch is None or np is None:
        raise HTTPException(status_code=503, detail="PINN model is not loaded")

    N = int(min(max(r.N, 10), 100))
    batch_sizes = [100, 1000, 10000, 100000]
    results = []

    for bs in batch_sizes:
        x = torch.rand(bs, 1).to(_device)
        y = torch.rand(bs, 1).to(_device)
        t = torch.rand(bs, 1).to(_device)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _model(x, y, t)

        if _device == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        n_iter = max(1, 100000 // bs)
        for _ in range(n_iter):
            with torch.no_grad():
                _model(x, y, t)
        if _device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        avg_ms = elapsed / n_iter * 1000.0
        pts_per_sec = bs * n_iter / elapsed

        # FDTD mock: O(N^3) per time step for NxN grid
        fdtd_grids = [20, 50, 100, 200]
        fdtd_estimates = {}
        for g in fdtd_grids:
            fdtd_estimates[f"{g}x{g}"] = {
                "time_per_step_ms": (g**3) * 1e-5 * 1000,  # rough estimate
                "steps_per_second": 1.0 / ((g**3) * 1e-5) if (g**3) * 1e-5 > 0 else float('inf')
            }

        results.append({
            "batch_size": bs,
            "pinn_time_ms": round(avg_ms, 3),
            "pinn_pts_per_sec": round(pts_per_sec, 0),
            "fdtd_comparison": fdtd_estimates
        })

    return {
        "benchmark": results,
        "device": str(_device),
        "note": "FDTD estimates are conceptual (O(N^3) per step). PINN is mesh-free."
    }
