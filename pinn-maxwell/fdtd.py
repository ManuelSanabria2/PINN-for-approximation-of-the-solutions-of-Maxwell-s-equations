"""
fdtd.py — Solver FDTD 2D TM (esquema de Yee) para las ecuaciones de Maxwell.

Resuelve exactamente el mismo problema que el PINN (physics.py):

    Modo TM 2D en unidades naturales (c = 1, ε₀ = 1, μ₀ = 1):
        ∂Bx/∂t = -∂Ez/∂y
        ∂By/∂t = +∂Ez/∂x
        ε_r · ∂Ez/∂t = (1/μ_r)·(∂By/∂x - ∂Bx/∂y) - σ·Ez

    Dominio: cavidad [0, L]² con paredes PEC (Ez = 0 en los bordes).
    Condición inicial: modo TM₁₁ → Ez = E0·sin(πx/L)·sin(πy/L), B = 0.

Malla de Yee (staggered):
    Ez  en nodos enteros        (i, j)          → shape (Nx, Ny)
    Bx  en medio-nodos en y     (i, j+½)        → shape (Nx, Ny-1)
    By  en medio-nodos en x     (i+½, j)        → shape (Nx-1, Ny)

Convención de indexado: arreglos [i, j] con i → x, j → y.

Características:
    - Mapa de materiales heterogéneo (ε_r, μ_r, σ) construido desde una
      lista de objetos geométricos (esferas/círculos, cajas).
    - Fuente de pulso gaussiano modulado (soft source) opcional.
    - Capas absorbentes tipo PML (conductividad graduada polinómica).
    - Condición de estabilidad de Courant (CFL).
    - Registro de snapshots, sondas puntuales Ez(t) y energía total U(t).
    - Callback de progreso para streaming a la consola de la UI.
"""
from __future__ import annotations

import time
import math
import numpy as np

# Unidades naturales (idénticas a physics.py / analytical.py)
C0   = 1.0
EPS0 = 1.0
MU0  = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────────────────────
class FDTDConfig:
    """Parámetros de una simulación FDTD 2D TM."""

    def __init__(self,
                 L: float = 1.0,
                 Nx: int = 101,
                 Ny: int = 101,
                 T_max: float = 2.828427,     # 2 periodos TM₁₁
                 courant: float = 0.5,        # S = c·dt/dx (límite 2D: 1/√2)
                 eps_r_bg: float = 1.0,
                 mu_r_bg: float = 1.0,
                 sigma_bg: float = 0.0,
                 objects: list | None = None,     # inclusiones de material
                 source: dict | None = None,      # pulso gaussiano opcional
                 init_mode: str = "tm11",         # "tm11" | "zero"
                 E0: float = 1.0,
                 pml: dict | None = None,         # {"cells": 8, "sigma_max": ..., "order": 3}
                 probes: list | None = None,      # [{"x":0.5,"y":0.5,"name":"centro"}]
                 n_snapshots: int = 60):
        self.L = float(L)
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.T_max = float(T_max)
        self.courant = float(courant)
        self.eps_r_bg = float(eps_r_bg)
        self.mu_r_bg = float(mu_r_bg)
        self.sigma_bg = float(sigma_bg)
        self.objects = objects or []
        self.source = source
        self.init_mode = init_mode
        self.E0 = float(E0)
        self.pml = pml
        self.probes = probes or [{"x": 0.5, "y": 0.5, "name": "centro"}]
        self.n_snapshots = int(n_snapshots)

        # Derivados
        self.dx = self.L / (self.Nx - 1)
        self.dy = self.L / (self.Ny - 1)
        # CFL 2D: c·dt ≤ 1/√(1/dx² + 1/dy²). courant = fracción del límite.
        self.dt_limit = 1.0 / (C0 * math.sqrt(1.0 / self.dx**2 + 1.0 / self.dy**2))
        self.dt = self.courant * self.dt_limit * math.sqrt(2.0)  # S·dx/c si dx=dy
        # Nota: para dx=dy, dt_limit = dx/(c√2); dt = courant·dx/c.
        self.dt = min(self.dt, 0.999 * self.dt_limit)
        self.n_steps = max(1, int(math.ceil(self.T_max / self.dt)))

    @classmethod
    def from_dict(cls, d: dict) -> "FDTDConfig":
        keys = ["L", "Nx", "Ny", "T_max", "courant", "eps_r_bg", "mu_r_bg",
                "sigma_bg", "objects", "source", "init_mode", "E0", "pml",
                "probes", "n_snapshots"]
        return cls(**{k: d[k] for k in keys if k in d})

    def to_dict(self) -> dict:
        return {
            "L": self.L, "Nx": self.Nx, "Ny": self.Ny, "T_max": self.T_max,
            "courant": self.courant, "eps_r_bg": self.eps_r_bg,
            "mu_r_bg": self.mu_r_bg, "sigma_bg": self.sigma_bg,
            "objects": self.objects, "source": self.source,
            "init_mode": self.init_mode, "E0": self.E0, "pml": self.pml,
            "probes": self.probes, "n_snapshots": self.n_snapshots,
            "dx": self.dx, "dy": self.dy, "dt": self.dt, "n_steps": self.n_steps,
            "dt_limit": self.dt_limit,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Materiales
# ─────────────────────────────────────────────────────────────────────────────
def build_material_maps(cfg: FDTDConfig):
    """
    Construye los mapas ε_r(i,j), μ_r(i,j), σ(i,j) en los nodos de Ez.

    Objetos soportados (dicts):
        {"shape": "sphere", "cx": 0.5, "cy": 0.5, "r": 0.15,
         "eps_r": 11.7, "mu_r": 1.0, "sigma": 0.0}
        {"shape": "box", "x0": .., "y0": .., "x1": .., "y1": .., ...}
        PEC interno: usar "pec": true (fuerza Ez=0 dentro del objeto).
    """
    Nx, Ny = cfg.Nx, cfg.Ny
    x = np.linspace(0.0, cfg.L, Nx)
    y = np.linspace(0.0, cfg.L, Ny)
    Xg, Yg = np.meshgrid(x, y, indexing="ij")     # [i, j]

    eps = np.full((Nx, Ny), cfg.eps_r_bg, dtype=np.float64)
    mu  = np.full((Nx, Ny), cfg.mu_r_bg,  dtype=np.float64)
    sig = np.full((Nx, Ny), cfg.sigma_bg, dtype=np.float64)
    pec_mask = np.zeros((Nx, Ny), dtype=bool)

    for obj in cfg.objects:
        if not obj.get("enabled", True):
            continue
        shape = obj.get("shape", "sphere")
        if shape in ("sphere", "circle"):
            m = (Xg - obj.get("cx", 0.5))**2 + (Yg - obj.get("cy", 0.5))**2 <= obj.get("r", 0.1)**2
        elif shape == "box":
            m = ((Xg >= obj.get("x0", 0.0)) & (Xg <= obj.get("x1", 1.0)) &
                 (Yg >= obj.get("y0", 0.0)) & (Yg <= obj.get("y1", 1.0)))
        else:
            continue
        if obj.get("pec", False):
            pec_mask |= m
        else:
            eps[m] = float(obj.get("eps_r", 1.0))
            mu[m]  = float(obj.get("mu_r", 1.0))
            sig[m] = float(obj.get("sigma", 0.0))

    # PML: conductividad graduada polinómica en un marco de n celdas.
    if cfg.pml and int(cfg.pml.get("cells", 0)) > 0:
        n = int(cfg.pml["cells"])
        order = float(cfg.pml.get("order", 3))
        # σ_max teórico (Taflove) adaptado a unidades naturales
        sigma_max = float(cfg.pml.get("sigma_max",
                          0.8 * (order + 1) / (cfg.dx * math.sqrt(cfg.eps_r_bg * cfg.mu_r_bg))))
        prof = np.zeros((Nx, Ny))
        ii = np.arange(Nx)[:, None] * np.ones((1, Ny))
        jj = np.ones((Nx, 1)) * np.arange(Ny)[None, :]
        for depth, idx in ((n - ii, ii), (ii - (Nx - 1 - n), ii),
                           (n - jj, jj), (jj - (Ny - 1 - n), jj)):
            d = np.clip(depth / n, 0.0, 1.0)
            prof = np.maximum(prof, sigma_max * d**order)
        sig = sig + prof

    return eps, mu, sig, pec_mask


# ─────────────────────────────────────────────────────────────────────────────
# Resultado
# ─────────────────────────────────────────────────────────────────────────────
class FDTDResult:
    """Contenedor de resultados de una corrida FDTD."""

    def __init__(self):
        self.times: list[float] = []           # tiempos de los snapshots
        self.Ez: list[np.ndarray] = []         # snapshots Ez (Nx, Ny)
        self.Bx: list[np.ndarray] = []         # Bx interpolado a nodos Ez
        self.By: list[np.ndarray] = []
        self.probe_t: list[float] = []
        self.probes: dict[str, list[float]] = {}
        self.energy_t: list[float] = []
        self.energy: list[float] = []
        self.wall_time: float = 0.0
        self.n_steps: int = 0
        self.memory_bytes: int = 0
        self.config: dict = {}

    def to_arrays(self):
        return (np.asarray(self.times), np.stack(self.Ez),
                np.stack(self.Bx), np.stack(self.By))


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────
class FDTDSolver:
    """Solver Yee 2D TM. Uso:  FDTDSolver(cfg).run(progress_cb)"""

    def __init__(self, cfg: FDTDConfig):
        self.cfg = cfg
        self.eps, self.mu, self.sigma, self.pec_mask = build_material_maps(cfg)

    # — Interpolación de B (staggered) a los nodos de Ez para snapshots/energía —
    def _bx_at_nodes(self, Bx):
        out = np.zeros((self.cfg.Nx, self.cfg.Ny))
        out[:, 1:-1] = 0.5 * (Bx[:, 1:] + Bx[:, :-1])
        out[:, 0] = Bx[:, 0]
        out[:, -1] = Bx[:, -1]
        return out

    def _by_at_nodes(self, By):
        out = np.zeros((self.cfg.Nx, self.cfg.Ny))
        out[1:-1, :] = 0.5 * (By[1:, :] + By[:-1, :])
        out[0, :] = By[0, :]
        out[-1, :] = By[-1, :]
        return out

    def _energy(self, Ez, Bxn, Byn):
        """U = ½ ∫ (ε·Ez² + (Bx²+By²)/μ) dA  (regla del trapecio 2D)."""
        if not hasattr(self, "_quad_w"):
            w = np.ones((self.cfg.Nx, self.cfg.Ny))
            w[0, :] *= 0.5; w[-1, :] *= 0.5
            w[:, 0] *= 0.5; w[:, -1] *= 0.5
            self._quad_w = w
        u = 0.5 * (self.eps * Ez**2 + (Bxn**2 + Byn**2) / self.mu)
        return float(np.sum(self._quad_w * u) * self.cfg.dx * self.cfg.dy)

    def run(self, progress_cb=None, cancel_cb=None) -> FDTDResult:
        cfg = self.cfg
        Nx, Ny, dx, dy, dt = cfg.Nx, cfg.Ny, cfg.dx, cfg.dy, cfg.dt

        x = np.linspace(0.0, cfg.L, Nx)
        y = np.linspace(0.0, cfg.L, Ny)
        Xg, Yg = np.meshgrid(x, y, indexing="ij")

        # ── Campos ────────────────────────────────────────────────────────
        Ez = np.zeros((Nx, Ny))
        Bx = np.zeros((Nx, Ny - 1))     # (i, j+½)
        By = np.zeros((Nx - 1, Ny))     # (i+½, j)

        if cfg.init_mode == "tm11":
            Ez[:] = cfg.E0 * np.sin(np.pi * Xg / cfg.L) * np.sin(np.pi * Yg / cfg.L)

        # PEC: bordes + máscaras internas
        Ez[0, :] = Ez[-1, :] = Ez[:, 0] = Ez[:, -1] = 0.0
        Ez[self.pec_mask] = 0.0

        # ── Coeficientes de actualización (materiales + pérdidas) ─────────
        # Ez: (ε + σ·dt/2)·Ez^{n+1} = (ε - σ·dt/2)·Ez^n + dt·curl(B/μ)
        ce_a = (self.eps - 0.5 * dt * self.sigma) / (self.eps + 0.5 * dt * self.sigma)
        ce_b = dt / (self.eps + 0.5 * dt * self.sigma)

        # μ en las posiciones staggered de Bx y By (media aritmética)
        mu_bx = 0.5 * (self.mu[:, 1:] + self.mu[:, :-1])       # (Nx, Ny-1)
        mu_by = 0.5 * (self.mu[1:, :] + self.mu[:-1, :])       # (Nx-1, Ny)

        # ── Fuente opcional ────────────────────────────────────────────────
        src = cfg.source
        if src and src.get("enabled", True) and cfg.init_mode != "tm11_only":
            si = int(round(src.get("x", 0.5) / dx))
            sj = int(round(src.get("y", 0.5) / dy))
            si = min(max(si, 1), Nx - 2)
            sj = min(max(sj, 1), Ny - 2)
            s_amp  = float(src.get("amplitude", 1.0))
            s_f    = float(src.get("freq", 1.5))         # frecuencia (unid. naturales)
            s_t0   = float(src.get("t0", 0.5))
            s_tau  = float(src.get("tau", 0.15))
            s_kind = src.get("kind", "pulse")            # "pulse" | "cw"
        else:
            src = None

        # ── Sondas ────────────────────────────────────────────────────────
        probe_idx = []
        for p in cfg.probes:
            pi = min(max(int(round(p["x"] / dx)), 0), Nx - 1)
            pj = min(max(int(round(p["y"] / dy)), 0), Ny - 1)
            probe_idx.append((p.get("name", f"p{len(probe_idx)}"), pi, pj))

        # ── Planificación de snapshots ─────────────────────────────────────
        n_snap = min(cfg.n_snapshots, cfg.n_steps + 1)
        snap_steps = set(np.unique(np.linspace(0, cfg.n_steps, n_snap).astype(int)).tolist())

        res = FDTDResult()
        res.config = cfg.to_dict()
        res.memory_bytes = (Ez.nbytes + Bx.nbytes + By.nbytes +
                            self.eps.nbytes + self.mu.nbytes + self.sigma.nbytes +
                            ce_a.nbytes + ce_b.nbytes + mu_bx.nbytes + mu_by.nbytes)
        for name, _, _ in probe_idx:
            res.probes[name] = []

        # B queda en n+½ (leapfrog); para energía se centra a n promediando
        # con el medio-paso anterior — evita una oscilación espuria a 2ω.
        prev_B = {"Bxn": None, "Byn": None}

        def record(step, t_now):
            Bxn = self._bx_at_nodes(Bx)
            Byn = self._by_at_nodes(By)
            if step in snap_steps:
                res.times.append(t_now)
                res.Ez.append(Ez.copy())
                res.Bx.append(Bxn)
                res.By.append(Byn)
            res.probe_t.append(t_now)
            for name, pi, pj in probe_idx:
                res.probes[name].append(float(Ez[pi, pj]))
            res.energy_t.append(t_now)
            if prev_B["Bxn"] is not None:
                Bxc = 0.5 * (prev_B["Bxn"] + Bxn)
                Byc = 0.5 * (prev_B["Byn"] + Byn)
            else:
                Bxc, Byc = Bxn, Byn
            res.energy.append(self._energy(Ez, Bxc, Byc))
            prev_B["Bxn"], prev_B["Byn"] = Bxn, Byn

        t_start = time.perf_counter()
        record(0, 0.0)

        # ── Medio paso inicial de B (leapfrog): B^{½} = B⁰ + (dt/2)·(-curl E) ──
        Bx -= (0.5 * dt / dy) * (Ez[:, 1:] - Ez[:, :-1])
        By += (0.5 * dt / dx) * (Ez[1:, :] - Ez[:-1, :])

        # ── Bucle temporal ─────────────────────────────────────────────────
        for n in range(1, cfg.n_steps + 1):
            if cancel_cb and cancel_cb():
                break

            # Ez^{n} ← Ez^{n-1} + curl(B^{n-½}/μ)
            curl = ((By[1:, 1:-1] / mu_by[1:, 1:-1] - By[:-1, 1:-1] / mu_by[:-1, 1:-1]) / dx
                    - (Bx[1:-1, 1:] / mu_bx[1:-1, 1:] - Bx[1:-1, :-1] / mu_bx[1:-1, :-1]) / dy)
            Ez[1:-1, 1:-1] = ce_a[1:-1, 1:-1] * Ez[1:-1, 1:-1] + ce_b[1:-1, 1:-1] * curl

            # PEC (bordes ya quedan fijos porque no se actualizan)
            if self.pec_mask.any():
                Ez[self.pec_mask] = 0.0

            t_now = n * dt

            # Fuente blanda
            if src is not None:
                if s_kind == "cw":
                    val = s_amp * math.sin(2 * math.pi * s_f * t_now)
                else:
                    val = (s_amp * math.exp(-((t_now - s_t0) / s_tau)**2)
                           * math.sin(2 * math.pi * s_f * t_now))
                Ez[si, sj] += dt * val

            # B^{n+½} ← B^{n-½} - dt·curl(E^n)
            Bx -= (dt / dy) * (Ez[:, 1:] - Ez[:, :-1])
            By += (dt / dx) * (Ez[1:, :] - Ez[:-1, :])

            record(n, t_now)

            if progress_cb and (n % max(1, cfg.n_steps // 50) == 0 or n == cfg.n_steps):
                progress_cb(n, cfg.n_steps, t_now)

        res.wall_time = time.perf_counter() - t_start
        res.n_steps = cfg.n_steps
        return res


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de verificación
# ─────────────────────────────────────────────────────────────────────────────
def run_reference(Nx=101, courant=0.5, T_max=2.828427, n_snapshots=60,
                  progress_cb=None) -> FDTDResult:
    """Corrida de referencia: cavidad vacía TM₁₁ (mismo problema que el PINN)."""
    cfg = FDTDConfig(Nx=Nx, Ny=Nx, courant=courant, T_max=T_max,
                     n_snapshots=n_snapshots)
    return FDTDSolver(cfg).run(progress_cb=progress_cb)


if __name__ == "__main__":
    # Auto-verificación contra la solución analítica TM₁₁
    from analytical import Ez_exact

    print("=" * 60)
    print("  VERIFICACIÓN FDTD 2D TM — cavidad PEC, modo TM₁₁")
    print("=" * 60)

    for Nx in (51, 101, 201):
        res = run_reference(Nx=Nx)
        t_arr, Ez_s, _, _ = res.to_arrays()
        x = np.linspace(0, 1, Nx)
        Xg, Yg = np.meshgrid(x, x, indexing="ij")

        errs = []
        for k, t in enumerate(t_arr):
            Ez_a = Ez_exact(Xg, Yg, t, 1, 1)
            denom = np.linalg.norm(Ez_a)
            if denom > 1e-12:
                errs.append(np.linalg.norm(Ez_s[k] - Ez_a) / denom * 100.0)
        e_mean = float(np.mean(errs))

        E = np.asarray(res.energy)
        drift = float(abs(E[-1] - E[0]) / E[0] * 100.0)
        print(f"  Nx={Nx:4d}  pasos={res.n_steps:5d}  "
              f"L2 medio={e_mean:7.4f}%  deriva energía={drift:.4f}%  "
              f"t_wall={res.wall_time:.2f}s  mem={res.memory_bytes/1e6:.1f}MB")

    print("\n  (El error L2 debe decrecer ~O(dx²) al refinar la malla.)")
