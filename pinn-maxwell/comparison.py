"""
comparison.py — Comparación cuantitativa PINN vs FDTD (y vs analítica).

Flujo:
    1. Ejecutar FDTD sobre la configuración del proyecto  → snapshots (t_k, Ez, Bx, By)
    2. Evaluar el PINN en la MISMA malla y los MISMOS tiempos
    3. Calcular mapas de error y métricas globales:
         - Error L2 relativo (%)          ‖u_PINN − u_FDTD‖₂ / ‖u_FDTD‖₂ × 100
         - Error RMS (%)                  RMS(u_PINN − u_FDTD) / RMS(u_FDTD) × 100
         - Error máximo (%)               max|Δ| / max|u_FDTD| × 100
         - Error vs tiempo, tiempos de cómputo, memoria, iteraciones

Notas de validez científica:
    El PINN fue entrenado para la cavidad vacía TM₁₁ en t ∈ [0, T_MAX].
    Si el proyecto añade materiales o fuentes, el PINN se evalúa con el
    escalado t_eff = t/√(ε_r·μ_r) (aproximación de medio homogéneo) y la
    comparación reporta honestamente el error mayor resultante.
"""
from __future__ import annotations

import time
import math
import numpy as np

from fdtd import FDTDConfig, FDTDSolver, FDTDResult


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────
def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = np.linalg.norm(ref.ravel())
    if denom < 1e-14:
        return 0.0
    return float(np.linalg.norm((pred - ref).ravel()) / denom * 100.0)


def rel_rms(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = math.sqrt(float(np.mean(ref**2)))
    if denom < 1e-14:
        return 0.0
    return float(math.sqrt(float(np.mean((pred - ref)**2))) / denom * 100.0)


def rel_max(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    if denom < 1e-14:
        return 0.0
    return float(np.max(np.abs(pred - ref)) / denom * 100.0)


def abs_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - ref)**2)))


def energy_series(Ez, Bx, By, dx, dy):
    """U(t) = ½ ∬ (Ez² + Bx² + By²) dA con regla del trapecio 2D.
    Ez/Bx/By: tensores (K, Nx, Ny). Retorna array (K,)."""
    K, Nx, Ny = Ez.shape
    w = np.ones((Nx, Ny))
    w[0, :] *= 0.5; w[-1, :] *= 0.5
    w[:, 0] *= 0.5; w[:, -1] *= 0.5
    u = 0.5 * (Ez**2 + Bx**2 + By**2)
    return np.sum(u * w[None, :, :], axis=(1, 2)) * dx * dy


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación del PINN sobre la malla FDTD
# ─────────────────────────────────────────────────────────────────────────────
def eval_pinn_on_grid(infer_fn, Nx: int, Ny: int, L: float, times,
                      eps_r: float = 1.0, mu_r: float = 1.0,
                      batch: int = 20000, progress_cb=None):
    """
    Evalúa el PINN en la malla (Nx × Ny) para cada tiempo.

    infer_fn(x_flat, y_flat, t_flat) → (Ez, Bx, By) arrays 1D
    Retorna (Ez[k], Bx[k], By[k]) apilados: shape (K, Nx, Ny), y wall_time.
    """
    x = np.linspace(0.0, L, Nx).astype(np.float32)
    y = np.linspace(0.0, L, Ny).astype(np.float32)
    Xg, Yg = np.meshgrid(x, y, indexing="ij")
    xf, yf = Xg.ravel(), Yg.ravel()
    scale = math.sqrt(max(eps_r * mu_r, 1e-12))

    K = len(times)
    Ez_all = np.empty((K, Nx, Ny), dtype=np.float64)
    Bx_all = np.empty((K, Nx, Ny), dtype=np.float64)
    By_all = np.empty((K, Nx, Ny), dtype=np.float64)

    t0 = time.perf_counter()
    for k, t in enumerate(times):
        te = float(t) / scale
        tf = np.full_like(xf, te, dtype=np.float32)
        Ez_l, Bx_l, By_l = [], [], []
        for s in range(0, xf.size, batch):
            e, bx, by = infer_fn(xf[s:s+batch], yf[s:s+batch], tf[s:s+batch])
            Ez_l.append(e); Bx_l.append(bx); By_l.append(by)
        Ez_all[k] = np.concatenate(Ez_l).reshape(Nx, Ny)
        Bx_all[k] = np.concatenate(Bx_l).reshape(Nx, Ny)
        By_all[k] = np.concatenate(By_l).reshape(Nx, Ny)
        if progress_cb:
            progress_cb(k + 1, K)
    wall = time.perf_counter() - t0
    return Ez_all, Bx_all, By_all, wall


# ─────────────────────────────────────────────────────────────────────────────
# Comparación completa
# ─────────────────────────────────────────────────────────────────────────────
def compare(cfg: FDTDConfig, infer_fn, pinn_meta: dict | None = None,
            log=None, cancel_cb=None) -> dict:
    """
    Ejecuta FDTD + PINN sobre la misma malla/tiempos y retorna un dict
    serializable con snapshots, mapas de error, series y métricas.

    Args:
        cfg:       configuración FDTD del proyecto.
        infer_fn:  closure de inferencia del PINN (o None → solo FDTD).
        pinn_meta: {"epochs": 12000, "params": N, "train_time": s} opcional.
        log:       callable(str) para la consola.
    """
    _log = log or (lambda m: None)

    # ── 1. FDTD ────────────────────────────────────────────────────────────
    _log(f"[FDTD] Malla {cfg.Nx}×{cfg.Ny}  dt={cfg.dt:.5f}  pasos={cfg.n_steps}  "
         f"Courant S={cfg.courant:.2f}")
    solver = FDTDSolver(cfg)

    def fdtd_prog(n, total, t):
        _log(f"[FDTD] paso {n}/{total}  t={t:.3f}")

    fdtd_res = solver.run(progress_cb=fdtd_prog, cancel_cb=cancel_cb)
    t_arr, Ez_f, Bx_f, By_f = fdtd_res.to_arrays()
    _log(f"[FDTD] Completado en {fdtd_res.wall_time:.2f}s  "
         f"({fdtd_res.memory_bytes/1e6:.1f} MB)")

    out = {
        "times": t_arr.tolist(),
        "grid": {"Nx": cfg.Nx, "Ny": cfg.Ny, "L": cfg.L,
                 "dx": cfg.dx, "dy": cfg.dy, "dt": cfg.dt},
        "fdtd": {
            "wall_time": fdtd_res.wall_time,
            "n_steps": fdtd_res.n_steps,
            "memory_bytes": fdtd_res.memory_bytes,
            "energy_t": fdtd_res.energy_t,
            "energy": fdtd_res.energy,
            "probe_t": fdtd_res.probe_t,
            "probes": fdtd_res.probes,
        },
        "snapshots": {
            "Ez_fdtd": Ez_f, "Bx_fdtd": Bx_f, "By_fdtd": By_f,   # np arrays (K,Nx,Ny)
        },
    }

    if infer_fn is None:
        return out

    # ── 2. PINN en la misma malla/tiempos ──────────────────────────────────
    _log(f"[PINN] Evaluando red en {len(t_arr)} instantes × {cfg.Nx*cfg.Ny} nodos…")

    def pinn_prog(k, K):
        if k % max(1, K // 10) == 0 or k == K:
            _log(f"[PINN] instante {k}/{K}")

    Ez_p, Bx_p, By_p, pinn_wall = eval_pinn_on_grid(
        infer_fn, cfg.Nx, cfg.Ny, cfg.L, t_arr,
        eps_r=cfg.eps_r_bg, mu_r=cfg.mu_r_bg, progress_cb=pinn_prog)
    _log(f"[PINN] Completado en {pinn_wall:.2f}s")

    # ── 3. Errores ─────────────────────────────────────────────────────────
    _log("[COMPARE] Calculando mapas de error y métricas…")
    err_abs = np.abs(Ez_p - Ez_f)                                  # (K,Nx,Ny)
    ref_max = np.max(np.abs(Ez_f))
    err_rel = err_abs / (ref_max if ref_max > 1e-14 else 1.0) * 100.0

    # Series de error por instante
    l2_t  = [rel_l2(Ez_p[k], Ez_f[k]) for k in range(len(t_arr))]
    rms_t = [rel_rms(Ez_p[k], Ez_f[k]) for k in range(len(t_arr))]
    max_t = [rel_max(Ez_p[k], Ez_f[k]) for k in range(len(t_arr))]

    # Métricas globales (sobre el tensor espaciotemporal completo)
    metrics = {
        "l2_Ez": rel_l2(Ez_p, Ez_f),
        "l2_Bx": rel_l2(Bx_p, Bx_f),
        "l2_By": rel_l2(By_p, By_f),
        "l2_abs_Ez": abs_l2(Ez_p, Ez_f),
        "rms_Ez": rel_rms(Ez_p, Ez_f),
        "max_Ez": rel_max(Ez_p, Ez_f),
        "pinn_wall_time": pinn_wall,
        "fdtd_wall_time": fdtd_res.wall_time,
        "speedup_inference": fdtd_res.wall_time / pinn_wall if pinn_wall > 0 else 0.0,
        "fdtd_memory_bytes": fdtd_res.memory_bytes,
        "fdtd_steps": fdtd_res.n_steps,
    }
    if pinn_meta:
        metrics.update({
            "pinn_epochs": pinn_meta.get("epochs"),
            "pinn_params": pinn_meta.get("params"),
            "pinn_train_time": pinn_meta.get("train_time"),
            "pinn_memory_bytes": pinn_meta.get("memory_bytes"),
        })

    # Energía del PINN (para la gráfica de conservación)
    energy_pinn = energy_series(Ez_p, Bx_p, By_p, cfg.dx, cfg.dy)

    out["snapshots"].update({
        "Ez_pinn": Ez_p, "Bx_pinn": Bx_p, "By_pinn": By_p,
        "err_abs": err_abs, "err_rel": err_rel,
    })
    out["pinn"] = {
        "wall_time": pinn_wall,
        "energy_t": t_arr.tolist(),
        "energy": energy_pinn.tolist(),
    }
    out["error_series"] = {"t": t_arr.tolist(), "l2": l2_t, "rms": rms_t, "max": max_t}
    out["metrics"] = metrics

    _log(f"[COMPARE] L2={metrics['l2_Ez']:.3f}%  RMS={metrics['rms_Ez']:.3f}%  "
         f"máx={metrics['max_Ez']:.3f}%")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Comparación contra la analítica (autoverificación sin PINN)
# ─────────────────────────────────────────────────────────────────────────────
def compare_with_analytical(cfg: FDTDConfig, log=None) -> dict:
    """Usa la solución exacta TM₁₁ como 'PINN' — para validar el pipeline."""
    from analytical import Ez_exact, Bx_exact, By_exact

    def infer(xf, yf, tf):
        t = float(tf[0])
        return (Ez_exact(xf, yf, t, 1, 1).astype(np.float64),
                Bx_exact(xf, yf, t, 1, 1).astype(np.float64),
                By_exact(xf, yf, t, 1, 1).astype(np.float64))

    return compare(cfg, infer, pinn_meta={"epochs": 0}, log=log)


if __name__ == "__main__":
    print("=" * 60)
    print("  PIPELINE DE COMPARACIÓN — FDTD vs analítica TM11")
    print("=" * 60)
    cfg = FDTDConfig(Nx=101, Ny=101, n_snapshots=30)
    res = compare_with_analytical(cfg, log=lambda m: None)
    m = res["metrics"]
    print(f"  L2  Ez : {m['l2_Ez']:.4f} %")
    print(f"  RMS Ez : {m['rms_Ez']:.4f} %")
    print(f"  Max Ez : {m['max_Ez']:.4f} %")
    print(f"  FDTD   : {m['fdtd_wall_time']:.2f} s, {m['fdtd_steps']} pasos")
    print("  (Valores ~0.1% confirman pipeline y solver correctos.)")
