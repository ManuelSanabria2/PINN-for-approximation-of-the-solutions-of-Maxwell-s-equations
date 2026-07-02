"""
exporters.py — Exportación de resultados de simulación a formatos científicos.

Formatos soportados (todos retornan `bytes` listos para descarga HTTP):

    CSV      — campo en malla o series temporales
    VTK      — legacy ASCII STRUCTURED_POINTS (ParaView / VisIt)
    MATLAB   — .mat v5 (scipy.io.savemat)
    NumPy    — .npz comprimido
    HDF5     — .h5 (h5py)
    PNG      — figura matplotlib del campo elegido
    GIF      — animación matplotlib (PillowWriter)
    PDF      — informe técnico multipágina (PdfPages)
    STL      — geometrías (caja / esfera extruida) en ASCII STL
"""
from __future__ import annotations

import io
import math
import datetime
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import animation


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────
def field_to_csv(field: np.ndarray, L: float = 1.0, name: str = "Ez",
                 t: float = 0.0) -> bytes:
    """Malla (Nx, Ny) → CSV largo: x, y, t, valor."""
    Nx, Ny = field.shape
    x = np.linspace(0, L, Nx)
    y = np.linspace(0, L, Ny)
    buf = io.StringIO()
    buf.write(f"x,y,t,{name}\n")
    for i in range(Nx):
        for j in range(Ny):
            buf.write(f"{x[i]:.6g},{y[j]:.6g},{t:.6g},{field[i, j]:.9g}\n")
    return buf.getvalue().encode("utf-8")


def series_to_csv(columns: dict[str, list | np.ndarray]) -> bytes:
    """{"t": [...], "Ez": [...], ...} → CSV por columnas."""
    keys = list(columns.keys())
    arrays = [np.asarray(columns[k], dtype=float) for k in keys]
    n = min(a.size for a in arrays)
    buf = io.StringIO()
    buf.write(",".join(keys) + "\n")
    for r in range(n):
        buf.write(",".join(f"{a[r]:.9g}" for a in arrays) + "\n")
    return buf.getvalue().encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# VTK legacy ASCII (STRUCTURED_POINTS, compatible con ParaView)
# ─────────────────────────────────────────────────────────────────────────────
def to_vtk(fields: dict[str, np.ndarray], L: float = 1.0,
           title: str = "Maxwell PINN vs FDTD", t: float = 0.0) -> bytes:
    """
    fields: {"Ez_fdtd": (Nx,Ny), "Ez_pinn": ..., "err_abs": ...}
    Escribe una malla 2D (Nz=1) con un SCALARS por campo.
    """
    first = next(iter(fields.values()))
    Nx, Ny = first.shape
    dx = L / (Nx - 1)
    dy = L / (Ny - 1)
    buf = io.StringIO()
    buf.write("# vtk DataFile Version 3.0\n")
    buf.write(f"{title} | t={t:.6g}\n")
    buf.write("ASCII\nDATASET STRUCTURED_POINTS\n")
    buf.write(f"DIMENSIONS {Nx} {Ny} 1\n")
    buf.write("ORIGIN 0 0 0\n")
    buf.write(f"SPACING {dx:.9g} {dy:.9g} 1\n")
    buf.write(f"POINT_DATA {Nx * Ny}\n")
    for name, arr in fields.items():
        buf.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
        # VTK espera orden x-más-rápido: recorrer j (y) exterior, i (x) interior
        for j in range(Ny):
            row = " ".join(f"{arr[i, j]:.6e}" for i in range(Nx))
            buf.write(row + "\n")
    return buf.getvalue().encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# MATLAB / NumPy / HDF5
# ─────────────────────────────────────────────────────────────────────────────
def to_mat(data: dict) -> bytes:
    from scipy.io import savemat
    buf = io.BytesIO()
    clean = {k.replace("-", "_"): np.asarray(v) for k, v in data.items() if v is not None}
    savemat(buf, clean, do_compression=True)
    return buf.getvalue()


def to_npz(data: dict) -> bytes:
    buf = io.BytesIO()
    clean = {k: np.asarray(v) for k, v in data.items() if v is not None}
    np.savez_compressed(buf, **clean)
    return buf.getvalue()


def to_hdf5(data: dict, attrs: dict | None = None) -> bytes:
    import h5py
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        for k, v in data.items():
            if v is None:
                continue
            f.create_dataset(k, data=np.asarray(v), compression="gzip")
        for k, v in (attrs or {}).items():
            try:
                f.attrs[k] = v
            except TypeError:
                f.attrs[k] = str(v)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Imágenes / animaciones (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
_CMAP = "RdBu_r"

def field_png(field: np.ndarray, L: float = 1.0, title: str = "Ez",
              t: float = 0.0, cmap: str = _CMAP, symmetric: bool = True,
              dpi: int = 150) -> bytes:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    vmax = float(np.max(np.abs(field))) or 1.0
    kw = dict(vmin=-vmax, vmax=vmax) if symmetric else {}
    im = ax.imshow(field.T, origin="lower", extent=[0, L, 0, L],
                   cmap=cmap, **kw)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"{title}   (t = {t:.3f})")
    fig.colorbar(im, ax=ax, shrink=0.85)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def comparison_png(Ez_p: np.ndarray, Ez_f: np.ndarray, L: float = 1.0,
                   t: float = 0.0, dpi: int = 150) -> bytes:
    """Cuadrícula 2×2: PINN | FDTD | error abs | métricas."""
    err = np.abs(Ez_p - Ez_f)
    vmax = max(float(np.max(np.abs(Ez_f))), 1e-12)
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 9))
    for ax, (arr, ttl, cm, vmn, vmx) in zip(
            axs.flat[:3],
            [(Ez_p, "PINN — Ez", _CMAP, -vmax, vmax),
             (Ez_f, "FDTD — Ez", _CMAP, -vmax, vmax),
             (err, "|PINN − FDTD|", "magma", 0, float(err.max()) or 1e-12)]):
        im = ax.imshow(arr.T, origin="lower", extent=[0, L, 0, L],
                       cmap=cm, vmin=vmn, vmax=vmx)
        ax.set_title(ttl); ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axs.flat[3]; ax.axis("off")
    denom = np.linalg.norm(Ez_f.ravel())
    l2 = np.linalg.norm((Ez_p - Ez_f).ravel()) / denom * 100 if denom > 1e-14 else 0
    rms = math.sqrt(np.mean((Ez_p - Ez_f)**2)) / math.sqrt(np.mean(Ez_f**2)) * 100 \
        if np.mean(Ez_f**2) > 1e-20 else 0
    mx = err.max() / vmax * 100
    ax.text(0.05, 0.85, f"t = {t:.4f}", fontsize=13, weight="bold")
    ax.text(0.05, 0.65, f"Error L2      : {l2:.4f} %", fontsize=12, family="monospace")
    ax.text(0.05, 0.50, f"Error RMS     : {rms:.4f} %", fontsize=12, family="monospace")
    ax.text(0.05, 0.35, f"Error máximo  : {mx:.4f} %", fontsize=12, family="monospace")
    fig.suptitle("Comparación PINN vs FDTD — Maxwell 2D TM", fontsize=14)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def animation_gif(frames: np.ndarray, times, L: float = 1.0,
                  title: str = "Ez", fps: int = 12, dpi: int = 80) -> bytes:
    """frames: (K, Nx, Ny) → GIF animado."""
    import tempfile, os
    vmax = float(np.max(np.abs(frames))) or 1.0
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(frames[0].T, origin="lower", extent=[0, L, 0, L],
                   cmap=_CMAP, vmin=-vmax, vmax=vmax)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ttl = ax.set_title(f"{title}   t = {times[0]:.3f}")
    fig.colorbar(im, ax=ax, shrink=0.85)

    def update(k):
        im.set_data(frames[k].T)
        ttl.set_text(f"{title}   t = {times[k]:.3f}")
        return [im, ttl]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        path = tmp.name
    try:
        ani.save(path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        with open(path, "rb") as f:
            data = f.read()
    finally:
        plt.close(fig)
        try:
            os.unlink(path)
        except OSError:
            pass
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Informe PDF técnico
# ─────────────────────────────────────────────────────────────────────────────
def technical_pdf(result: dict, project: dict | None = None) -> bytes:
    """
    Informe multipágina a partir del dict de comparison.compare():
      Pág 1: portada + configuración + métricas
      Pág 2: comparación 2×2 en t medio
      Pág 3: series (error vs t, energía, sonda)
    """
    snaps = result.get("snapshots", {})
    times = np.asarray(result.get("times", []))
    has_pinn = "Ez_pinn" in snaps
    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # ── Portada ─────────────────────────────────────────────────────
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.92, "Informe Técnico de Simulación",
                 ha="center", fontsize=20, weight="bold")
        fig.text(0.5, 0.88, "Ecuaciones de Maxwell 2D TM — PINN vs FDTD",
                 ha="center", fontsize=13, color="#444")
        fig.text(0.5, 0.85, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                 ha="center", fontsize=10, color="#888")

        g = result.get("grid", {})
        m = result.get("metrics", {})
        lines = [
            "CONFIGURACIÓN",
            f"  Malla                : {g.get('Nx')} × {g.get('Ny')}",
            f"  Dominio              : [0, {g.get('L', 1.0)}]²  (unidades naturales, c=1)",
            f"  dx, dy               : {g.get('dx', 0):.5f}, {g.get('dy', 0):.5f}",
            f"  dt (CFL)             : {g.get('dt', 0):.6f}",
            f"  Pasos FDTD           : {m.get('fdtd_steps', result.get('fdtd', {}).get('n_steps', '—'))}",
            "",
        ]
        if has_pinn and m:
            lines += [
                "MÉTRICAS PINN vs FDTD (campo Ez, tensor espaciotemporal)",
                f"  Error L2 relativo    : {m.get('l2_Ez', 0):.4f} %",
                f"  Error RMS            : {m.get('rms_Ez', 0):.4f} %",
                f"  Error máximo         : {m.get('max_Ez', 0):.4f} %",
                f"  L2 Bx / By           : {m.get('l2_Bx', 0):.4f} % / {m.get('l2_By', 0):.4f} %",
                "",
                "RENDIMIENTO",
                f"  Tiempo PINN (infer.) : {m.get('pinn_wall_time', 0):.2f} s",
                f"  Tiempo FDTD          : {m.get('fdtd_wall_time', 0):.2f} s",
                f"  Memoria FDTD         : {m.get('fdtd_memory_bytes', 0)/1e6:.1f} MB",
                f"  Épocas PINN          : {m.get('pinn_epochs', '—')}",
                f"  Parámetros PINN      : {m.get('pinn_params', '—')}",
            ]
        if project:
            lines += ["", "PROYECTO",
                      f"  Nombre               : {project.get('name', '—')}"]
            for mat in project.get("materials", []):
                lines.append(f"  Material             : {mat.get('name')}  "
                             f"εr={mat.get('eps_r')}  μr={mat.get('mu_r')}  "
                             f"σ={mat.get('sigma')}")
        fig.text(0.1, 0.78, "\n".join(lines), fontsize=9.5,
                 family="monospace", va="top")
        pdf.savefig(fig); plt.close(fig)

        # ── Comparación 2×2 ─────────────────────────────────────────────
        if has_pinn and len(times):
            kmid = len(times) // 2
            png = comparison_png(snaps["Ez_pinn"][kmid], snaps["Ez_fdtd"][kmid],
                                 L=g.get("L", 1.0), t=float(times[kmid]))
            import matplotlib.image as mpimg
            img = mpimg.imread(io.BytesIO(png))
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.05, 0.15, 0.9, 0.75])
            ax.imshow(img); ax.axis("off")
            pdf.savefig(fig); plt.close(fig)

        # ── Series ──────────────────────────────────────────────────────
        fig, axs = plt.subplots(3, 1, figsize=(8.27, 11.69))
        es = result.get("error_series")
        if es:
            axs[0].plot(es["t"], es["l2"], label="L2 %")
            axs[0].plot(es["t"], es["rms"], label="RMS %")
            axs[0].plot(es["t"], es["max"], label="máx %")
            axs[0].set_title("Error PINN vs FDTD en el tiempo")
            axs[0].set_xlabel("t"); axs[0].set_ylabel("%"); axs[0].legend()
        fd = result.get("fdtd", {})
        if fd.get("energy"):
            axs[1].plot(fd["energy_t"], fd["energy"], label="FDTD")
            pn = result.get("pinn", {})
            if pn.get("energy"):
                axs[1].plot(pn["energy_t"], pn["energy"], "--", label="PINN")
            axs[1].set_title("Conservación de energía electromagnética")
            axs[1].set_xlabel("t"); axs[1].set_ylabel("U"); axs[1].legend()
        if fd.get("probes"):
            name0 = next(iter(fd["probes"]))
            axs[2].plot(fd["probe_t"], fd["probes"][name0])
            axs[2].set_title(f"Sonda Ez(t) — {name0}")
            axs[2].set_xlabel("t"); axs[2].set_ylabel("Ez")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# STL de geometrías
# ─────────────────────────────────────────────────────────────────────────────
def _facet(buf, n, v1, v2, v3):
    buf.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n    outer loop\n")
    for v in (v1, v2, v3):
        buf.write(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
    buf.write("    endloop\n  endfacet\n")


def geometry_stl(objects: list[dict], L: float = 1.0, height: float = 0.2) -> bytes:
    """
    Exporta la geometría del proyecto como STL ASCII.
    - El dominio (caja L×L×height) siempre se incluye.
    - Esferas → esfera 3D centrada en el plano medio; cajas → prisma.
    """
    buf = io.StringIO()
    buf.write("solid fluxia_geometry\n")

    def box(x0, y0, z0, x1, y1, z1):
        # 12 triángulos
        p = [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
             (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)]
        quads = [((0,1,2,3),(0,0,-1)), ((4,7,6,5),(0,0,1)),
                 ((0,4,5,1),(0,-1,0)), ((2,6,7,3),(0,1,0)),
                 ((1,5,6,2),(1,0,0)),  ((0,3,7,4),(-1,0,0))]
        for (a,b,c,d), n in quads:
            _facet(buf, n, p[a], p[b], p[c])
            _facet(buf, n, p[a], p[c], p[d])

    def sphere(cx, cy, cz, r, nu=24, nv=16):
        for i in range(nu):
            for j in range(nv):
                th0, th1 = 2*math.pi*i/nu, 2*math.pi*(i+1)/nu
                ph0, ph1 = math.pi*j/nv, math.pi*(j+1)/nv
                def pt(th, ph):
                    return (cx + r*math.sin(ph)*math.cos(th),
                            cy + r*math.sin(ph)*math.sin(th),
                            cz + r*math.cos(ph))
                a, b, c, d = pt(th0,ph0), pt(th1,ph0), pt(th1,ph1), pt(th0,ph1)
                nrm = ((a[0]-cx)/r, (a[1]-cy)/r, (a[2]-cz)/r)
                _facet(buf, nrm, a, b, c)
                _facet(buf, nrm, a, c, d)

    box(0, 0, 0, L, L, height)          # dominio computacional
    for obj in objects or []:
        sh = obj.get("shape", "sphere")
        if sh in ("sphere", "circle"):
            sphere(obj.get("cx", .5), obj.get("cy", .5), height/2,
                   min(obj.get("r", .1), height/2 * 0.99 + obj.get("r", .1)))
        elif sh == "box":
            box(obj.get("x0", 0), obj.get("y0", 0), 0,
                obj.get("x1", 1), obj.get("y1", 1), height)
    buf.write("endsolid fluxia_geometry\n")
    return buf.getvalue().encode("utf-8")


if __name__ == "__main__":
    # Prueba rápida de todos los exportadores
    print("Probando exportadores…")
    Nx = 41
    x = np.linspace(0, 1, Nx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Ez = np.sin(np.pi * X) * np.sin(np.pi * Y)
    Ezp = Ez * 1.001

    print(f"  CSV    : {len(field_to_csv(Ez)):8d} bytes")
    print(f"  VTK    : {len(to_vtk({'Ez': Ez, 'err': np.abs(Ezp-Ez)})):8d} bytes")
    print(f"  MAT    : {len(to_mat({'Ez': Ez, 't': 0.0})):8d} bytes")
    print(f"  NPZ    : {len(to_npz({'Ez': Ez})):8d} bytes")
    print(f"  HDF5   : {len(to_hdf5({'Ez': Ez}, {'solver': 'fdtd'})):8d} bytes")
    print(f"  PNG    : {len(field_png(Ez)):8d} bytes")
    print(f"  CMP-PNG: {len(comparison_png(Ezp, Ez)):8d} bytes")
    frames = np.stack([Ez * math.cos(0.3 * k) for k in range(8)])
    print(f"  GIF    : {len(animation_gif(frames, [0.3*k for k in range(8)])):8d} bytes")
    fake = {
        "times": [0.0, 0.5], "grid": {"Nx": Nx, "Ny": Nx, "L": 1.0, "dx": .025, "dy": .025, "dt": .01},
        "snapshots": {"Ez_pinn": np.stack([Ezp, Ezp]), "Ez_fdtd": np.stack([Ez, Ez])},
        "metrics": {"l2_Ez": .1, "rms_Ez": .1, "max_Ez": .2, "l2_Bx": .1, "l2_By": .1,
                    "pinn_wall_time": 1, "fdtd_wall_time": 2, "fdtd_memory_bytes": 8e5,
                    "fdtd_steps": 500, "pinn_epochs": 12000, "pinn_params": 50000},
        "error_series": {"t": [0, .5], "l2": [.1, .1], "rms": [.1, .1], "max": [.2, .2]},
        "fdtd": {"energy_t": [0, .5], "energy": [1, 1], "probe_t": [0, .5],
                 "probes": {"centro": [1, .9]}, "n_steps": 500},
        "pinn": {"energy_t": [0, .5], "energy": [1, 1]},
    }
    print(f"  PDF    : {len(technical_pdf(fake)):8d} bytes")
    print(f"  STL    : {len(geometry_stl([{'shape': 'sphere', 'cx': .5, 'cy': .5, 'r': .15}])):8d} bytes")
    print("OK — todos los exportadores funcionan.")
