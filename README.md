---
title: Maxwell PINN Faraday Digital
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Fluxia — PINN vs FDTD para las Ecuaciones de Maxwell

> Entorno profesional de simulación electromagnética (estilo CAD) que resuelve las **Ecuaciones de Maxwell 2D TM** con dos métodos — una **Red Neuronal Informada por la Física (PINN)** y un **solver FDTD (esquema de Yee)** — y los compara célula a célula, en la misma malla y los mismos instantes de tiempo.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.1xx-green?logo=fastapi)
![Three.js](https://img.shields.io/badge/Three.js-r165-black?logo=threedotjs)
![Error L2](https://img.shields.io/badge/PINN%20vs%20FDTD-L2%20%3D%200.12%25-brightgreen)

---

## ¿Qué es?

La mayoría de las herramientas permiten ejecutar distintos solvers; **muy pocas están diseñadas para evaluar directamente cómo una PINN reproduce una solución FDTD**. Fluxia hace exactamente eso:

- **PINN** (`model.py`, `physics.py`): red FCN `(x,y,t) → (Ez,Bx,By)` con Fourier Feature Encoding, entrenada minimizando los residuales de Maxwell + condiciones PEC + condición inicial TM₁₁.
- **FDTD** (`fdtd.py`): solver Yee 2D TM de orden 2, condición CFL, materiales heterogéneos (ε_r, μ_r, σ), PEC, PML graduada, fuentes de pulso gaussiano y conservación de energía verificada.
- **Comparación** (`comparison.py`): el PINN se evalúa en la malla y los tiempos exactos del FDTD → error L2 / RMS / máximo, mapas de error, series temporales, tiempos de cómputo, memoria e iteraciones.

### Problema físico

Modo TM en una cavidad PEC `[0,L]²` (unidades naturales, c = ε₀ = μ₀ = 1):

$$\frac{\partial B_x}{\partial t} = -\frac{\partial E_z}{\partial y}, \qquad \frac{\partial B_y}{\partial t} = \frac{\partial E_z}{\partial x} \quad \text{(Faraday)}$$

$$\varepsilon\frac{\partial E_z}{\partial t} = \frac{\partial B_y}{\partial x} - \frac{\partial B_x}{\partial y} - \sigma E_z \quad \text{(Ampère–Maxwell)}$$

Condición inicial (modo TM₁₁): `Ez = E₀·sin(πx/L)·sin(πy/L)`, `B = 0`. Existe solución analítica exacta (`analytical.py`), usada para validar ambos solvers de forma independiente.

---

## Resultados verificados

**PINN vs FDTD** (malla 101×101, 566 pasos, 2 periodos TM₁₁, todo el tensor espacio-temporal):

| Métrica | Valor |
|---|---|
| Error L2 (Ez) | **0.118 %** |
| Error RMS (Ez) | **0.118 %** |
| Error máximo (Ez) | **1.24 %** |
| Error L2 (Bx / By) | 1.12 % / 1.12 % |
| PINN | 58 243 parámetros · 12 000 épocas · 0.2 MB |
| FDTD | 566 pasos · 0.8 MB |

**FDTD vs solución analítica** (convergencia de orden 2 y conservación de energía):

| Malla | Error L2 medio | Deriva de energía |
|---|---|---|
| 51×51 | 0.216 % | 0.000 % |
| 101×101 | 0.058 % | 0.000 % |
| 201×201 | 0.011 % | 0.000 % |

---

## Inicio rápido

```bash
git clone https://github.com/ManuelSanabria2/PINN-for-approximation-of-the-solutions-of-Maxwell-s-equations.git
cd PINN-for-approximation-of-the-solutions-of-Maxwell-s-equations
pip install -r requirements.txt

cd pinn-maxwell
uvicorn server:app --port 8000
# → http://localhost:8000/demo
```

En Windows basta con doble clic en `start_demo.bat`.

**Prueba en 30 segundos:** menú **Simulación → Ejecutar ambos (F7)** → la pestaña *Comparación PINN vs FDTD* muestra 4 paneles sincronizados (PINN | FDTD | mapa de error | métricas) que se mueven juntos con el slider temporal.

---

## La interfaz

```
┌────────────────────────── Menús ──────────────────────────────┐
│ Archivo · Simulación · Herramientas · Visualización · Ayuda   │
├───────────┬──────────────────────────────────┬────────────────┤
│ Explorador│  Vista 3D  /  Comparación 2×2    │  Propiedades   │
│ (árbol    │  rotar·zoom·pan·selección·medir  │  contextuales  │
│  CAD)     │  ⏮ ◀ ▶ ⏸ ⏭  slider  0.25×–10×   │                │
├───────────┴──────────────────────────────────┴────────────────┤
│ Consola · Entrenamiento PINN (en vivo) · Gráficas             │
└───────────────────────────────────────────────────────────────┘
```

| Componente | Detalle |
|---|---|
| **Explorador del proyecto** | Geometría (caja, esfera, antena) · Materiales (aire, PEC, silicio) · Fuentes (modo TM₁₁, pulso gaussiano) · Monitores · Solver PINN · Solver FDTD · Resultados. Ojo ◉ para mostrar/ocultar. |
| **Vista 3D** (Three.js) | Dominio transparente, malla, materiales coloreados, fuente como flecha roja, PML, campo como superficie desplazada, isolíneas, cortes XY/XZ/YZ, ejes, medición de distancias. |
| **Propiedades** | Material: ε_r, μ_r, σ, color, modelo dispersivo · Fuente: tipo, frecuencia, potencia, τ, polarización · Dominio: Nx, Ny, Δx, Δy, Courant, T máx. |
| **Comparación 2×2** | PINN \| FDTD \| error (absoluto/relativo, escala auto/global) \| métricas — sincronizados por un slider temporal. |
| **Entrenamiento PINN** | Época, Loss PDE/BC/IC, learning rate y tiempo con gráficas log **en vivo** (WebSocket), estilo TensorBoard. Reentrenable desde la UI. |
| **Gráficas** | Ez(t) en sonda · Espectro FFT (pico en f = ω/2π ≈ 0.707) · Energía U(t) · Error vs tiempo · Residuales PDE vía autograd. |
| **Consola** | Streaming en vivo del backend: compilación de geometría, generación de malla, pasos FDTD, épocas de entrenamiento, errores. |
| **Herramientas** | Generador de malla (puntos por λ) · Refinamiento adaptativo (extrapolación de Richardson) · Inspector de materiales · Configuración GPU/hilos. |
| **Exportación** | PDF técnico · CSV · NumPy (.npz) · HDF5 · VTK (ParaView) · MATLAB (.mat) · PNG · GIF · video WebM del viewport · STL de geometrías. |

---

## Estructura del proyecto

```
pinn-maxwell/
├── fdtd.py          # Solver FDTD Yee 2D TM (materiales, PML, fuentes, energía)
├── comparison.py    # Comparación PINN vs FDTD en la misma malla/tiempos
├── exporters.py     # CSV, VTK, MATLAB, NumPy, HDF5, PNG, GIF, PDF, STL
├── server.py        # Backend FastAPI: jobs, WebSocket, entrenamiento en vivo, export
├── model.py         # FCN + Fourier Feature Encoding
├── physics.py       # Residuales de Maxwell + función de pérdida PINN
├── train.py         # Entrenamiento Adam + L-BFGS
├── sampling.py      # Muestreo de colocación / frontera / inicial
├── analytical.py    # Solución analítica exacta TM_mn
├── main.py          # Pipeline CLI de entrenamiento/evaluación
└── demo/            # Frontend Fluxia (sin frameworks)
    ├── index.html   #   Layout CAD: menús, árbol, viewport, propiedades, dock
    ├── style.css    #   Tema oscuro de ingeniería
    └── js/
        ├── app.js       # Arranque y orquestación
        ├── api.js       # Cliente REST + WebSocket
        ├── state.js     # Estado central (proyecto, selección, frames)
        ├── viewport.js  # Escena 3D (Three.js)
        ├── ui.js        # Árbol, propiedades, consola, menús
        ├── panels.js    # Entrenamiento, comparación 2×2, gráficas, reproductor
        └── charts.js    # Gráficas de línea + heatmaps + FFT (canvas puro)
```

## API principal

```
POST /api/run                {mode: pinn|fdtd|both, project}   → job en segundo plano
GET  /api/jobs/{id}                                            → estado
GET  /api/results/{id}/meta                                    → métricas, series, tiempos
GET  /api/results/{id}/frames/{campo}                          → frames binarios float32
POST /api/train              {epochs, lr, …}                   → entrenamiento con eventos por época
WS   /ws/events                                                → consola/progreso/losses en vivo
POST /api/export             {result_id, format, field, frame} → 10 formatos
POST /api/tools/mesh · /api/tools/refine · GET/POST /api/gpu
```

Los solvers se pueden verificar sin servidor:

```bash
cd pinn-maxwell
python fdtd.py          # convergencia O(dx²) vs analítica + conservación de energía
python comparison.py    # pipeline de comparación validado con la solución exacta
python exporters.py     # prueba de los 9 exportadores
python main.py --train  # entrenamiento completo del PINN (Adam + L-BFGS)
```

---

## Referencias

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics.
- Tancik, M. et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* NeurIPS.
- Yee, K. (1966). *Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media.* IEEE Transactions on Antennas and Propagation.
- Taflove, A., & Hagness, S. C. (2005). *Computational Electrodynamics: The Finite-Difference Time-Domain Method.* Artech House.

---

## Autor

**Manuel Sanabria** — Proyecto de investigación universitaria
Aproximación de las Ecuaciones de Maxwell con Redes Neuronales Informadas por la Física, validada contra FDTD.

[![GitHub](https://img.shields.io/badge/GitHub-ManuelSanabria2-black?logo=github)](https://github.com/ManuelSanabria2)
