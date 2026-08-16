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

## 1. Problema

Las PINN se proponen como alternativa a los solvers numéricos clásicos en electromagnetismo, pero casi siempre se reportan contra sí mismas: una curva de pérdida que baja no dice **cuánto se parece el campo aprendido al que produciría un solver de referencia**. La mayoría de las herramientas permiten ejecutar distintos solvers; **muy pocas están diseñadas para evaluar directamente cómo una PINN reproduce una solución FDTD**, sobre la misma malla y los mismos pasos de tiempo.

El caso de estudio es el modo TM en una cavidad PEC `[0,L]²` (unidades naturales, c = ε₀ = μ₀ = 1):

$$\frac{\partial B_x}{\partial t} = -\frac{\partial E_z}{\partial y}, \qquad \frac{\partial B_y}{\partial t} = \frac{\partial E_z}{\partial x} \quad \text{(Faraday)}$$

$$\varepsilon\frac{\partial E_z}{\partial t} = \frac{\partial B_y}{\partial x} - \frac{\partial B_x}{\partial y} - \sigma E_z \quad \text{(Ampère–Maxwell)}$$

Condición inicial (modo TM₁₁): `Ez = E₀·sin(πx/L)·sin(πy/L)`, `B = 0`. Este problema tiene **solución analítica exacta**, lo que permite validar de forma independiente cada uno de los dos solvers antes de compararlos entre sí — sin esa tercera referencia, una comparación PINN vs FDTD no distingue quién de los dos se equivoca.

---

## 2. Solución

Fluxia resuelve el mismo problema por dos caminos independientes y los enfrenta sobre un terreno común:

- **PINN** (`model.py`, `physics.py`, `train.py`) — red FCN `(x,y,t) → (Ez,Bx,By)` con Fourier Feature Encoding, entrenada minimizando los residuales de Maxwell (vía autograd) + condiciones de frontera PEC + condición inicial TM₁₁. Optimización Adam seguida de L-BFGS.
- **FDTD** (`fdtd.py`) — solver Yee 2D TM de orden 2, condición CFL, materiales heterogéneos (ε_r, μ_r, σ), PEC, PML graduada, fuentes de pulso gaussiano y conservación de energía verificada.
- **Comparación** (`comparison.py`) — el PINN se evalúa en la malla y los tiempos **exactos** del FDTD → error L2 / RMS / máximo, mapas de error, series temporales, tiempos de cómputo, memoria e iteraciones.
- **Validación cruzada** (`analytical.py`, `validation.py`) — ambos solvers se contrastan contra la solución analítica: convergencia O(Δx²) del FDTD y deriva de energía nula.

Todo esto vive dentro de una interfaz CAD completa: árbol de proyecto, viewport 3D, propiedades contextuales, entrenamiento en vivo y exportación técnica — no un notebook, sino una herramienta usable.

---

## 3. Arquitectura

```
┌─────────────────────── Frontend (demo/, sin frameworks) ───────────────────────┐
│  app.js · api.js · state.js · viewport.js (Three.js) · ui.js · panels.js       │
│  charts.js (canvas puro: líneas, heatmaps, FFT)                                │
└───────────────┬──────────────── REST + WebSocket ──────────────────────────────┘
                │
┌───────────────▼────────────── Backend (server.py, FastAPI) ────────────────────┐
│  /api/run · /api/train  → jobs en segundo plano                                │
│  /ws/events             → consola, progreso y losses en vivo                    │
│  project_to_fdtd_config → proyecto JSON del frontend  ⇒  FDTDConfig             │
│  /api/export · /api/tools/* · /api/slice · /api/residuals                      │
└───────┬──────────────────────────────────┬─────────────────────────────────────┘
        │                                  │
┌───────▼──────── Solvers ────────┐  ┌─────▼──── Persistencia (store.py) ────────┐
│ PINN: model · physics · train   │  │ SQLite (jobs/resultados/métricas)          │
│ FDTD: fdtd.py (Yee, PML, mat.)  │  │ .npz (arrays de campo) + caché LRU         │
│ Referencia: analytical.py       │  │ sobrevive a reinicios del servidor         │
│ Análisis: comparison · metrics  │  └────────────────────────────────────────────┘
│           sensitivity · validation                                              │
└─────────────────────────────────┘
```

**Flujo de una ejecución:** el usuario edita el proyecto en el árbol CAD → `POST /api/run {mode: both}` → el backend traduce el JSON a `FDTDConfig`, lanza el job en segundo plano y emite eventos por WebSocket → los campos se guardan en `.npz` y las métricas en SQLite → el frontend lee frames binarios float32 bajo demanda y los sincroniza en la vista 2×2 con un único slider temporal.

### Estructura del proyecto

```
pinn-maxwell/
├── fdtd.py          # Solver FDTD Yee 2D TM (materiales, PML, fuentes, energía)
├── comparison.py    # Comparación PINN vs FDTD en la misma malla/tiempos
├── exporters.py     # CSV, VTK, MATLAB, NumPy, HDF5, PNG, GIF, PDF, STL
├── server.py        # Backend FastAPI: jobs, WebSocket, entrenamiento en vivo, export
├── store.py         # Persistencia: jobs/resultados en SQLite + arrays en .npz
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

---

## 4. Tecnologías

| Capa | Stack | Por qué |
|---|---|---|
| **PINN** | PyTorch 2.x (autograd, Adam, L-BFGS) | Los residuales de Maxwell se obtienen por diferenciación automática, no por diferencias finitas |
| **Solver numérico** | NumPy (esquema de Yee vectorizado) | Referencia clásica de segundo orden, sin dependencias pesadas |
| **Backend** | FastAPI + Uvicorn, WebSocket, jobs en segundo plano | Streaming de épocas y pasos FDTD en vivo sin polling |
| **Persistencia** | SQLite (stdlib) + `.npz` + caché LRU | Resultados que sobreviven a reinicios; frames binarios sin serializar JSON |
| **Frontend** | HTML/CSS/JS puro + Three.js r165 | Sin build step: se sirve directo desde FastAPI; canvas puro para todas las gráficas |
| **Exportación** | matplotlib, h5py, scipy.io, VTK, STL | 10 formatos técnicos para ParaView, MATLAB y reportes |
| **Despliegue** | Docker → Hugging Face Spaces (puerto 7860) | CPU-only por defecto; GPU opcional vía `/api/gpu` |

---

## 5. Resultados

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

El error del FDTD cae ≈4× al duplicar la resolución, confirmando el orden O(Δx²) esperado; la energía electromagnética se conserva sin deriva apreciable en toda la ventana simulada. Con esa referencia validada, el 0.118 % de error L2 del PINN sobre Ez es una medida real de fidelidad física, no un artefacto del entrenamiento.

Los solvers se pueden verificar sin levantar el servidor:

```bash
cd pinn-maxwell
python fdtd.py          # convergencia O(dx²) vs analítica + conservación de energía
python comparison.py    # pipeline de comparación validado con la solución exacta
python exporters.py     # prueba de los exportadores
python main.py --train  # entrenamiento completo del PINN (Adam + L-BFGS)
```

---

## 6. Demo

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

**Prueba en 30 segundos:** menú **Simulación → Ejecutar ambos (F7)** → la pestaña *Comparación PINN vs FDTD* muestra 4 paneles sincronizados (PINN | FDTD | mapa de error | métricas) que se mueven juntos con el slider temporal.

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

### API principal

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

---

## 7. Instalación

```bash
git clone https://github.com/ManuelSanabria2/PINN-for-approximation-of-the-solutions-of-Maxwell-s-equations.git
cd PINN-for-approximation-of-the-solutions-of-Maxwell-s-equations
pip install -r requirements.txt

cd pinn-maxwell
uvicorn server:app --port 8000
# → http://localhost:8000/demo
```

En Windows basta con doble clic en `start_demo.bat`.

Con Docker (misma imagen que el Space de Hugging Face, puerto 7860):

```bash
docker build -t fluxia .
docker run -p 7860:7860 fluxia   # → http://localhost:7860
```

### Persistencia

Los trabajos y resultados de simulación (metadatos en SQLite + campos en `.npz`) se guardan en disco y sobreviven a un reinicio del servidor.

- Ruta de datos: `FLUXIA_DATA_DIR` si está definida → si no, `/data` cuando existe y es escribible (addon de pago "Persistent Storage" de Hugging Face Spaces) → si no, `pinn-maxwell/data/` dentro del contenedor.
- **Sin el addon de HF Spaces activado**, `pinn-maxwell/data/` vive en el disco efímero del contenedor: sobrevive a caídas del proceso pero se pierde en un rebuild/redeploy completo del Space. Para persistencia real en HF Spaces, activa "Persistent Storage" (monta `/data`) — el código lo detecta automáticamente sin cambios.
- Retención configurable con `FLUXIA_MAX_RESULTS` (por defecto 200): los resultados más viejos se purgan (fila + `.npz`) al superar el umbral.

### Seguridad / concurrencia

- **CORS**: el frontend (`demo/`) es same-origin y no necesita CORS abierto. Por defecto el backend no permite ningún origen cruzado (evita que otro sitio web use el navegador de un visitante para disparar acciones contra el servidor). Para consumir la API desde un dashboard/frontend separado, definí `FLUXIA_CORS_ORIGINS` con los orígenes permitidos, separados por coma (ej. `FLUXIA_CORS_ORIGINS=https://mi-dashboard.com`).
- **Entrenamiento**: solo puede haber un entrenamiento activo a la vez (`/api/train` responde 409 si ya hay uno en curso), y `/api/gpu` (cambia hilos de PyTorch, configuración global del proceso) se rechaza mientras haya cualquier simulación o entrenamiento en curso, para que no pise trabajos de otras pestañas/usuarios.

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
