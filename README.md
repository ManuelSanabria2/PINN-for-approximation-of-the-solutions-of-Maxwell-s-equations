#  PINN para Ecuaciones de Maxwell — Cámara de Faraday Digital

> **Red Neuronal Informada por la Física (PINN)** que aproxima la solución analítica exacta de las Ecuaciones de Maxwell en una cavidad rectangular PEC 2D, con demo interactiva 3D en tiempo real.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-green?logo=fastapi)
![Three.js](https://img.shields.io/badge/Three.js-r165-black?logo=threedotjs)
![Error L2](https://img.shields.io/badge/Error%20L2-<2%25-brightgreen)

---

##  ¿Qué resuelve este proyecto?

La PINN aprende a resolver las **Ecuaciones de Maxwell** en modo TM₁₁ para una cavidad rectangular con condiciones de contorno PEC (Perfect Electric Conductor):

$$\frac{\partial E_z}{\partial t} = \frac{1}{\mu_0}\left(\frac{\partial B_x}{\partial y} - \frac{\partial B_y}{\partial x}\right) \quad \text{(Ampere)}$$

$$\frac{\partial B_x}{\partial t} = -\frac{\partial E_z}{\partial y}, \quad \frac{\partial B_y}{\partial t} = \frac{\partial E_z}{\partial x} \quad \text{(Faraday)}$$

**Sin malla. Sin diferencias finitas. Una red neuronal aprende la ley física.**

---

##  Resultados

| Métrica | Antes | Después |
|---|---|---|
| Error L2 Ez | ~64% | **< 2%** |
| Error L2 Bx | ~63% | **< 2%** |
| Error L2 By | ~63% | **< 2%** |
| Conservación Energía | ~320% | **< 5%** |

---

##  Demo Interactiva — Cámara de Faraday Digital

Una visualización 3D inmersiva donde el usuario interroga al modelo en tiempo real.

### Inicio Rápido

```bash
# 1. Clonar el repositorio
git clone https://github.com/ManuelSanabria2/PINN-for-approximation-of-the-solutions-of-Maxwell-s-equations.git
cd PINN-for-approximation-of-the-solutions-of-Maxwell-s-equations

# 2. Instalar dependencias
pip install torch numpy matplotlib scipy fastapi uvicorn

# 3. Levantar el servidor de la demo
cd pinn-maxwell
uvicorn server:app --port 8000

# 4. Abrir en el navegador
# http://localhost:8000/demo
```

### Controles de la Demo

| Control | Acción |
|---|---|
| **Arrastrar** | Rotar la cámara 3D |
| **Scroll** | Zoom |
| `Space` | Play / Pause |
| **Slider ε_r** | Cambiar permitividad → la onda se contrae |
| **Slider μ_r** | Cambiar permeabilidad → la onda se expande |
| **XY / XT / YT** | Cambiar modo de corte (espacial / espacio-temporal) |
| **Malla Fantasma** | Ver puntos de colocación del entrenamiento |

> **Modo offline:** Si el servidor no está activo, la demo usa automáticamente la solución analítica exacta calculada en el navegador.

---

##  Arquitectura de la Red

```
Entrada: (x, y, t) ∈ [0,1]×[0,1]×[0,T]
     ↓
Fourier Feature Encoding (σ = 1.5, N = 32 frecuencias)
     ↓
4 capas ocultas × 128 neuronas (Tanh)
     ↓
Salida: (Ez, Bx, By) — los 3 campos electromagnéticos
```

### Técnicas para Convergencia

| Técnica | Descripción |
|---|---|
| **Fourier Feature Encoding** | Supera el sesgo espectral — permite aprender funciones de alta frecuencia como sin(πx)·sin(πy)·cos(ωt) |
| **Warm-up Curricular** | λ_BC/IC = 100 durante las primeras 8k épocas para anclar las condiciones físicas |
| **Cosine Annealing** | Scheduler de LR con reinicios calientes (T₀=4000, T_mult=2) |
| **Gradient Clipping** | Previene explosión de gradientes durante Adam |
| **L-BFGS** | Refinamiento fino post-Adam (5000 iteraciones) |
| **λ_Gauss = 2×** | Peso doble para la restricción ∇·B = 0 |
| **Resampling BC/IC** | Puntos de frontera re-muestreados cada 3000 épocas |

---

##  Estructura del Proyecto

```
 PINN-Maxwell/
├──  pinn-maxwell/
│   ├── main.py          # Punto de entrada — pipeline completo
│   ├── model.py         # Arquitectura FCN con Fourier Features
│   ├── train.py         # Entrenamiento Adam + L-BFGS
│   ├── physics.py       # Residuales de Maxwell + función de pérdida
│   ├── sampling.py      # Muestreo de puntos de colocación
│   ├── analytical.py    # Solución analítica exacta TM₁₁
│   ├── metrics.py       # Evaluación L2, energía, BC, IC (GPU-compatible)
│   ├── visualization.py # Gráficas científicas (GPU-compatible)
│   ├── server.py        # FastAPI backend para la demo 3D
│   ├── higher_mode.py   # Modo experimental TM₂₁
│   ├── sensitivity.py   # Análisis de sensibilidad de hiperparámetros
│   └──  demo/
│       ├── index.html   #  Cámara de Faraday Digital (Three.js)
│       └── README.md    # Instrucciones de la demo
├──  results/
│   ├── maxwell_pinn.pth # Modelo entrenado
│   └── training_log.txt # Log de entrenamiento
└── README.md            # Este archivo
```

---

##  Entrenamiento desde Cero

```bash
cd pinn-maxwell

# Entrenamiento completo (Adam 30k + L-BFGS 5k épocas)
python main.py --train

# Entrenamiento rápido para prueba (50 + 10 épocas)
python main.py --train --quick

# Pipeline completo: entrenar + evaluar + visualizar
python main.py --full-pipeline

# Evaluar modelo guardado
python main.py --evaluate

# Análisis de sensibilidad
python main.py --sensitivity

# Modo TM₂₁ experimental
python main.py --higher-mode
```

### Uso de GPU

El entrenamiento detecta CUDA automáticamente. Para forzar CPU:

```bash
# En Windows, verificar GPU disponible
python -c "import torch; print(torch.cuda.is_available())"
```

---

##  ¿Qué se puede demostrar con la demo?

### 1. Solución sin Malla
La PINN evalúa (x, y, t) → (Ez, Bx, By) en cualquier punto continuo del dominio  
→ **sin discretización, sin interpolación**

### 2. Generalización del Medio
Cambiar ε_r y μ_r **sin re-entrenar** → la onda cambia velocidad en tiempo real  
→ `t_eff = t / √(ε_r · μ_r)` adapta la física post-proceso

### 3. Naturaleza Vectorial de Maxwell
Las flechas B rotan perpendiculares al gradiente de Ez  
→ Visualización directa de las leyes de Faraday y Ampere

### 4. El Cubo Espacio-Temporal
El eje Z del cubo = tiempo t ∈ [0, T_MAX]  
→ La losa que sube es el "ahora" moviéndose a través del espacio-tiempo EM

---

##  Referencias

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics.
- Tancik, M. et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* NeurIPS.
- Jin, X. et al. (2021). *NSFnets (Navier-Stokes Flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations.* JCP.

---

##  Autor

**Manuel Sanabria** — Proyecto de investigación universitaria  
Aproximación de Ecuaciones de Maxwell con Redes Neuronales Informadas por la Física

[![GitHub](https://img.shields.io/badge/GitHub-ManuelSanabria2-black?logo=github)](https://github.com/ManuelSanabria2)
