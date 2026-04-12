 PINN para Ecuaciones de Maxwell Completas — Guía de Desarrollo
Proyecto: Resolución de las Ecuaciones de Maxwell Mediante Physics-Informed Neural Networks
Autor: Manuel José Sanabria Gil
Programa: Ingeniería de Datos e Inteligencia Artificial — Universidad Santo Tomás, Tunja

📋 Descripción del Problema Real
Este proyecto resuelve el sistema completo de ecuaciones de Maxwell en el modo
TM (Transverse Magnetic) 2D dentro de una cavidad electromagnética rectangular
con paredes conductoras perfectas (PEC — Perfect Electric Conductor).
¿Por qué modo TM 2D?
El modo TM es el caso más simple que requiere las cuatro ecuaciones de Maxwell acopladas
y tiene solución analítica exacta conocida (modos de resonancia de cavidad), lo que permite
validar cuantitativamente la PINN. El campo eléctrico apunta en dirección z (fuera del plano)
y el campo magnético yace en el plano xy.
Campos que se resuelven simultáneamente
Ez(x, y, t)  →  componente eléctrica perpendicular al plano
Bx(x, y, t)  →  componente magnética horizontal
By(x, y, t)  →  componente magnética vertical
La red neuronal predice los tres campos al mismo tiempo a partir de la entrada (x, y, t).

🔬 Formulación Física Completa
Ecuaciones de Maxwell en vacío (modo TM 2D, sin cargas ni corrientes)
Las cuatro ecuaciones de Maxwell se reducen al siguiente sistema acoplado de 3 EDPs:
Faraday (componente x):   ∂Bx/∂t  =  −∂Ez/∂y
Faraday (componente y):   ∂By/∂t  =  +∂Ez/∂x
Ampère-Maxwell (comp. z): μ₀ε₀ ∂Ez/∂t  =  ∂By/∂x − ∂Bx/∂y
Gauss magnética (2D):     ∂Bx/∂x + ∂By/∂y  =  0     (restricción de consistencia)
Estas son las 4 leyes de Maxwell expresadas directamente como residuales.
Dominio espacio-temporal
Dominio espacial:   Ω  = [0, L] × [0, L]   con L = 1.0 m
Dominio temporal:   T  = [0, T_max]          con T_max = 2 periodos de oscilación
Condiciones de contorno (paredes PEC)
Ez(0, y, t) = 0    Ez(L, y, t) = 0
Ez(x, 0, t) = 0    Ez(x, L, t) = 0
Condiciones iniciales (modo fundamental m=1, n=1)
Ez(x, y, 0) = E₀ · sin(πx/L) · sin(πy/L)
Bx(x, y, 0) = 0
By(x, y, 0) = 0
Solución analítica exacta (modo TM₁₁)
ω₁₁   = c · π√2 / L          (frecuencia angular de resonancia)
T_res  = 2π / ω₁₁             (periodo de oscilación)

Ez(x,y,t) = E₀ · sin(πx/L) · sin(πy/L) · cos(ω₁₁·t)
Bx(x,y,t) = −(π/ω₁₁L) · E₀ · sin(πx/L) · cos(πy/L) · sin(ω₁₁·t)
By(x,y,t) = +(π/ω₁₁L) · E₀ · cos(πx/L) · sin(πy/L) · sin(ω₁₁·t)

🏗️ Arquitectura de la Red Neuronal
Entrada:   (x, y, t)            →  3 neuronas
Oculta 1:  128 neuronas         →  activación tanh
Oculta 2:  128 neuronas         →  activación tanh
Oculta 3:  128 neuronas         →  activación tanh
Oculta 4:  128 neuronas         →  activación tanh
Salida:    (Ez, Bx, By)         →  3 neuronas, activación lineal
La red es más profunda y ancha que en el caso Laplace porque debe aprender
simultáneamente tres funciones que varían en tres dimensiones (x, y, t).

🗂️ Estructura de Archivos
pinn-maxwell/
│
├── 📄 model.py            # Red neuronal con 3 entradas y 3 salidas
├── 📄 physics.py          # Las 4 ecuaciones de Maxwell como residuales
├── 📄 sampling.py         # Puntos interiores, frontera e instante inicial
├── 📄 analytical.py       # Modos de resonancia TM₁₁ como ground truth
├── 📄 train.py            # Entrenamiento Adam + L-BFGS
├── 📄 visualization.py    # Snapshots de Ez, B, energía electromagnética
├── 📄 metrics.py          # Error L2 por campo y residuales por ecuación
├── 📄 main.py             # Pipeline completo
└── 📁 results/            # Figuras, métricas y modelo guardado

⚙️ Requisitos Técnicos
ComponenteVersiónPython3.10+PyTorch2.0+NumPy1.24+Matplotlib3.7+SciPy1.10+
bashpip install torch numpy matplotlib scipy

📊 Métricas de Éxito
Campo / EcuaciónMétricaMetaEzError relativo L2< 2%BxError relativo L2< 2%ByError relativo L2< 2%Faraday (x)Residual medio< 1e-3Faraday (y)Residual medio< 1e-3Ampère-MaxwellResidual medio< 1e-3Gauss magnéticaResidual medio< 1e-4

