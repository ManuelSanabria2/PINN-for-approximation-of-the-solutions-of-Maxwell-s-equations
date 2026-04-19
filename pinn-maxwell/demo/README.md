# Cámara de Faraday Digital — Demo de Presentación

## Inicio Rápido

```bash
# 1. Desde el directorio pinn-maxwell/
pip install fastapi uvicorn

# 2. Levantar el servidor (el modelo debe estar entrenado)
uvicorn server:app --port 8000

# 3. Abrir en el navegador
# http://localhost:8000/demo
```

> Si el modelo no está entrenado: `python main.py --train` primero.

## Controles

| Control | Acción |
|---|---|
| **Arrastrar** | Rotar la cámara 3D |
| **Scroll** | Zoom in/out |
| **Space** | Play / Pause |
| Slider **Tiempo t** | Scrubbing temporal manual |
| Slider **Velocidad** | x0.1 a x4 |
| Slider **ε_r** | Cambiar permitividad (contrae λ) |
| Slider **μ_r** | Cambiar permeabilidad (contrae λ) |
| Botones **XY / XT / YT** | Cambiar modo de corte |
| **Malla Fantasma** | Toggle puntos de colocación |
| **Vectores B** | Toggle flechas del campo magnético |

## Modos de Visualización

- **XY (t)** — Vista instantánea del campo Ez en el plano espacial 2D. La losa se desplaza
  verticalmente a lo largo del eje Z (tiempo) del cubo espacio-temporal.
- **XT (y fijo)** — Diagrama espacio-temporal Ez(x, t) a y = posición fija.
- **YT (x fijo)** — Diagrama espacio-temporal Ez(y, t) a x = posición fija.

## Modo Offline

Si el servidor no está corriendo, la demo usa automáticamente la **solución analítica exacta**
calculada en el navegador. El badge superior indica el modo activo.

## Dependencias

```
fastapi
uvicorn
torch (ya instalado para el PINN)
```
