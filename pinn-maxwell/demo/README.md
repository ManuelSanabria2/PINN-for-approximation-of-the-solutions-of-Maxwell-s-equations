# Fluxia — Frontend

Interfaz web profesional (estilo CAD) del entorno de simulación **PINN vs FDTD**.
Sustituye a la antigua "Cámara de Faraday Digital".

## Inicio rápido

```bash
# Desde pinn-maxwell/
pip install -r ../requirements.txt
uvicorn server:app --port 8000
# → http://localhost:8000/demo
```

## Módulos

| Archivo | Rol |
|---|---|
| `index.html` | Layout: menús, explorador, viewport 3D, propiedades, dock inferior, diálogos |
| `style.css` | Tema oscuro de ingeniería |
| `js/app.js` | Arranque, acciones de menú, orquestación de simulaciones |
| `js/api.js` | Cliente REST + WebSocket (`/api/*`, `/ws/events`) |
| `js/state.js` | Estado central: proyecto, selección, resultado, reproductor |
| `js/viewport.js` | Escena Three.js: dominio, materiales, fuente, PML, campo, isolíneas, cortes, medición |
| `js/ui.js` | Árbol del proyecto, panel de propiedades, consola, menús, herramientas |
| `js/panels.js` | Entrenamiento en vivo, comparación 2×2, gráficas, reproductor, exportación |
| `js/charts.js` | Gráficas de línea, heatmaps y FFT en canvas puro (sin dependencias) |

## Atajos

`F5` PINN · `F6` FDTD · `F7` ambos · `Espacio` play/pausa · `Ctrl+S` guardar · `Ctrl+O` abrir · `Ctrl+N` nuevo proyecto

Ver el README principal del repositorio para la documentación completa.
