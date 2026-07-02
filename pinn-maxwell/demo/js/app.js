/* app.js — arranque de Fluxia: conecta API, estado, viewport y paneles */
import { api, connectEvents, waitJob, downloadBlob } from "./api.js";
import { state, on, setProject, setResult, select } from "./state.js";
import { initViewport, resizeViewport, refreshField, viewportCanvas } from "./viewport.js";
import { initMenus, initDialogs, renderTree, renderProps, consoleLog,
         openGpuDialog, openMaterialsInspector, syncVisMenu } from "./ui.js";
import { initDock, initTraining, onTrainEvent, trainFinished, initCompare,
         drawCompare, renderMetrics, initPlayer, togglePlay, initGraphs,
         drawGraph, invalidateResiduals, initExport, openExportDialog,
         recordViewportVideo } from "./panels.js";

/* ═══════════ Estado del servidor (chips) ═══════════ */
function setStatus(kind, text) {
  const chip = document.getElementById("chip-status");
  chip.className = `chip chip-status ${kind}`;
  chip.textContent = text;
}

/* ═══════════ Ejecutar simulaciones ═══════════ */
async function runSimulation(mode) {
  if (state.activeJob) {
    consoleLog("Ya hay una simulación en curso.", "warn");
    return;
  }
  try {
    setStatus("busy", "Simulando…");
    const labels = { pinn: "PINN", fdtd: "FDTD", both: "PINN + FDTD (comparación)" };
    consoleLog(`Lanzando simulación: ${labels[mode]}`);
    const r = await api.run(mode, state.project);
    state.activeJob = r.job_id;
    const st = await waitJob(r.job_id);
    state.activeJob = null;
    await loadResult(st.result_id);
    setStatus("online", "En línea");
    if (mode === "both") {
      // cambiar a la vista de comparación automáticamente
      document.querySelector('.view-tab[data-view="compare"]').click();
    }
  } catch (e) {
    state.activeJob = null;
    setStatus("online", "En línea");
    consoleLog(`Simulación falló: ${e.message}`, "error");
  }
}

async function loadResult(rid) {
  consoleLog("Descargando resultados…");
  const meta = await api.resultMeta(rid);
  const frames = {};
  for (const f of meta.fields ?? []) {
    frames[f] = await api.resultFrames(rid, f);
  }
  await setResult(rid, meta, frames);
  renderTree();
  renderMetrics();
  drawGraph();
  refreshField();
  const m = meta.metrics ?? {};
  if (m.l2_Ez !== undefined) {
    consoleLog(`✔ Comparación completa — L2 = ${m.l2_Ez.toFixed(4)} %, ` +
      `RMS = ${m.rms_Ez.toFixed(4)} %, máx = ${m.max_Ez.toFixed(4)} %`);
  } else {
    consoleLog(`✔ Resultados cargados (${meta.times?.length ?? 0} frames)`);
  }
}

async function stopAll() {
  if (state.activeJob) {
    await api.cancelJob(state.activeJob).catch(() => {});
    consoleLog("Cancelación solicitada.", "warn");
  }
  if (state.trainJob) {
    await api.trainStop().catch(() => {});
  }
}

/* ═══════════ Archivo: nuevo / abrir / guardar ═══════════ */
async function fileNew() {
  const p = await api.defaultProject();
  p.name = "Nuevo proyecto";
  setProject(p);
  select(null);
  renderTree();
  renderProps();
  consoleLog("Nuevo proyecto creado (cavidad TM11 por defecto).");
}

function fileSave() {
  const blob = new Blob([JSON.stringify(state.project, null, 2)],
                        { type: "application/json" });
  const name = (state.project?.name ?? "proyecto").replace(/\s+/g, "_");
  downloadBlob(blob, `${name}.fluxia.json`);
  state.dirty = false;
  consoleLog(`Proyecto guardado: ${name}.fluxia.json`);
}

function fileOpen() {
  document.getElementById("file-open-input").click();
}

function initFileOpen() {
  document.getElementById("file-open-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    try {
      const p = JSON.parse(await file.text());
      if (!p.domain || !p.materials) throw new Error("no parece un proyecto Fluxia");
      const v = await api.validateProject(p);
      if (!v.ok) throw new Error(v.error);
      setProject(p);
      select(null);
      renderTree();
      renderProps();
      consoleLog(`Proyecto abierto: ${file.name} — ${v.config.n_steps} pasos FDTD (CFL ok)`);
    } catch (err) {
      consoleLog(`No se pudo abrir el proyecto: ${err.message}`, "error");
    }
  });
}

/* ═══════════ Exportar imagen del viewport ═══════════ */
function exportViewportPNG() {
  const url = viewportCanvas().toDataURL("image/png");
  const a = document.createElement("a");
  a.href = url;
  a.download = `fluxia_viewport_${Date.now()}.png`;
  a.click();
  consoleLog("Imagen del viewport exportada (PNG).");
}

/* ═══════════ Ayuda ═══════════ */
const HELP = {
  manual: {
    title: "Manual de uso",
    html: `
      <h4>Flujo de trabajo</h4>
      <p>1. Configura el <b>dominio</b> (clic en «Caja (dominio)» del árbol): Nx, Ny, Courant, T máximo.</p>
      <p>2. Activa geometrías y materiales opcionales (esfera de silicio, antena PEC) desde el árbol.</p>
      <p>3. Ejecuta con el menú <b>Simulación</b>: PINN (rápido, red pre-entrenada), FDTD
         (referencia numérica) o <b>ambos</b> para la comparación científica.</p>
      <p>4. Explora con el <b>reproductor</b> (⏮ ◀ ▶ ⏸ ⏭, velocidades 0.25×–10×) y la
         pestaña <b>Comparación PINN vs FDTD</b>: cuatro vistas sincronizadas con el slider temporal.</p>
      <p>5. Exporta desde <b>Archivo → Exportar…</b> (PDF, CSV, NumPy, HDF5, VTK, MATLAB, GIF, PNG, STL, WebM).</p>
      <h4>Vista 3D</h4>
      <ul>
        <li>Rotar: arrastrar con botón izquierdo · Zoom: rueda · Pan: botón derecho.</li>
        <li>Seleccionar objetos: clic (esfera, antena, fuente, monitores, dominio).</li>
        <li>Medir distancias: botón 📐 y dos clics sobre el dominio.</li>
        <li>El ojo (◉) del árbol muestra/oculta objetos; la casilla «Incluida en simulación» los quita del solver.</li>
      </ul>
      <h4>Entrenamiento PINN</h4>
      <p>Pestaña «Entrenamiento PINN» del panel inferior → «Entrenar…». Las pérdidas
         (PDE, BC, IC), el learning rate y el tiempo se grafican en vivo, época a época.</p>
      <h4>Atajos</h4>
      <p><code>F5</code> PINN · <code>F6</code> FDTD · <code>F7</code> ambos ·
         <code>Espacio</code> reproducir/pausa · <code>Ctrl+S</code> guardar ·
         <code>Ctrl+O</code> abrir · <code>Ctrl+N</code> nuevo.</p>`,
  },
  docs: {
    title: "Documentación técnica",
    html: `
      <h4>Arquitectura</h4>
      <p><b>Backend</b> (FastAPI + PyTorch + NumPy):</p>
      <ul>
        <li><code>fdtd.py</code> — solver Yee 2D TM, orden 2, condición CFL, materiales
            heterogéneos (ε_r, μ_r, σ), PEC, PML graduada, fuentes pulso/CW, sondas y energía.</li>
        <li><code>model.py / physics.py</code> — PINN FCN [3,128,128,128,128,3] con Fourier
            features (n=32, σ=1.5); pérdida = residuales de Maxwell + BC PEC + IC TM₁₁.</li>
        <li><code>comparison.py</code> — evalúa el PINN en la malla y tiempos exactos del FDTD
            y calcula L2, RMS, máximo, series de error, energía, tiempos y memoria.</li>
        <li><code>exporters.py</code> — CSV, VTK, MATLAB, NumPy, HDF5, PNG, GIF, PDF, STL.</li>
      </ul>
      <p><b>Frontend</b> (Three.js + Canvas 2D, sin frameworks): árbol de proyecto,
         viewport 3D, comparación 2×2 sincronizada, consola en vivo vía WebSocket,
         panel de entrenamiento y gráficas.</p>
      <h4>Unidades</h4>
      <p>Naturales: c = ε₀ = μ₀ = 1, L = 1. La frecuencia del modo TM₁₁ es
         ω = π√2 (f ≈ 0.707, T ≈ 1.414).</p>
      <h4>API principal</h4>
      <pre>POST /api/run            {mode: pinn|fdtd|both, project}
POST /api/train          {epochs, lr, …}
GET  /api/results/{id}/meta
GET  /api/results/{id}/frames/{campo}   ← binario float32
POST /api/export         {result_id, format, field, frame}
POST /api/tools/mesh · /api/tools/refine · /api/gpu</pre>`,
  },
  math: {
    title: "Información matemática",
    html: `
      <h4>Problema físico</h4>
      <p>Modo TM en una cavidad PEC 2D [0,L]²; incógnitas Ez(x,y,t), Bx, By:</p>
      <pre>∂Bx/∂t = −∂Ez/∂y            (Faraday, x)
∂By/∂t = +∂Ez/∂x            (Faraday, y)
ε ∂Ez/∂t = ∂By/∂x − ∂Bx/∂y − σEz   (Ampère–Maxwell)
∂Bx/∂x + ∂By/∂y = 0         (Gauss magnético)</pre>
      <p>Condición de frontera PEC: Ez = 0 en los cuatro bordes.
         Condición inicial (modo TM₁₁): Ez = E₀ sin(πx/L) sin(πy/L), B = 0.</p>
      <h4>Solución analítica</h4>
      <pre>Ez = E₀ sin(kx x) sin(ky y) cos(ωt),  ω = c·√(kx²+ky²)
Bx = −E₀ (ky/ω) sin(kx x) cos(ky y) sin(ωt)
By = +E₀ (kx/ω) cos(kx x) sin(ky y) sin(ωt)</pre>
      <h4>PINN</h4>
      <p>Red u_θ(x,y,t) → (Ez,Bx,By) entrenada minimizando
         L = λ_PDE·MSE(residuales) + λ_BC·MSE(Ez|∂Ω) + λ_IC·MSE(u(·,0) − u₀),
         con derivadas exactas por diferenciación automática y Fourier feature
         encoding para combatir el sesgo espectral.</p>
      <h4>FDTD (Yee)</h4>
      <p>Malla escalonada: Ez en nodos enteros, Bx/By en medio-nodos; integración
         <i>leapfrog</i> de orden 2. Estabilidad (CFL 2D): c·Δt ≤ 1/√(1/Δx²+1/Δy²).
         El error de discretización decrece como O(Δx²).</p>
      <h4>Métricas de comparación</h4>
      <pre>L2 rel  = ‖u_PINN − u_FDTD‖₂ / ‖u_FDTD‖₂ × 100
RMS rel = RMS(Δu) / RMS(u_FDTD) × 100
máx rel = max|Δu| / max|u_FDTD| × 100</pre>
      <p>La energía U(t) = ½∬(εE² + B²/μ)dA debe conservarse: su deriva es un
         diagnóstico independiente para ambos métodos (pestaña «Energía»).</p>`,
  },
};

async function showHelp(kind) {
  const dlg = document.getElementById("dlg-help");
  if (kind === "version") {
    let vtxt = "";
    try {
      const v = await api.version();
      vtxt = `<pre>${v.app}  v${v.version}
Python ${v.python} · PyTorch ${v.torch} · NumPy ${v.numpy}
FDTD : ${v.solver_fdtd}
PINN : ${v.solver_pinn}</pre>`;
    } catch { vtxt = "<p>Servidor no disponible.</p>"; }
    document.getElementById("help-title").textContent = "Versión";
    document.getElementById("help-body").innerHTML = vtxt;
  } else {
    document.getElementById("help-title").textContent = HELP[kind].title;
    document.getElementById("help-body").innerHTML = HELP[kind].html;
  }
  dlg.showModal();
}

/* ═══════════ Vistas centrales (3D / comparación) ═══════════ */
function initViewTabs() {
  document.querySelectorAll(".view-tab").forEach((t) =>
    t.addEventListener("click", () => {
      document.querySelectorAll(".view-tab").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      const isVp = t.dataset.view === "viewport";
      document.getElementById("viewport-container").hidden = !isVp;
      document.getElementById("compare-container").hidden = isVp;
      if (isVp) { resizeViewport(); refreshField(); }
      else { drawCompare(); renderMetrics(); }
    }));
}

/* ═══════════ Eventos del servidor → consola / paneles ═══════════ */
function handleServerEvent(e) {
  if (e.kind === "log") {
    consoleLog(e.msg, e.level ?? "info");
  } else if (e.kind === "train") {
    consoleLog(`Epoch ${e.epoch}/${e.epochs} · PDE ${e.loss_pde.toExponential(2)} · ` +
               `BC ${e.loss_bc.toExponential(2)} · IC ${e.loss_ic.toExponential(2)}`, "train");
    onTrainEvent(e);
  } else if (e.kind === "job") {
    if (e.status === "done" && e.jtype === "train") {
      trainFinished();
      invalidateResiduals();
      consoleLog("✔ Entrenamiento terminado — el modelo activo fue actualizado.");
    }
    if (e.status === "error") consoleLog(`Job ${e.jtype} falló: ${e.error}`, "error");
  }
}

/* ═══════════ Arranque ═══════════ */
async function boot() {
  initDock();
  initViewTabs();
  initDialogs();
  initTraining();
  initCompare();
  initPlayer();
  initGraphs();
  initExport();
  initFileOpen();
  initViewport();

  initMenus({
    "file-new": fileNew,
    "file-open": fileOpen,
    "file-save": fileSave,
    "export-dialog": () => openExportDialog(),
    "export-png": exportViewportPNG,
    "export-video": () => recordViewportVideo(viewportCanvas(), 6),
    "export-gif": () => openExportDialog("gif"),
    "export-vtk": () => openExportDialog("vtk"),
    "export-csv": () => openExportDialog("csv"),
    "export-mat": () => openExportDialog("mat"),
    "run-pinn": () => runSimulation("pinn"),
    "run-fdtd": () => runSimulation("fdtd"),
    "run-both": () => runSimulation("both"),
    "run-train": () => document.getElementById("dlg-train").showModal(),
    "run-stop": stopAll,
    "tool-mesh": () => document.getElementById("dlg-mesh").showModal(),
    "tool-refine": () => document.getElementById("dlg-refine").showModal(),
    "tool-materials": openMaterialsInspector,
    "tool-gpu": openGpuDialog,
    "help-manual": () => showHelp("manual"),
    "help-docs": () => showHelp("docs"),
    "help-math": () => showHelp("math"),
    "help-version": () => showHelp("version"),
    "play-pause": togglePlay,
  });

  // Reactividad de alto nivel
  on("selection", renderProps);
  on("project", () => { renderTree(); });
  on("frame", refreshField);

  // Conexión con el servidor
  connectEvents(handleServerEvent, (s) => {
    if (s === "online") setStatus("online", "En línea");
    else setStatus("offline", "Sin conexión");
  });

  consoleLog("Fluxia iniciado. Cargando proyecto por defecto…");
  try {
    const [health, project] = await Promise.all([api.health(), api.defaultProject()]);
    document.getElementById("chip-device").textContent = health.device.toUpperCase();
    document.getElementById("chip-model").textContent =
      health.model_loaded ? "PINN ✓" : "PINN sin entrenar";
    setStatus("online", "En línea");
    setProject(project);
    renderTree();
    syncVisMenu();
    consoleLog(`Servidor ${health.app} v${health.version} — dispositivo ${health.device.toUpperCase()}`);
    consoleLog("Listo. Ejecuta Simulación → Ejecutar ambos (F7) para comparar PINN vs FDTD.");
  } catch (e) {
    setStatus("offline", "Sin conexión");
    consoleLog(`No se pudo conectar con el servidor: ${e.message}`, "error");
  }

  refreshField();
}

boot();
