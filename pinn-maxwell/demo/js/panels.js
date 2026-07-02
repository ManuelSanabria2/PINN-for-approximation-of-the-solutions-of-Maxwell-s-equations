/* panels.js — dock (entrenamiento, gráficas), comparación 2×2, reproductor, exportación */
import { state, on, setFrame, frameTimes } from "./state.js";
import { api } from "./api.js";
import { LineChart, drawHeatmap, fftMag } from "./charts.js";
import { consoleLog } from "./ui.js";

/* ═══════════════ DOCK: pestañas ═══════════════ */
export function initDock() {
  const tabs = document.querySelectorAll(".dock-tab");
  tabs.forEach((t) =>
    t.addEventListener("click", () => {
      tabs.forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      document.querySelectorAll(".dock-view").forEach((v) => (v.hidden = true));
      document.getElementById(`dock-${t.dataset.dock}`).hidden = false;
      if (t.dataset.dock === "training") drawTrainCharts();
      if (t.dataset.dock === "graphs") drawGraph();
    }));
  document.getElementById("dock-collapse").addEventListener("click", () =>
    document.getElementById("dock").classList.toggle("collapsed"));
}

/* ═══════════════ PANEL DE ENTRENAMIENTO (TensorBoard-like) ═══════════════ */
const trainHist = { epoch: [], total: [], pde: [], bc: [], ic: [], lr: [] };
let lossChart, lrChart;

export function initTraining() {
  lossChart = new LineChart(document.getElementById("tr-chart-loss"),
    { title: "Pérdidas (escala log)", logY: true, xlabel: "época" });
  lrChart = new LineChart(document.getElementById("tr-chart-lr"),
    { title: "Learning rate", logY: true, xlabel: "época" });

  document.getElementById("tr-start").addEventListener("click", () =>
    document.getElementById("dlg-train").showModal());
  document.getElementById("tr-stop").addEventListener("click", async () => {
    await api.trainStop();
    consoleLog("Solicitada detención del entrenamiento…", "warn");
  });
  document.getElementById("tr-go").addEventListener("click", startTraining);

  // Cargar historial pre-entrenado como referencia inicial
  api.trainingHistory().then((r) => {
    if (!r.history?.length || trainHist.epoch.length) return;
    for (const e of r.history) {
      trainHist.epoch.push(e.epoch);
      trainHist.total.push(e.loss_total);
      trainHist.pde.push(e.loss_pde);
      trainHist.bc.push(e.loss_bc);
      trainHist.ic.push(e.loss_ic);
    }
    const last = r.history.at(-1);
    setTrainStats({ epoch: last.epoch, loss_pde: last.loss_pde,
                    loss_bc: last.loss_bc, loss_ic: last.loss_ic });
    drawTrainCharts();
  }).catch(() => {});
}

async function startTraining() {
  document.getElementById("dlg-train").close();
  const opts = {
    epochs: +document.getElementById("tr-opt-epochs").value,
    lr: +document.getElementById("tr-opt-lr").value,
    n_collocation: +document.getElementById("tr-opt-ncol").value,
    n_initial: +document.getElementById("tr-opt-nic").value,
    reset: document.getElementById("tr-opt-reset").checked,
    save: document.getElementById("tr-opt-save").checked,
  };
  // reiniciar historial en vivo
  trainHist.epoch.length = 0; trainHist.total.length = 0; trainHist.pde.length = 0;
  trainHist.bc.length = 0; trainHist.ic.length = 0; trainHist.lr.length = 0;
  try {
    const r = await api.train(opts);
    state.trainJob = r.job_id;
    document.getElementById("tr-start").hidden = true;
    document.getElementById("tr-stop").hidden = false;
    // activar pestaña de entrenamiento
    document.querySelector('.dock-tab[data-dock="training"]').click();
    document.getElementById("dock").classList.remove("collapsed");
  } catch (e) {
    consoleLog(`No se pudo iniciar el entrenamiento: ${e.message}`, "error");
  }
}

export function onTrainEvent(e) {
  trainHist.epoch.push(e.epoch);
  trainHist.total.push(e.loss_total);
  trainHist.pde.push(e.loss_pde);
  trainHist.bc.push(e.loss_bc);
  trainHist.ic.push(e.loss_ic);
  trainHist.lr.push(e.lr);
  setTrainStats(e);
  if (!document.getElementById("dock-training").hidden) drawTrainCharts();
  if (e.epoch >= e.epochs) trainFinished();
}

export function trainFinished() {
  document.getElementById("tr-start").hidden = false;
  document.getElementById("tr-stop").hidden = true;
  state.trainJob = null;
}

function setTrainStats(e) {
  const f = (v) => (v === undefined ? "—" : v.toExponential(3));
  document.getElementById("tr-epoch").textContent = e.epoch ?? "—";
  document.getElementById("tr-pde").textContent = f(e.loss_pde);
  document.getElementById("tr-bc").textContent = f(e.loss_bc);
  document.getElementById("tr-ic").textContent = f(e.loss_ic);
  document.getElementById("tr-lr").textContent =
    e.lr !== undefined ? e.lr.toExponential(2) : "—";
  if (e.elapsed !== undefined) {
    const m = Math.floor(e.elapsed / 60), s = Math.round(e.elapsed % 60);
    document.getElementById("tr-time").textContent = m ? `${m} min ${s} s` : `${s} s`;
  }
}

function drawTrainCharts() {
  lossChart.setSeries([
    { name: "Total", x: trainHist.epoch, y: trainHist.total },
    { name: "PDE", x: trainHist.epoch, y: trainHist.pde },
    { name: "BC", x: trainHist.epoch, y: trainHist.bc },
    { name: "IC", x: trainHist.epoch, y: trainHist.ic },
  ]);
  lrChart.setSeries(trainHist.lr.length
    ? [{ name: "lr", x: trainHist.epoch, y: trainHist.lr }] : []);
}

/* ═══════════════ COMPARACIÓN 2×2 SINCRONIZADA ═══════════════ */
export function initCompare() {
  document.getElementById("cmp-err-mode").addEventListener("change", drawCompare);
  document.getElementById("cmp-err-scale").addEventListener("change", drawCompare);
  on("frame", () => { if (isCompareVisible()) drawCompare(); });
  on("result", () => { drawCompare(); renderMetrics(); });
  window.addEventListener("resize", () => { if (isCompareVisible()) drawCompare(); });
}

function isCompareVisible() {
  return !document.getElementById("compare-container").hidden;
}

function frameOf(name) {
  const f = state.frames[name];
  if (!f) return null;
  const k = Math.min(state.frame, f.K - 1);
  return { arr: f.data.subarray(k * f.Nx * f.Ny, (k + 1) * f.Nx * f.Ny),
           Nx: f.Nx, Ny: f.Ny };
}

export function drawCompare() {
  if (!isCompareVisible()) return;
  const has = state.frames.Ez_pinn && state.frames.Ez_fdtd;
  document.getElementById("cmp-no-data").style.display = has ? "none" : "";
  if (!has) return;

  const pinn = frameOf("Ez_pinn");
  const fdtd = frameOf("Ez_fdtd");
  const errMode = document.getElementById("cmp-err-mode").value;
  const err = frameOf(errMode === "rel" ? "err_rel" : "err_abs");
  document.getElementById("cmp-err-title").textContent =
    errMode === "rel" ? "Error relativo (%)" : "|PINN − FDTD|";

  // escala compartida PINN/FDTD
  let vmax = 0;
  for (let n = 0; n < fdtd.arr.length; n++)
    vmax = Math.max(vmax, Math.abs(fdtd.arr[n]), Math.abs(pinn.arr[n]));
  vmax = vmax || 1;

  drawHeatmap(document.getElementById("cmp-pinn"), pinn.arr, pinn.Nx, pinn.Ny,
              { vmin: -vmax, vmax });
  drawHeatmap(document.getElementById("cmp-fdtd"), fdtd.arr, fdtd.Nx, fdtd.Ny,
              { vmin: -vmax, vmax });

  if (err) {
    let opts = { cmap: "magma", symmetric: false };
    if (document.getElementById("cmp-err-scale").value === "global") {
      const all = state.frames[errMode === "rel" ? "err_rel" : "err_abs"];
      let gmax = 0;
      for (let n = 0; n < all.data.length; n++)
        if (all.data[n] > gmax) gmax = all.data[n];
      opts.vmin = 0; opts.vmax = gmax || 1;
    }
    drawHeatmap(document.getElementById("cmp-err"), err.arr, err.Nx, err.Ny, opts);
  }
}

export function renderMetrics() {
  const body = document.getElementById("cmp-metrics-body");
  const m = state.resultMeta?.metrics;
  if (!m || m.l2_Ez === undefined) {
    body.innerHTML = `<p class="muted">Sin métricas de comparación.
      Ejecuta «Simulación → Ejecutar ambos».</p>`;
    return;
  }
  const fmtMB = (b) => (b / 1e6).toFixed(1) + " MB";
  const fmtT = (s) => (s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`
                              : `${s.toFixed(2)} s`);
  const rows = [
    ["sep", "Errores (Ez, todo el espacio-tiempo)"],
    ["Error RMS", `${m.rms_Ez.toFixed(4)} %`, "good"],
    ["Error L2", `${m.l2_Ez.toFixed(4)} %`, "good"],
    ["Error L2 (absoluto)", m.l2_abs_Ez.toExponential(3)],
    ["Error máximo", `${m.max_Ez.toFixed(4)} %`],
    ["L2 Bx / By", `${m.l2_Bx.toFixed(3)} / ${m.l2_By.toFixed(3)} %`],
    ["sep", "Tiempo de simulación"],
    ["PINN (inferencia)", fmtT(m.pinn_wall_time), "accent"],
    ["FDTD", fmtT(m.fdtd_wall_time), "accent"],
    ["sep", "Uso de memoria"],
    ["PINN (pesos)", m.pinn_memory_bytes ? fmtMB(m.pinn_memory_bytes) : "—"],
    ["FDTD (campos)", fmtMB(m.fdtd_memory_bytes)],
    ["sep", "Iteraciones"],
    ["PINN", m.pinn_epochs ? `${m.pinn_epochs} épocas` : "—"],
    ["FDTD", `${m.fdtd_steps} pasos`],
    ["Parámetros PINN", m.pinn_params ? m.pinn_params.toLocaleString("es") : "—"],
  ];
  body.innerHTML = rows.map((r) =>
    r[0] === "sep"
      ? `<div class="metric-sep">${r[1]}</div>`
      : `<div class="metric-row"><span class="k">${r[0]}</span>
         <span class="v ${r[2] ?? ""}">${r[1]}</span></div>`).join("");
}

/* ═══════════════ REPRODUCTOR ═══════════════ */
let playTimer = null;

export function initPlayer() {
  const slider = document.getElementById("pl-slider");
  const btnPlay = document.getElementById("pl-play");
  const btnPause = document.getElementById("pl-pause");

  slider.addEventListener("input", () => setFrame(+slider.value));
  document.getElementById("pl-first").addEventListener("click", () => setFrame(0));
  document.getElementById("pl-last").addEventListener("click", () =>
    setFrame(frameTimes().length - 1));
  document.getElementById("pl-prev").addEventListener("click", () =>
    setFrame(state.frame - 1));
  btnPlay.addEventListener("click", play);
  btnPause.addEventListener("click", pause);
  document.getElementById("pl-speed").addEventListener("change", (e) => {
    state.speed = +e.target.value;
    if (state.playing) { pause(); play(); }
  });

  on("frame", (k) => {
    slider.value = k;
    const t = frameTimes()[k] ?? 0;
    document.getElementById("pl-time").textContent = `t = ${t.toFixed(3)}`;
  });
  on("result", () => {
    slider.max = Math.max((state.resultMeta?.times?.length ?? 60) - 1, 1);
    setFrame(0);
  });
}

export function play() {
  if (state.playing) return;
  state.playing = true;
  document.getElementById("pl-play").hidden = true;
  document.getElementById("pl-pause").hidden = false;
  const base = 90;  // ms por frame a 1×
  playTimer = setInterval(() => {
    const K = frameTimes().length;
    setFrame((state.frame + 1) % K);
  }, base / state.speed);
}

export function pause() {
  state.playing = false;
  document.getElementById("pl-play").hidden = false;
  document.getElementById("pl-pause").hidden = true;
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
}

export function togglePlay() { state.playing ? pause() : play(); }

/* ═══════════════ GRÁFICAS ═══════════════ */
let graphChart, activeGraph = "point";
let residualCache = null;

export function initGraphs() {
  graphChart = new LineChart(document.getElementById("graph-canvas"), {});
  document.querySelectorAll(".gtab").forEach((t) =>
    t.addEventListener("click", () => {
      document.querySelectorAll(".gtab").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      activeGraph = t.dataset.graph;
      drawGraph();
    }));
  on("result", drawGraph);
  window.addEventListener("resize", () => {
    if (!document.getElementById("dock-graphs").hidden) drawGraph();
  });
}

export async function drawGraph() {
  const meta = state.resultMeta;
  const note = document.getElementById("graph-note");
  note.textContent = "";
  const fd = meta?.fdtd, pn = meta?.pinn;

  if (activeGraph === "point") {
    const series = [];
    if (fd?.probes) {
      const name = Object.keys(fd.probes)[0];
      series.push({ name: `FDTD — ${name}`, x: fd.probe_t, y: fd.probes[name] });
    }
    if (pn?.probes) {
      const name = Object.keys(pn.probes)[0];
      series.push({ name: `PINN — ${name}`, x: pn.probe_t, y: pn.probes[name] });
    }
    graphChart.opts.title = "Ez(t) en la sonda";
    graphChart.opts.xlabel = "t";
    graphChart.opts.logY = false;
    graphChart.setSeries(series);
    if (!series.length) note.textContent = "Ejecuta una simulación con una sonda puntual activa.";

  } else if (activeGraph === "fft") {
    const series = [];
    const add = (src, label) => {
      if (!src?.probes) return;
      const name = Object.keys(src.probes)[0];
      const y = src.probes[name];
      if (y.length < 8) return;
      const dt = src.probe_t[1] - src.probe_t[0];
      const { freq, mag } = fftMag(y, dt);
      // limitar a frecuencias útiles (hasta 4× la fundamental TM11 ≈ 0.707)
      const cut = freq.findIndex((f) => f > 3);
      series.push({ name: label, x: freq.slice(1, cut > 0 ? cut : undefined),
                    y: mag.slice(1, cut > 0 ? cut : undefined) });
    };
    add(fd, "FDTD");
    add(pn, "PINN");
    graphChart.opts.title = "Espectro |FFT{Ez}| — f fundamental TM₁₁ = ω/2π ≈ 0.707";
    graphChart.opts.xlabel = "frecuencia (unid. naturales)";
    graphChart.opts.logY = false;
    graphChart.setSeries(series);
    if (!series.length) note.textContent = "Se necesita la serie temporal de una sonda.";

  } else if (activeGraph === "energy") {
    const series = [];
    if (fd?.energy) series.push({ name: "FDTD", x: fd.energy_t, y: fd.energy });
    if (pn?.energy) series.push({ name: "PINN", x: pn.energy_t, y: pn.energy });
    graphChart.opts.title = "Energía electromagnética total U(t) — debe conservarse";
    graphChart.opts.xlabel = "t";
    graphChart.opts.logY = false;
    graphChart.setSeries(series);
    if (fd?.energy?.length) {
      const E = fd.energy;
      const drift = Math.abs(E[E.length - 1] - E[0]) / E[0] * 100;
      note.textContent = `Deriva de energía FDTD: ${drift.toFixed(4)} %`;
    }

  } else if (activeGraph === "error") {
    const es = meta?.error_series;
    graphChart.opts.title = "Error PINN vs FDTD en el tiempo";
    graphChart.opts.xlabel = "t";
    graphChart.opts.logY = false;
    graphChart.setSeries(es ? [
      { name: "L2 %", x: es.t, y: es.l2 },
      { name: "RMS %", x: es.t, y: es.rms },
      { name: "máx %", x: es.t, y: es.max },
    ] : []);
    if (!es) note.textContent = "Ejecuta «Simulación → Ejecutar ambos».";

  } else if (activeGraph === "residual") {
    graphChart.opts.title = "Residuales de las ecuaciones de Maxwell |R| del PINN (autograd)";
    graphChart.opts.xlabel = "t";
    graphChart.opts.logY = true;
    if (!residualCache) {
      note.textContent = "Calculando residuales vía autograd en el servidor…";
      graphChart.setSeries([]);
      try {
        residualCache = await api.residualSeries({ n_t: 24, N: 14 });
      } catch (e) {
        note.textContent = `Error: ${e.message}`;
        return;
      }
      note.textContent = "";
    }
    const r = residualCache;
    graphChart.setSeries([
      { name: "Faraday x", x: r.t, y: r.faraday_x },
      { name: "Faraday y", x: r.t, y: r.faraday_y },
      { name: "Ampère", x: r.t, y: r.ampere },
      { name: "Gauss B", x: r.t, y: r.gauss_b },
    ]);
  }
}

export function invalidateResiduals() { residualCache = null; }

/* ═══════════════ EXPORTACIÓN ═══════════════ */
export function initExport() {
  document.getElementById("exp-go").addEventListener("click", async () => {
    const dlg = document.getElementById("dlg-export");
    const fmt = document.getElementById("exp-format").value;
    const field = document.getElementById("exp-field").value;
    const frame = +document.getElementById("exp-frame").value;
    if (!state.resultId) { consoleLog("No hay resultados que exportar.", "warn"); return; }
    dlg.close();
    consoleLog(`Exportando ${fmt}…`);
    try {
      await api.export({ result_id: state.resultId, format: fmt, field, frame });
      consoleLog(`Exportación ${fmt} completada — revisa tus descargas.`);
    } catch (e) {
      consoleLog(`Error exportando ${fmt}: ${e.message}`, "error");
    }
  });
}

export function openExportDialog(preset) {
  const dlg = document.getElementById("dlg-export");
  const info = document.getElementById("exp-result-info");
  const fieldSel = document.getElementById("exp-field");
  if (!state.resultId) {
    consoleLog("Primero ejecuta una simulación (Simulación → Ejecutar…).", "warn");
    return;
  }
  const meta = state.resultMeta;
  info.textContent = `Resultado activo: ${meta.mode} · malla ${meta.grid?.Nx}×${meta.grid?.Ny} · ${meta.times?.length} frames`;
  fieldSel.innerHTML = (meta.fields ?? [])
    .map((f) => `<option value="${f}">${f}</option>`).join("");
  document.getElementById("exp-frame").max = (meta.times?.length ?? 1) - 1;
  document.getElementById("exp-frame").value = state.frame;
  if (preset) document.getElementById("exp-format").value = preset;
  dlg.showModal();
}

/* Grabación WebM del viewport (Exportar video) */
export async function recordViewportVideo(canvas, seconds = 6) {
  if (!canvas.captureStream) {
    consoleLog("Este navegador no soporta captura de canvas.", "error");
    return;
  }
  consoleLog(`Grabando video del viewport (${seconds} s)…`);
  const stream = canvas.captureStream(30);
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
    ? "video/webm;codecs=vp9" : "video/webm";
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 6e6 });
  const chunks = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  const done = new Promise((res) => (rec.onstop = res));
  const wasPlaying = state.playing;
  if (!wasPlaying) play();
  rec.start();
  await new Promise((res) => setTimeout(res, seconds * 1000));
  rec.stop();
  await done;
  if (!wasPlaying) pause();
  const blob = new Blob(chunks, { type: "video/webm" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `fluxia_viewport_${Date.now()}.webm`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  consoleLog("Video WebM exportado.");
}
