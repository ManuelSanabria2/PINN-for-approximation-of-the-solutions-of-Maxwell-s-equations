/* ui.js — árbol del proyecto, propiedades, consola, menús y diálogos de herramientas */
import { state, on, fire, select, touchProject, toggleHidden, setField,
         toggleShow, findObject } from "./state.js";
import { api } from "./api.js";

/* ═══════════════ CONSOLA ═══════════════ */
const conOut = () => document.getElementById("console-out");

export function consoleLog(msg, level = "info") {
  const el = document.createElement("span");
  el.className = `con-line con-${level}`;
  const t = new Date().toLocaleTimeString("es", { hour12: false });
  el.innerHTML = `<span class="con-time">${t}</span>${escapeHtml(msg)}`;
  const out = conOut();
  out.appendChild(el);
  while (out.children.length > 800) out.removeChild(out.firstChild);
  out.parentElement.scrollTop = out.parentElement.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

/* ═══════════════ MENÚS ═══════════════ */
export function initMenus(actions) {
  const menus = document.querySelectorAll("#menus .menu");
  menus.forEach((m) => {
    m.querySelector(".menu-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      const wasOpen = m.classList.contains("open");
      menus.forEach((x) => x.classList.remove("open"));
      if (!wasOpen) m.classList.add("open");
    });
    m.addEventListener("mouseenter", () => {
      if (document.querySelector("#menus .menu.open")) {
        menus.forEach((x) => x.classList.remove("open"));
        m.classList.add("open");
      }
    });
  });
  document.addEventListener("click", () => menus.forEach((x) => x.classList.remove("open")));

  // Acciones
  document.querySelectorAll(".menu-drop button[data-action]").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#menus .menu").forEach((x) => x.classList.remove("open"));
      actions[b.dataset.action]?.();
    });
  });

  // Visualización: campo activo (exclusivo) + toggles
  const visMenu = document.getElementById("menu-vis");
  visMenu.querySelectorAll("button[data-field]").forEach((b) => {
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      setField(b.dataset.field);
      syncVisMenu();
    });
  });
  visMenu.querySelectorAll("button[data-toggle]").forEach((b) => {
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleShow(b.dataset.toggle);
      syncVisMenu();
    });
  });
  syncVisMenu();

  // Atajos de teclado
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, select")) return;
    if (e.ctrlKey && e.key === "s") { e.preventDefault(); actions["file-save"]?.(); }
    else if (e.ctrlKey && e.key === "o") { e.preventDefault(); actions["file-open"]?.(); }
    else if (e.ctrlKey && e.key === "n") { e.preventDefault(); actions["file-new"]?.(); }
    else if (e.key === "F5") { e.preventDefault(); actions["run-pinn"]?.(); }
    else if (e.key === "F6") { e.preventDefault(); actions["run-fdtd"]?.(); }
    else if (e.key === "F7") { e.preventDefault(); actions["run-both"]?.(); }
    else if (e.key === " ") { e.preventDefault(); actions["play-pause"]?.(); }
  });
}

export function syncVisMenu() {
  const visMenu = document.getElementById("menu-vis");
  visMenu.querySelectorAll("button[data-field]").forEach((b) =>
    b.classList.toggle("checked", b.dataset.field === state.field));
  visMenu.querySelectorAll("button[data-toggle]").forEach((b) =>
    b.classList.toggle("checked", !!state.show[b.dataset.toggle]));
}

/* ═══════════════ ÁRBOL DEL PROYECTO ═══════════════ */
const ICONS = {
  geometry: "▣", material: "◉", source: "⚡", monitor: "▤",
  solver: "ƒ", results: "▦", domain: "▢",
};

export function renderTree() {
  const p = state.project;
  const tree = document.getElementById("tree");
  tree.innerHTML = "";
  if (!p) return;

  const root = mkNode(`Proyecto — ${p.name}`, "▦", null, null, { bold: true });

  const gGeo = mkNode("Geometría", "▼", null, null, { group: true });
  addChild(root, gGeo);
  addChild(gGeo, mkNode("Caja (dominio)", ICONS.domain, "domain", "domain"));
  for (const g of p.geometry ?? []) {
    if (g.shape === "domain") continue;
    addChild(gGeo, mkNode(g.name, ICONS.geometry, "geometry", g.id,
                          { eye: true, disabled: g.enabled === false }));
  }

  const gMat = mkNode("Materiales", "▼", null, null, { group: true });
  addChild(root, gMat);
  for (const m of p.materials ?? []) {
    addChild(gMat, mkNode(m.name, null, "material", m.id, { dot: m.color }));
  }

  const gSrc = mkNode("Fuentes", "▼", null, null, { group: true });
  addChild(root, gSrc);
  for (const s of p.sources ?? []) {
    addChild(gSrc, mkNode(s.name, ICONS.source, "source", s.id,
                          { eye: true, disabled: s.enabled === false }));
  }

  const gMon = mkNode("Monitores", "▼", null, null, { group: true });
  addChild(root, gMon);
  for (const m of p.monitors ?? []) {
    addChild(gMon, mkNode(m.name, ICONS.monitor, "monitor", m.id,
                          { eye: true, disabled: m.enabled === false }));
  }

  addChild(root, mkNode("Solver PINN", ICONS.solver, "solver-pinn", "solver-pinn"));
  addChild(root, mkNode("Solver FDTD", ICONS.solver, "solver-fdtd", "solver-fdtd"));

  const gRes = mkNode("Resultados", "▼", null, null, { group: true });
  addChild(root, gRes);
  if (state.resultMeta) {
    const label = { pinn: "PINN", fdtd: "FDTD", both: "PINN + FDTD" }[state.resultMeta.mode]
                  ?? state.resultMeta.mode;
    addChild(gRes, mkNode(`Resultado: ${label} (${state.resultMeta.times?.length ?? 0} frames)`,
                          ICONS.results, "result", state.resultId));
  } else {
    addChild(gRes, mkNode("— sin resultados —", "·", null, null, { muted: true }));
  }

  tree.appendChild(root.node);
  highlightSelection();
}

function mkNode(label, icon, kind, id, opts = {}) {
  const node = document.createElement("div");
  node.className = "tree-node";
  const row = document.createElement("div");
  row.className = "tree-row";
  if (kind && id) { row.dataset.kind = kind; row.dataset.id = id; }

  const caret = document.createElement("span");
  caret.className = "tree-caret";
  caret.textContent = "▾";
  caret.style.visibility = "hidden";
  row.appendChild(caret);

  if (opts.dot) {
    const d = document.createElement("span");
    d.className = "mat-dot";
    d.style.background = opts.dot;
    row.appendChild(d);
  } else if (icon) {
    const ic = document.createElement("span");
    ic.className = "tree-icon";
    ic.textContent = icon === "▼" ? "" : icon;
    row.appendChild(ic);
  }

  const lb = document.createElement("span");
  lb.className = "tree-label" + (opts.disabled ? " disabled" : "");
  if (opts.muted) lb.classList.add("muted");
  if (opts.bold) lb.style.fontWeight = "600";
  if (opts.group) { lb.style.color = "var(--text-muted)"; lb.style.fontSize = "11px";
                    lb.style.textTransform = "uppercase"; lb.style.letterSpacing = ".06em"; }
  lb.textContent = label;
  row.appendChild(lb);

  if (opts.eye && kind && id) {
    const eye = document.createElement("button");
    eye.className = "tree-eye" + (state.hidden.has(id) ? " off" : "");
    eye.textContent = state.hidden.has(id) ? "◌" : "◉";
    eye.title = "Mostrar/ocultar en la vista 3D";
    eye.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleHidden(id);
      renderTree();
    });
    row.appendChild(eye);
  }

  node.appendChild(row);
  const children = document.createElement("div");
  children.className = "tree-children";
  node.appendChild(children);

  caret.addEventListener("click", (e) => {
    e.stopPropagation();
    node.classList.toggle("collapsed");
  });
  if (kind && id) {
    row.addEventListener("click", () => select(kind, id));
  } else if (opts.group) {
    row.addEventListener("click", () => node.classList.toggle("collapsed"));
  }

  return { node, row, children, caret };
}

function addChild(parent, child) {
  parent.children.appendChild(child.node);
  parent.caret.style.visibility = "visible";
}

function highlightSelection() {
  document.querySelectorAll(".tree-row").forEach((r) => {
    const sel = state.selection;
    r.classList.toggle("selected",
      !!sel && r.dataset.kind === sel.kind && r.dataset.id === sel.id);
  });
}

/* ═══════════════ PANEL DE PROPIEDADES ═══════════════ */
export function renderProps() {
  const body = document.getElementById("props-body");
  const sel = state.selection;
  const p = state.project;
  highlightSelection();
  if (!sel || !p) {
    body.innerHTML = `<p class="muted">Selecciona un objeto del árbol o de la vista 3D.</p>`;
    return;
  }

  if (sel.kind === "domain") return renderDomainProps(body);
  if (sel.kind === "material") return renderMaterialProps(body, sel.id);
  if (sel.kind === "source") return renderSourceProps(body, sel.id);
  if (sel.kind === "geometry") return renderGeometryProps(body, sel.id);
  if (sel.kind === "monitor") return renderMonitorProps(body, sel.id);
  if (sel.kind === "solver-pinn") return renderSolverPinn(body);
  if (sel.kind === "solver-fdtd") return renderSolverFdtd(body);
  if (sel.kind === "result") return renderResultProps(body);
  body.innerHTML = "";
}

function h(html) { const d = document.createElement("div"); d.innerHTML = html; return d; }

function numRow(label, value, onchange, opts = {}) {
  const row = h(`<div class="prop-row"><label>${label}</label>
    <input type="number" value="${value}" step="${opts.step ?? "any"}"
      ${opts.min !== undefined ? `min="${opts.min}"` : ""}
      ${opts.max !== undefined ? `max="${opts.max}"` : ""}></div>`);
  row.querySelector("input").addEventListener("change", (e) => {
    onchange(parseFloat(e.target.value));
    touchProject();
  });
  return row;
}

function selectRow(label, value, options, onchange) {
  const row = h(`<div class="prop-row"><label>${label}</label>
    <select>${options.map((o) =>
      `<option value="${o[0]}" ${o[0] === value ? "selected" : ""}>${o[1]}</option>`).join("")}
    </select></div>`);
  row.querySelector("select").addEventListener("change", (e) => {
    onchange(e.target.value);
    touchProject();
  });
  return row;
}

function checkRow(label, value, onchange) {
  const row = h(`<div class="prop-row"><label>${label}</label>
    <input type="checkbox" ${value ? "checked" : ""}></div>`);
  row.querySelector("input").addEventListener("change", (e) => {
    onchange(e.target.checked);
    touchProject();
  });
  return row;
}

function staticRow(label, value) {
  return h(`<div class="prop-row"><label>${label}</label>
    <span class="prop-static">${value}</span></div>`);
}

function renderDomainProps(body) {
  const d = state.project.domain;
  body.innerHTML = `<h4>▢ Dominio computacional</h4>
    <div class="prop-type">Cavidad PEC · unidades naturales (c = 1)</div>`;
  body.appendChild(h(`<div class="prop-group">Tamaño</div>`));
  body.appendChild(numRow("L", d.L, (v) => (d.L = v), { step: 0.1, min: 0.1 }));
  body.appendChild(numRow("Nx", d.Nx, (v) => (d.Nx = Math.round(v)), { step: 1, min: 11, max: 401 }));
  body.appendChild(numRow("Ny", d.Ny, (v) => (d.Ny = Math.round(v)), { step: 1, min: 11, max: 401 }));
  const dx = d.L / (d.Nx - 1), dy = d.L / (d.Ny - 1);
  body.appendChild(staticRow("Δx", dx.toFixed(5)));
  body.appendChild(staticRow("Δy", dy.toFixed(5)));
  body.appendChild(h(`<div class="prop-group">Tiempo</div>`));
  body.appendChild(numRow("T máximo", d.T_max, (v) => (d.T_max = v), { step: 0.1, min: 0.1 }));
  body.appendChild(numRow("Courant S", d.courant, (v) => (d.courant = v),
                          { step: 0.05, min: 0.05, max: 0.7 }));
  const dt = d.courant * dx;
  body.appendChild(staticRow("Δt (CFL)", dt.toFixed(6)));
  body.appendChild(staticRow("Pasos FDTD", Math.ceil(d.T_max / dt)));
  body.appendChild(numRow("Snapshots", d.n_snapshots, (v) => (d.n_snapshots = Math.round(v)),
                          { step: 1, min: 10, max: 200 }));
  body.appendChild(h(`<p class="prop-note">Condición de Courant 2D: S ≤ 1/√2 ≈ 0.707.
    El PINN fue entrenado en t ∈ [0, 2.83] (2 periodos TM₁₁).</p>`));
}

function renderMaterialProps(body, id) {
  const m = findObject("material", id);
  if (!m) return;
  body.innerHTML = `<h4><span class="mat-dot" style="background:${m.color}"></span> ${m.name}</h4>
    <div class="prop-type">Material</div>`;
  const nameRow = h(`<div class="prop-row"><label>Nombre</label>
    <input type="text" value="${m.name}"></div>`);
  nameRow.querySelector("input").addEventListener("change", (e) => {
    m.name = e.target.value; touchProject();
  });
  body.appendChild(nameRow);
  body.appendChild(numRow("Permitividad ε_r", m.eps_r, (v) => (m.eps_r = v), { step: 0.1, min: 1 }));
  body.appendChild(numRow("Permeabilidad μ_r", m.mu_r, (v) => (m.mu_r = v), { step: 0.1, min: 0.1 }));
  body.appendChild(numRow("Conductividad σ", m.sigma, (v) => (m.sigma = v), { step: 0.1, min: 0 }));
  const colorRow = h(`<div class="prop-row"><label>Color</label>
    <input type="color" value="${m.color}"></div>`);
  colorRow.querySelector("input").addEventListener("input", (e) => {
    m.color = e.target.value; touchProject();
  });
  body.appendChild(colorRow);
  body.appendChild(selectRow("Modelo dispersivo", m.dispersive ?? "none",
    [["none", "No dispersivo"], ["lorentz", "Lorentz"], ["drude", "Drude"], ["debye", "Debye"]],
    (v) => (m.dispersive = v)));
  if ((m.dispersive ?? "none") !== "none") {
    body.appendChild(h(`<p class="prop-note">⚠ Los modelos dispersivos (Lorentz/Drude/Debye)
      se registran en el proyecto pero el solver FDTD actual los trata con ε_r, μ_r, σ
      constantes (aproximación no dispersiva).</p>`));
  }
}

function renderSourceProps(body, id) {
  const s = findObject("source", id);
  if (!s) return;
  body.innerHTML = `<h4>⚡ ${s.name}</h4><div class="prop-type">Fuente</div>`;
  body.appendChild(checkRow("Activa", s.enabled !== false, (v) => (s.enabled = v)));
  body.appendChild(selectRow("Tipo", s.type,
    [["mode", "Modo resonante (IC)"], ["pulse", "Pulso gaussiano"]],
    (v) => { s.type = v; renderProps(); }));
  if (s.type === "mode") {
    body.appendChild(numRow("Amplitud E₀", s.E0 ?? 1, (v) => (s.E0 = v), { step: 0.1 }));
    body.appendChild(staticRow("Modo", `TM₁₁`));
    body.appendChild(staticRow("Frecuencia", "ω = π√2 ≈ 4.443"));
    body.appendChild(staticRow("Polarización", "Ez (TM)"));
    body.appendChild(h(`<p class="prop-note">Condición inicial: Ez = E₀·sin(πx/L)·sin(πy/L),
      B = 0. Es el problema para el que fue entrenado el PINN.</p>`));
  } else {
    body.appendChild(selectRow("Forma", s.kind ?? "pulse",
      [["pulse", "Pulso"], ["cw", "Continuo (CW)"]], (v) => (s.kind = v)));
    body.appendChild(numRow("Posición x", s.x ?? 0.5, (v) => (s.x = v), { step: 0.05, min: 0, max: 1 }));
    body.appendChild(numRow("Posición y", s.y ?? 0.5, (v) => (s.y = v), { step: 0.05, min: 0, max: 1 }));
    body.appendChild(numRow("Frecuencia", s.freq ?? 1.5, (v) => (s.freq = v), { step: 0.1, min: 0.1 }));
    body.appendChild(numRow("Potencia (amplitud)", s.amplitude ?? 1, (v) => (s.amplitude = v), { step: 0.5 }));
    body.appendChild(numRow("Retardo t₀", s.t0 ?? 0.5, (v) => (s.t0 = v), { step: 0.05, min: 0 }));
    body.appendChild(numRow("Duración τ", s.tau ?? 0.15, (v) => (s.tau = v), { step: 0.01, min: 0.01 }));
    body.appendChild(staticRow("Polarización", "Ez (TM)"));
    body.appendChild(staticRow("Dirección", "isótropa (soft source)"));
    body.appendChild(h(`<p class="prop-note">⚠ Con fuente de pulso el PINN pre-entrenado
      (modo TM₁₁) ya no es la referencia correcta — la comparación mostrará el
      error honesto entre ambos.</p>`));
  }
}

function renderGeometryProps(body, id) {
  const g = findObject("geometry", id);
  if (!g) return;
  const mats = state.project.materials.map((m) => [m.id, m.name]);
  body.innerHTML = `<h4>▣ ${g.name}</h4><div class="prop-type">Geometría — ${g.shape}</div>`;
  body.appendChild(checkRow("Incluida en simulación", g.enabled !== false, (v) => (g.enabled = v)));
  body.appendChild(selectRow("Material", g.material, mats, (v) => (g.material = v)));
  if (g.shape === "sphere" || g.shape === "circle") {
    body.appendChild(numRow("Centro x", g.cx, (v) => (g.cx = v), { step: 0.05, min: 0, max: 1 }));
    body.appendChild(numRow("Centro y", g.cy, (v) => (g.cy = v), { step: 0.05, min: 0, max: 1 }));
    body.appendChild(numRow("Radio", g.r, (v) => (g.r = v), { step: 0.01, min: 0.01, max: 0.5 }));
  } else if (g.shape === "box") {
    body.appendChild(numRow("x₀", g.x0, (v) => (g.x0 = v), { step: 0.05, min: 0, max: 1 }));
    body.appendChild(numRow("y₀", g.y0, (v) => (g.y0 = v), { step: 0.05, min: 0, max: 1 }));
    body.appendChild(numRow("x₁", g.x1, (v) => (g.x1 = v), { step: 0.05, min: 0, max: 1 }));
    body.appendChild(numRow("y₁", g.y1, (v) => (g.y1 = v), { step: 0.05, min: 0, max: 1 }));
  }
}

function renderMonitorProps(body, id) {
  const m = findObject("monitor", id);
  if (!m) return;
  body.innerHTML = `<h4>▤ ${m.name}</h4><div class="prop-type">Monitor</div>`;
  body.appendChild(checkRow("Activo", m.enabled !== false, (v) => (m.enabled = v)));
  if (m.plane === "point") {
    body.appendChild(numRow("x", m.x, (v) => (m.x = v), { step: 0.05, min: 0, max: 1 }));
    body.appendChild(numRow("y", m.y, (v) => (m.y = v), { step: 0.05, min: 0, max: 1 }));
    body.appendChild(h(`<p class="prop-note">Registra Ez(t) en cada paso FDTD.
      Se usa en las gráficas «Campo en un punto» y «Espectro».</p>`));
  } else {
    body.appendChild(staticRow("Plano", m.plane.toUpperCase()));
    body.appendChild(numRow("Posición", m.position ?? 0.5, (v) => (m.position = v),
                            { step: 0.05, min: 0, max: 1 }));
  }
}

function renderSolverPinn(body) {
  const sp = state.project.solver_pinn ?? {};
  body.innerHTML = `<h4>ƒ Solver PINN</h4>
    <div class="prop-type">Physics-Informed Neural Network</div>`;
  body.appendChild(staticRow("Arquitectura", (sp.arch ?? []).join("–")));
  body.appendChild(staticRow("Fourier features", sp.fourier ? `sí (n=${sp.n_fourier}, σ=${sp.sigma})` : "no"));
  body.appendChild(staticRow("Épocas (pre-entr.)", sp.epochs ?? "—"));
  body.appendChild(h(`<div class="prop-group">Pérdida</div>`));
  body.appendChild(staticRow("L_PDE", "residuales de Maxwell"));
  body.appendChild(staticRow("L_BC", "Ez = 0 en paredes PEC ×10"));
  body.appendChild(staticRow("L_IC", "modo TM₁₁ en t=0 ×10"));
  body.appendChild(h(`<p class="prop-note">Usa <b>Simulación → Entrenar PINN…</b> para
    re-entrenar en vivo y ver las pérdidas en el panel de Entrenamiento.</p>`));
}

function renderSolverFdtd(body) {
  const d = state.project.domain;
  const dx = d.L / (d.Nx - 1);
  body.innerHTML = `<h4>ƒ Solver FDTD</h4>
    <div class="prop-type">Diferencias finitas en el dominio del tiempo</div>`;
  body.appendChild(staticRow("Esquema", "Yee 2D TM"));
  body.appendChild(staticRow("Orden", "2 (espacio y tiempo)"));
  body.appendChild(staticRow("Frontera", "PEC (Ez = 0)"));
  body.appendChild(staticRow("Malla", `${d.Nx} × ${d.Ny}`));
  body.appendChild(staticRow("Δt", (d.courant * dx).toFixed(6)));
  const pml = state.project.pml ?? {};
  body.appendChild(h(`<div class="prop-group">PML</div>`));
  body.appendChild(checkRow("Capas absorbentes", !!pml.enabled,
                            (v) => (state.project.pml = { ...pml, enabled: v })));
  body.appendChild(numRow("Celdas PML", pml.cells ?? 8,
                          (v) => (state.project.pml = { ...state.project.pml, cells: Math.round(v) }),
                          { step: 1, min: 4, max: 24 }));
  body.appendChild(h(`<p class="prop-note">Para la cavidad resonante se usa PEC puro
    (sin PML). Activa PML si simulas fuentes radiantes.</p>`));
}

function renderResultProps(body) {
  const meta = state.resultMeta;
  if (!meta) { body.innerHTML = ""; return; }
  const m = meta.metrics ?? {};
  body.innerHTML = `<h4>▦ Resultado</h4>
    <div class="prop-type">${{ pinn: "PINN", fdtd: "FDTD", both: "PINN vs FDTD" }[meta.mode] ?? meta.mode}</div>`;
  body.appendChild(staticRow("Frames", meta.times?.length ?? 0));
  body.appendChild(staticRow("Malla", `${meta.grid?.Nx} × ${meta.grid?.Ny}`));
  if (m.l2_Ez !== undefined) {
    body.appendChild(h(`<div class="prop-group">Errores (Ez)</div>`));
    body.appendChild(staticRow("L2", `${m.l2_Ez.toFixed(4)} %`));
    body.appendChild(staticRow("RMS", `${m.rms_Ez.toFixed(4)} %`));
    body.appendChild(staticRow("Máximo", `${m.max_Ez.toFixed(4)} %`));
  }
  if (m.fdtd_wall_time !== undefined)
    body.appendChild(staticRow("Tiempo FDTD", `${m.fdtd_wall_time.toFixed(2)} s`));
  if (m.pinn_wall_time !== undefined)
    body.appendChild(staticRow("Tiempo PINN", `${m.pinn_wall_time.toFixed(2)} s`));
}

/* ═══════════════ DIÁLOGOS DE HERRAMIENTAS ═══════════════ */
export function initDialogs() {
  // Cerrar genérico
  document.querySelectorAll("dialog [data-close]").forEach((b) =>
    b.addEventListener("click", () => b.closest("dialog").close()));

  // Generador de malla
  document.getElementById("mesh-calc").addEventListener("click", async () => {
    const out = document.getElementById("mesh-out");
    out.textContent = "Calculando…";
    try {
      const r = await api.toolMesh({
        m: +document.getElementById("mesh-m").value,
        n: +document.getElementById("mesh-n").value,
        ppw: +document.getElementById("mesh-ppw").value,
        courant: +document.getElementById("mesh-courant").value,
      });
      window._meshResult = r;
      out.textContent =
        `λ (modo TM${document.getElementById("mesh-m").value}${document.getElementById("mesh-n").value}) : ${r.lambda.toFixed(4)}\n` +
        `Nx = Ny recomendado : ${r.Nx}\n` +
        `Δx                  : ${r.dx.toFixed(5)}\n` +
        `Δt (CFL)            : ${r.dt.toFixed(6)}\n` +
        `Pasos de tiempo     : ${r.n_steps}\n` +
        `Puntos/λ efectivos  : ${r.ppw_effective.toFixed(1)}\n` +
        `ω del modo          : ${r.omega.toFixed(4)}  (T = ${r.T_period.toFixed(4)})`;
    } catch (e) { out.textContent = `Error: ${e.message}`; }
  });
  document.getElementById("mesh-apply").addEventListener("click", () => {
    const r = window._meshResult;
    if (!r) return;
    state.project.domain.Nx = r.Nx;
    state.project.domain.Ny = r.Nx;
    state.project.domain.courant = +document.getElementById("mesh-courant").value;
    touchProject();
    consoleLog(`Malla aplicada al dominio: ${r.Nx} × ${r.Nx}`);
    document.getElementById("dlg-mesh").close();
  });

  // Refinamiento adaptativo
  document.getElementById("refine-run").addEventListener("click", async () => {
    const out = document.getElementById("refine-out");
    out.textContent = "Ejecutando estudio de convergencia (FDTD a varias resoluciones)…";
    document.getElementById("refine-apply").disabled = true;
    try {
      const r = await api.toolRefine({
        project: state.project,
        target_error_pct: +document.getElementById("refine-target").value,
      });
      window._refineResult = r;
      let txt = "Nx      err. estimado    t_wall\n";
      for (const s of r.study)
        txt += `${String(s.Nx).padEnd(7)} ${s.err_est_pct.toFixed(5).padEnd(15)}% ${s.wall_time.toFixed(2)}s\n`;
      txt += `\nNx recomendado: ${r.recommended_Nx}  (objetivo ${r.target_error_pct}%)\n${r.note}`;
      out.textContent = txt;
      document.getElementById("refine-apply").disabled = false;
    } catch (e) { out.textContent = `Error: ${e.message}`; }
  });
  document.getElementById("refine-apply").addEventListener("click", () => {
    const r = window._refineResult;
    if (!r) return;
    state.project.domain.Nx = r.recommended_Nx;
    state.project.domain.Ny = r.recommended_Nx;
    touchProject();
    consoleLog(`Refinamiento aplicado: Nx = ${r.recommended_Nx}`);
    document.getElementById("dlg-refine").close();
  });

  // GPU
  document.getElementById("gpu-apply").addEventListener("click", async () => {
    try {
      const r = await api.gpuConfig({ num_threads: +document.getElementById("gpu-threads").value });
      showGpuInfo(r);
      consoleLog(`Configuración de cómputo aplicada (${r.num_threads} hilos)`);
    } catch (e) { consoleLog(`Error GPU: ${e.message}`, "error"); }
  });
}

export async function openGpuDialog() {
  const dlg = document.getElementById("dlg-gpu");
  dlg.showModal();
  try {
    const r = await api.gpu();
    showGpuInfo(r);
    document.getElementById("gpu-threads").value = r.num_threads;
  } catch (e) {
    document.getElementById("gpu-out").textContent = `Error: ${e.message}`;
  }
}

function showGpuInfo(r) {
  document.getElementById("gpu-out").textContent =
    `Dispositivo activo : ${r.device.toUpperCase()}\n` +
    `CUDA disponible    : ${r.cuda_available ? "sí — " + (r.gpu_name ?? "") : "no"}\n` +
    (r.gpu_memory_gb ? `Memoria GPU        : ${r.gpu_memory_gb.toFixed(1)} GB\n` : "") +
    `PyTorch            : ${r.torch_version}\n` +
    `Hilos CPU          : ${r.num_threads} (interop ${r.num_interop_threads})\n` +
    `Procesador         : ${r.cpu}`;
}

export function openMaterialsInspector() {
  const dlg = document.getElementById("dlg-materials");
  const tbody = dlg.querySelector("tbody");
  tbody.innerHTML = "";
  for (const m of state.project?.materials ?? []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><span class="mat-dot" style="background:${m.color}"></span></td>
      <td>${m.name}</td><td>${m.eps_r}</td><td>${m.mu_r}</td><td>${m.sigma}</td>
      <td>${m.dispersive ?? "none"}</td>`;
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => { dlg.close(); select("material", m.id); });
    tbody.appendChild(tr);
  }
  dlg.showModal();
}
