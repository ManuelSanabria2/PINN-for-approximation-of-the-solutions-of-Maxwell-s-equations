/* state.js — estado central de la aplicación (proyecto, selección, resultados) */

const listeners = {};

export function on(topic, fn) {
  (listeners[topic] ??= []).push(fn);
}
export function fire(topic, payload) {
  for (const fn of listeners[topic] ?? []) {
    try { fn(payload); } catch (e) { console.error(`listener ${topic}:`, e); }
  }
}

export const state = {
  project: null,          // JSON del proyecto (editable)
  dirty: false,

  selection: null,        // {kind: "material"|"geometry"|"source"|"monitor"|"domain"|"solver-pinn"|"solver-fdtd"|"result", id}

  // Resultado activo
  resultId: null,
  resultMeta: null,       // meta del backend (metrics, times, series…)
  frames: {},             // {Ez_fdtd: {data: Float32Array, K, Nx, Ny}, ...}

  // Reproductor
  frame: 0,
  playing: false,
  speed: 1,

  // Visualización
  field: "E",             // E | H | energy | poynting | errPINN | errFDTD
  show: { iso: false, sliceXY: true, sliceXZ: false, sliceYZ: false,
          mesh: false, axes: true },
  hidden: new Set(),      // ids de objetos ocultos en la vista 3D

  // Jobs
  activeJob: null,
  trainJob: null,
};

export function setProject(p) {
  state.project = p;
  state.dirty = false;
  fire("project", p);
}

export function touchProject() {
  state.dirty = true;
  fire("project", state.project);
}

export function select(kind, id) {
  state.selection = kind ? { kind, id } : null;
  fire("selection", state.selection);
}

export function setFrame(k) {
  const K = state.resultMeta?.times?.length ?? 60;
  state.frame = Math.max(0, Math.min(k, K - 1));
  fire("frame", state.frame);
}

export function setField(f) {
  state.field = f;
  fire("field", f);
}

export function toggleShow(key) {
  state.show[key] = !state.show[key];
  fire("show", state.show);
}

export function toggleHidden(id) {
  if (state.hidden.has(id)) state.hidden.delete(id);
  else state.hidden.add(id);
  fire("hidden", state.hidden);
}

export async function setResult(rid, meta, frames) {
  state.resultId = rid;
  state.resultMeta = meta;
  state.frames = frames;
  state.frame = 0;
  fire("result", { rid, meta });
}

/* Utilidades sobre el proyecto */
export function findMaterial(id) {
  return state.project?.materials?.find((m) => m.id === id);
}
export function findObject(kind, id) {
  const p = state.project;
  if (!p) return null;
  const bag = { material: p.materials, geometry: p.geometry,
                source: p.sources, monitor: p.monitors }[kind];
  return bag?.find((o) => o.id === id);
}

/* Tiempos del resultado o del dominio por defecto */
export function frameTimes() {
  if (state.resultMeta?.times) return state.resultMeta.times;
  const n = state.project?.domain?.n_snapshots ?? 60;
  const T = state.project?.domain?.T_max ?? 2.828427;
  return Array.from({ length: n }, (_, i) => (i * T) / (n - 1));
}
