/* api.js — cliente REST + WebSocket del backend Fluxia */

const BASE = "";   // mismo origen

async function jget(url) {
  const r = await fetch(BASE + url);
  if (!r.ok) throw new Error(`${url} → ${r.status}: ${await r.text()}`);
  return r.json();
}

async function jpost(url, body) {
  const r = await fetch(BASE + url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (!r.ok) throw new Error(`${url} → ${r.status}: ${await r.text()}`);
  return r.json();
}

export const api = {
  health:          ()        => jget("/api/health"),
  version:         ()        => jget("/api/version"),
  defaultProject:  ()        => jget("/api/project/default"),
  validateProject: (project) => jpost("/api/project/validate", { project }),

  run:   (mode, project) => jpost("/api/run", { mode, project }),
  train: (opts)          => jpost("/api/train", opts),
  trainStop: ()          => jpost("/api/train/stop"),
  job:   (id)            => jget(`/api/jobs/${id}`),
  cancelJob: (id)        => jpost(`/api/jobs/${id}/cancel`),

  resultMeta: (rid)      => jget(`/api/results/${rid}/meta`),

  async resultFrames(rid, field) {
    const r = await fetch(`${BASE}/api/results/${rid}/frames/${field}`);
    if (!r.ok) throw new Error(`frames ${field} → ${r.status}`);
    const shape = (r.headers.get("X-Shape") || "").split(",").map(Number);
    const buf = await r.arrayBuffer();
    return { data: new Float32Array(buf), K: shape[0], Nx: shape[1], Ny: shape[2] };
  },

  gpu:       ()     => jget("/api/gpu"),
  gpuConfig: (opts) => jpost("/api/gpu", opts),
  toolMesh:  (opts) => jpost("/api/tools/mesh", opts),
  toolRefine:(opts) => jpost("/api/tools/refine", opts),

  slice:     (opts) => jpost("/slice", opts),
  residuals: (opts) => jpost("/residuals", opts),
  residualSeries: (opts) => jpost("/api/graphs/residual_series", opts),
  trainingHistory: ()   => jget("/api/training/history"),

  /* Exportación: dispara descarga del archivo */
  async export(opts) {
    const r = await fetch(BASE + "/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts),
    });
    if (!r.ok) throw new Error(`export → ${r.status}: ${await r.text()}`);
    const blob = await r.blob();
    const cd = r.headers.get("Content-Disposition") || "";
    const m = cd.match(/filename="?([^";]+)"?/);
    downloadBlob(blob, m ? m[1] : "export.bin");
  },

  async exportStl(project) {
    const r = await fetch(BASE + "/api/export/stl", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project }),
    });
    if (!r.ok) throw new Error(`stl → ${r.status}`);
    downloadBlob(await r.blob(), "geometria.stl");
  },
};

export function downloadBlob(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

/* ─── Bus de eventos del servidor (WS con fallback a polling) ─── */
export function connectEvents(onEvent, onStatus) {
  let lastSeq = 0;
  let pollTimer = null;

  function handle(events) {
    for (const e of events) {
      if (e.seq > lastSeq) lastSeq = e.seq;
      onEvent(e);
    }
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try {
        const r = await jget(`/api/events?since=${lastSeq}`);
        handle(r.events);
        onStatus?.("online");
      } catch { onStatus?.("offline"); }
    }, 700);
  }

  function connectWS() {
    try {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws/events`);
      ws.onmessage = (m) => { try { handle(JSON.parse(m.data).events); } catch {} };
      ws.onopen = () => { onStatus?.("online"); if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } };
      ws.onclose = () => { onStatus?.("offline"); startPolling(); setTimeout(connectWS, 4000); };
      ws.onerror = () => ws.close();
    } catch { startPolling(); }
  }

  connectWS();
}

/* Espera a que un job termine (polling ligero de respaldo) */
export async function waitJob(jobId, { interval = 600 } = {}) {
  for (;;) {
    const st = await api.job(jobId);
    if (st.status === "done") return st;
    if (st.status === "error") throw new Error(st.error || "Job falló");
    await new Promise((res) => setTimeout(res, interval));
  }
}
