/* charts.js — gráficas de línea y mapas de calor sobre canvas 2D (sin dependencias) */

/* ─── Mapas de color ─── */
function lerp(a, b, t) { return a + (b - a) * t; }

const RDBU = [ // azul → blanco → rojo (coolwarm)
  [59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249],
  [221, 221, 221], [246, 189, 164], [244, 141, 111], [222, 81, 58], [180, 4, 38],
];
const MAGMA = [
  [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
  [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191],
];

export function colormap(v, map = "rdbu") {
  const pal = map === "magma" ? MAGMA : RDBU;
  const t = Math.max(0, Math.min(1, v)) * (pal.length - 1);
  const i = Math.min(Math.floor(t), pal.length - 2);
  const f = t - i;
  return [
    Math.round(lerp(pal[i][0], pal[i + 1][0], f)),
    Math.round(lerp(pal[i][1], pal[i + 1][1], f)),
    Math.round(lerp(pal[i][2], pal[i + 1][2], f)),
  ];
}

/* ─── Heatmap: dibuja field (Nx×Ny, index [i*Ny+j], i→x, j→y) en un canvas ─── */
export function drawHeatmap(canvas, field, Nx, Ny, opts = {}) {
  const { vmin = null, vmax = null, cmap = "rdbu", symmetric = true } = opts;
  const ctx = canvas.getContext("2d");

  // buffer del tamaño de la malla, luego se escala al canvas
  const off = document.createElement("canvas");
  off.width = Nx; off.height = Ny;
  const octx = off.getContext("2d");
  const img = octx.createImageData(Nx, Ny);

  let lo = vmin, hi = vmax;
  if (lo === null || hi === null) {
    let mx = 0, mn = Infinity;
    for (let n = 0; n < field.length; n++) {
      const a = Math.abs(field[n]);
      if (a > mx) mx = a;
      if (field[n] < mn) mn = field[n];
    }
    if (symmetric) { hi = mx || 1; lo = -hi; }
    else { lo = 0; hi = mx || 1; }
  }
  const span = hi - lo || 1;

  for (let j = 0; j < Ny; j++) {
    for (let i = 0; i < Nx; i++) {
      const v = (field[i * Ny + j] - lo) / span;
      const [r, g, b] = colormap(v, cmap);
      // fila superior del canvas = y máximo (origen abajo-izquierda)
      const p = ((Ny - 1 - j) * Nx + i) * 4;
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);

  // Ajustar resolución del canvas destino a su tamaño CSS
  const rect = canvas.getBoundingClientRect();
  if (rect.width && rect.height) {
    canvas.width = Math.round(rect.width * devicePixelRatio);
    canvas.height = Math.round(rect.height * devicePixelRatio);
  }
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // encajar preservando aspecto
  const s = Math.min(canvas.width / Nx, canvas.height / Ny);
  const w = Nx * s, h = Ny * s;
  ctx.drawImage(off, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
  return { lo, hi };
}

/* ─── LineChart ligero ─── */
export class LineChart {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = Object.assign({
      title: "", xlabel: "", ylabel: "", logY: false,
      colors: ["#4cc2ff", "#ff9d4c", "#58d68d", "#ec7063", "#b58cf5", "#f5d76e"],
      legend: true,
    }, opts);
    this.series = [];   // [{name, x: [], y: []}]
  }

  setSeries(series) { this.series = series; this.draw(); }

  draw() {
    const c = this.canvas;
    const rect = c.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    c.width = Math.round(rect.width * devicePixelRatio);
    c.height = Math.round(rect.height * devicePixelRatio);
    const ctx = c.getContext("2d");
    ctx.scale(devicePixelRatio, devicePixelRatio);
    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const padL = 52, padR = 12, padT = this.opts.title ? 24 : 12, padB = 30;
    const pw = W - padL - padR, ph = H - padT - padB;
    if (pw < 20 || ph < 20) return;

    // Rango
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const s of this.series) {
      for (let i = 0; i < s.x.length; i++) {
        let yv = s.y[i];
        if (this.opts.logY && yv <= 0) continue;
        if (this.opts.logY) yv = Math.log10(yv);
        if (s.x[i] < xmin) xmin = s.x[i];
        if (s.x[i] > xmax) xmax = s.x[i];
        if (yv < ymin) ymin = yv;
        if (yv > ymax) ymax = yv;
      }
    }
    if (!isFinite(xmin) || !isFinite(ymin)) {
      ctx.fillStyle = "#566070";
      ctx.font = "12px Inter";
      ctx.textAlign = "center";
      ctx.fillText("Sin datos — ejecuta una simulación", W / 2, H / 2);
      return;
    }
    if (xmax === xmin) xmax = xmin + 1;
    if (ymax === ymin) { ymax += 1; ymin -= 1; }
    const yr = ymax - ymin; ymin -= yr * 0.06; ymax += yr * 0.06;

    const X = (x) => padL + ((x - xmin) / (xmax - xmin)) * pw;
    const Y = (y) => padT + ph - ((y - ymin) / (ymax - ymin)) * ph;

    // Rejilla + ejes
    ctx.strokeStyle = "#20262f";
    ctx.fillStyle = "#8a93a3";
    ctx.font = "10px JetBrains Mono";
    ctx.lineWidth = 1;
    const nTicks = 5;
    for (let i = 0; i <= nTicks; i++) {
      const yv = ymin + ((ymax - ymin) * i) / nTicks;
      const ypix = Y(yv);
      ctx.beginPath(); ctx.moveTo(padL, ypix); ctx.lineTo(W - padR, ypix); ctx.stroke();
      const lbl = this.opts.logY ? `1e${yv.toFixed(1)}` : fmtNum(yv);
      ctx.textAlign = "right";
      ctx.fillText(lbl, padL - 6, ypix + 3);
    }
    for (let i = 0; i <= nTicks; i++) {
      const xv = xmin + ((xmax - xmin) * i) / nTicks;
      const xpix = X(xv);
      ctx.beginPath(); ctx.moveTo(xpix, padT); ctx.lineTo(xpix, padT + ph); ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(fmtNum(xv), xpix, H - padB + 14);
    }

    // Series
    this.series.forEach((s, si) => {
      ctx.strokeStyle = this.opts.colors[si % this.opts.colors.length];
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < s.x.length; i++) {
        let yv = s.y[i];
        if (this.opts.logY) {
          if (yv <= 0) continue;
          yv = Math.log10(yv);
        }
        const xp = X(s.x[i]), yp = Y(yv);
        if (!started) { ctx.moveTo(xp, yp); started = true; }
        else ctx.lineTo(xp, yp);
      }
      ctx.stroke();
    });

    // Título / etiquetas
    if (this.opts.title) {
      ctx.fillStyle = "#d7dde6";
      ctx.font = "600 11px Inter";
      ctx.textAlign = "left";
      ctx.fillText(this.opts.title, padL, 14);
    }
    if (this.opts.xlabel) {
      ctx.fillStyle = "#8a93a3"; ctx.font = "10px Inter"; ctx.textAlign = "center";
      ctx.fillText(this.opts.xlabel, padL + pw / 2, H - 4);
    }

    // Leyenda
    if (this.opts.legend && this.series.length > 1) {
      let lx = padL + 8;
      ctx.font = "10px Inter";
      this.series.forEach((s, si) => {
        ctx.fillStyle = this.opts.colors[si % this.opts.colors.length];
        ctx.fillRect(lx, padT + 5, 10, 3);
        ctx.fillStyle = "#8a93a3";
        ctx.textAlign = "left";
        ctx.fillText(s.name, lx + 14, padT + 10);
        lx += 14 + ctx.measureText(s.name).width + 16;
      });
    }
  }
}

function fmtNum(v) {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

/* ─── FFT (radix-2, para el espectro) ─── */
export function fftMag(signal, dt) {
  // Rellenar a potencia de 2
  let n = 1;
  while (n < signal.length) n <<= 1;
  const re = new Float64Array(n), im = new Float64Array(n);
  const mean = signal.reduce((a, b) => a + b, 0) / signal.length;
  for (let i = 0; i < signal.length; i++) re[i] = signal[i] - mean;

  // Cooley–Tukey iterativo
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cwr = 1, cwi = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cwr - im[i + k + len / 2] * cwi;
        const vi = re[i + k + len / 2] * cwi + im[i + k + len / 2] * cwr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const nwr = cwr * wr - cwi * wi;
        cwi = cwr * wi + cwi * wr; cwr = nwr;
      }
    }
  }
  const half = n >> 1;
  const freq = new Array(half), mag = new Array(half);
  for (let i = 0; i < half; i++) {
    freq[i] = i / (n * dt);
    mag[i] = Math.hypot(re[i], im[i]) / n;
  }
  return { freq, mag };
}
