/* viewport.js — escena 3D interactiva (Three.js)
 *
 * Muestra: dominio (caja transparente), malla, materiales coloreados,
 * fuente (flecha roja), PML, campo como superficie desplazada con colores,
 * cortes XY/XZ/YZ, isolíneas, ejes, medición de distancias, selección.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { state, on, select, findMaterial, frameTimes } from "./state.js";
import { colormap } from "./charts.js";
import { api } from "./api.js";

let renderer, scene, camera, controls, raycaster;
let fieldMesh, fieldGeo;              // superficie del campo (plano desplazado)
let domainBox, gridHelper, axesGroup, pmlGroup;
let objectMeshes = {};                // id → mesh (geometría/fuentes/monitores)
let isoGroup, sliceXZ, sliceYZ, sliceXY;
let measureState = { active: false, pts: [], line: null };
let RES = 61;                         // resolución de la superficie del campo
let canvasEl, containerEl, labelEl;
let needsSliceRefresh = true;

const FIELD_LABEL = {
  E: "Ez", H: "|B|", energy: "u", poynting: "|S|",
  errPINN: "|PINN−FDTD|", errFDTD: "|FDTD−exacta|",
};

export function initViewport() {
  canvasEl = document.getElementById("viewport");
  containerEl = document.getElementById("viewport-container");
  labelEl = document.getElementById("vp-measure-label");

  renderer = new THREE.WebGLRenderer({ canvas: canvasEl, antialias: true,
                                       preserveDrawingBuffer: true });
  renderer.setPixelRatio(devicePixelRatio);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c0f13);
  scene.fog = new THREE.Fog(0x0c0f13, 4, 12);

  camera = new THREE.PerspectiveCamera(46, 1, 0.01, 100);
  camera.position.set(1.7, 1.35, 2.1);

  controls = new OrbitControls(camera, canvasEl);
  controls.target.set(0.5, 0.18, 0.5);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  raycaster = new THREE.Raycaster();

  // Iluminación
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dir = new THREE.DirectionalLight(0xffffff, 1.1);
  dir.position.set(2, 3, 1.5);
  scene.add(dir);

  buildStatic();
  buildFieldSurface();

  // Interacción
  canvasEl.addEventListener("pointerdown", onPointerDown);
  window.addEventListener("resize", resize);
  document.getElementById("vp-btn-home").onclick = resetCamera;
  document.getElementById("vp-btn-measure").onclick = toggleMeasure;

  // Reactividad
  on("project", rebuildObjects);
  on("hidden", rebuildObjects);
  on("show", applyShow);
  on("field", () => { updateColorbarLabel(); refreshField(); });
  on("frame", refreshField);
  on("result", () => { needsSliceRefresh = true; refreshField(); });

  resize();
  animate();
}

function resize() {
  const r = containerEl.getBoundingClientRect();
  if (!r.width || !r.height) return;
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}

export function resizeViewport() { resize(); }

function resetCamera() {
  camera.position.set(1.7, 1.35, 2.1);
  controls.target.set(0.5, 0.18, 0.5);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

/* ═══ Escena estática: dominio, rejilla, ejes ═══ */
function buildStatic() {
  // Caja del dominio [0,1]×[alto]×[0,1] — transparente
  const H = 0.5;
  const boxGeo = new THREE.BoxGeometry(1, H, 1);
  boxGeo.translate(0.5, H / 2, 0.5);
  domainBox = new THREE.Mesh(
    boxGeo,
    new THREE.MeshBasicMaterial({ color: 0x4cc2ff, transparent: true,
                                  opacity: 0.045, depthWrite: false }));
  domainBox.userData = { pick: { kind: "domain", id: "domain" } };
  scene.add(domainBox);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(boxGeo),
    new THREE.LineBasicMaterial({ color: 0x39445a }));
  scene.add(edges);

  // Rejilla (malla FDTD visual)
  gridHelper = new THREE.GridHelper(1, 20, 0x2a3140, 0x1d232e);
  gridHelper.position.set(0.5, 0.001, 0.5);
  gridHelper.visible = false;
  scene.add(gridHelper);

  // Ejes con etiquetas
  axesGroup = new THREE.Group();
  const mkAxis = (dir, color) => {
    const a = new THREE.ArrowHelper(dir, new THREE.Vector3(0, 0, 0), 0.28, color, 0.05, 0.03);
    axesGroup.add(a);
  };
  mkAxis(new THREE.Vector3(1, 0, 0), 0xec7063);   // x
  mkAxis(new THREE.Vector3(0, 1, 0), 0x58d68d);   // z-field (vertical)
  mkAxis(new THREE.Vector3(0, 0, 1), 0x5dade2);   // y
  axesGroup.add(mkTextSprite("x", 0.33, 0.02, 0, "#ec7063"));
  axesGroup.add(mkTextSprite("Ez", 0.02, 0.33, 0, "#58d68d"));
  axesGroup.add(mkTextSprite("y", 0, 0.02, 0.33, "#5dade2"));
  scene.add(axesGroup);

  // PML
  pmlGroup = new THREE.Group();
  pmlGroup.visible = false;
  scene.add(pmlGroup);

  // Isolíneas
  isoGroup = new THREE.Group();
  isoGroup.visible = false;
  scene.add(isoGroup);

  buildSlicePlanes();
}

function mkTextSprite(text, x, y, z, color = "#d7dde6", size = 90) {
  const cnv = document.createElement("canvas");
  cnv.width = 128; cnv.height = 64;
  const ctx = cnv.getContext("2d");
  ctx.font = `600 ${size * 0.42}px JetBrains Mono`;
  ctx.fillStyle = color;
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, 64, 32);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(cnv), transparent: true, depthTest: false }));
  sp.scale.set(0.14, 0.07, 1);
  sp.position.set(x, y, z);
  return sp;
}

/* ═══ Superficie del campo ═══ */
function buildFieldSurface() {
  fieldGeo = new THREE.PlaneGeometry(1, 1, RES - 1, RES - 1);
  fieldGeo.rotateX(-Math.PI / 2);          // plano XZ de three = plano XY físico
  fieldGeo.translate(0.5, 0.02, 0.5);
  const colors = new Float32Array(fieldGeo.attributes.position.count * 3);
  fieldGeo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  fieldMesh = new THREE.Mesh(fieldGeo, new THREE.MeshStandardMaterial({
    vertexColors: true, side: THREE.DoubleSide,
    roughness: 0.65, metalness: 0.05,
  }));
  scene.add(fieldMesh);
}

function buildSlicePlanes() {
  // Cortes XZ / YZ: texturas espacio-tiempo del PINN, paredes verticales
  const mk = () => {
    const tex = new THREE.DataTexture(new Uint8Array(4), 1, 1);
    tex.needsUpdate = true;
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 0.5),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.92,
                                    side: THREE.DoubleSide }));
    m.visible = false;
    scene.add(m);
    return m;
  };
  sliceXZ = mk();
  sliceXZ.position.set(0.5, 0.25, 0.5);
  sliceYZ = mk();
  sliceYZ.rotation.y = Math.PI / 2;
  sliceYZ.position.set(0.5, 0.25, 0.5);
}

/* ═══ Objetos del proyecto (materiales, fuentes, monitores, PML) ═══ */
function rebuildObjects() {
  for (const id in objectMeshes) {
    scene.remove(objectMeshes[id]);
    objectMeshes[id].traverse?.((o) => { o.geometry?.dispose(); });
  }
  objectMeshes = {};
  const p = state.project;
  if (!p) return;
  const H = 0.5;

  // Geometría con material
  for (const g of p.geometry ?? []) {
    if (g.shape === "domain") continue;
    const mat = findMaterial(g.material) ?? {};
    const color = new THREE.Color(mat.color || "#888888");
    const enabled = g.enabled !== false;
    const visible = !state.hidden.has(g.id);
    let mesh;
    if (g.shape === "sphere" || g.shape === "circle") {
      mesh = new THREE.Mesh(
        new THREE.SphereGeometry(g.r ?? 0.1, 28, 20),
        new THREE.MeshStandardMaterial({ color, transparent: true,
          opacity: enabled ? 0.85 : 0.18, roughness: 0.4 }));
      mesh.position.set(g.cx ?? 0.5, Math.min(g.r ?? 0.1, H / 2) + 0.02, g.cy ?? 0.5);
    } else if (g.shape === "box") {
      const w = (g.x1 ?? 1) - (g.x0 ?? 0), d = (g.y1 ?? 1) - (g.y0 ?? 0);
      mesh = new THREE.Mesh(
        new THREE.BoxGeometry(Math.max(w, 1e-3), 0.3, Math.max(d, 1e-3)),
        new THREE.MeshStandardMaterial({ color, transparent: true,
          opacity: enabled ? 0.85 : 0.18, roughness: 0.4 }));
      mesh.position.set(((g.x0 ?? 0) + (g.x1 ?? 1)) / 2, 0.15 + 0.02,
                        ((g.y0 ?? 0) + (g.y1 ?? 1)) / 2);
    }
    if (mesh) {
      mesh.visible = visible;
      mesh.userData = { pick: { kind: "geometry", id: g.id } };
      scene.add(mesh);
      objectMeshes[g.id] = mesh;
    }
  }

  // Fuentes → flecha roja
  for (const s of p.sources ?? []) {
    const grp = new THREE.Group();
    const isMode = s.type === "mode";
    const x = isMode ? 0.5 : (s.x ?? 0.5);
    const y = isMode ? 0.5 : (s.y ?? 0.5);
    const arrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, -1, 0), new THREE.Vector3(x, 0.62, y),
      0.22, 0xff4444, 0.07, 0.045);
    grp.add(arrow);
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.045, 0.006, 8, 24),
      new THREE.MeshBasicMaterial({ color: 0xff4444 }));
    ring.rotation.x = Math.PI / 2;
    ring.position.set(x, 0.03, y);
    grp.add(ring);
    grp.visible = (s.enabled !== false) && !state.hidden.has(s.id);
    // hit-proxy invisible para selección
    const proxy = new THREE.Mesh(
      new THREE.SphereGeometry(0.06, 8, 8),
      new THREE.MeshBasicMaterial({ visible: false }));
    proxy.position.set(x, 0.45, y);
    proxy.userData = { pick: { kind: "source", id: s.id } };
    grp.add(proxy);
    scene.add(grp);
    objectMeshes[s.id] = grp;
  }

  // Monitores → marcos delgados
  for (const m of p.monitors ?? []) {
    if (m.plane === "point") {
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.014, 12, 10),
        new THREE.MeshBasicMaterial({ color: 0xf5d76e }));
      dot.position.set(m.x ?? 0.5, 0.04, m.y ?? 0.5);
      dot.userData = { pick: { kind: "monitor", id: m.id } };
      dot.visible = (m.enabled !== false) && !state.hidden.has(m.id);
      scene.add(dot);
      objectMeshes[m.id] = dot;
      continue;
    }
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(1, m.plane === "xy" ? 1 : 0.5)),
      new THREE.LineBasicMaterial({ color: 0xf5d76e, transparent: true, opacity: 0.6 }));
    if (m.plane === "xy") {
      frame.rotation.x = -Math.PI / 2;
      frame.position.set(0.5, 0.02 + 0.002, 0.5);
    } else if (m.plane === "xz") {
      frame.position.set(0.5, 0.25, m.position ?? 0.5);
    } else {
      frame.rotation.y = Math.PI / 2;
      frame.position.set(m.position ?? 0.5, 0.25, 0.5);
    }
    frame.visible = (m.enabled !== false) && !state.hidden.has(m.id);
    frame.userData = { pick: { kind: "monitor", id: m.id } };
    scene.add(frame);
    objectMeshes[m.id] = frame;
  }

  // PML
  pmlGroup.clear();
  if (p.pml?.enabled) {
    const cells = p.pml.cells ?? 8;
    const th = cells / ((p.domain?.Nx ?? 101) - 1);
    const mat = new THREE.MeshBasicMaterial({ color: 0xb58cf5, transparent: true,
                                              opacity: 0.10, depthWrite: false });
    const mk = (w, d, x, z) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, 0.5, d), mat);
      m.position.set(x, 0.25, z);
      pmlGroup.add(m);
    };
    mk(1, th, 0.5, th / 2);
    mk(1, th, 0.5, 1 - th / 2);
    mk(th, 1 - 2 * th, th / 2, 0.5);
    mk(th, 1 - 2 * th, 1 - th / 2, 0.5);
    pmlGroup.visible = true;
  } else {
    pmlGroup.visible = false;
  }

  applyShow();
}

function applyShow() {
  const s = state.show;
  gridHelper.visible = s.mesh;
  axesGroup.visible = s.axes;
  fieldMesh.visible = s.sliceXY;
  isoGroup.visible = s.iso;
  if (sliceXZ) sliceXZ.visible = s.sliceXZ;
  if (sliceYZ) sliceYZ.visible = s.sliceYZ;
  if (s.sliceXZ || s.sliceYZ) refreshSpaceTimeSlices();
}

/* ═══ Datos del campo ═══ */
function getFrame(name) {
  const f = state.frames[name];
  if (!f) return null;
  const k = Math.min(state.frame, f.K - 1);
  return { arr: f.data.subarray(k * f.Nx * f.Ny, (k + 1) * f.Nx * f.Ny),
           Nx: f.Nx, Ny: f.Ny };
}

function analyticalEz(i, j, Nx, Ny, t) {
  const x = i / (Nx - 1), y = j / (Ny - 1);
  const w = Math.PI * Math.SQRT2;
  return Math.sin(Math.PI * x) * Math.sin(Math.PI * y) * Math.cos(w * t);
}

/* Calcula el campo escalar a mostrar según state.field.
   Retorna {arr, Nx, Ny, symmetric} o null si no hay datos. */
export function computeDisplayField() {
  const fld = state.field;
  const pref = state.frames.Ez_fdtd ? "fdtd" : (state.frames.Ez_pinn ? "pinn" : null);
  const t = frameTimes()[state.frame] ?? 0;

  const Ez = getFrame(pref === "fdtd" ? "Ez_fdtd" : "Ez_pinn");
  const Bx = getFrame(pref === "fdtd" ? "Bx_fdtd" : "Bx_pinn");
  const By = getFrame(pref === "fdtd" ? "By_fdtd" : "By_pinn");

  if (fld === "errPINN") {
    const e = getFrame("err_abs");
    return e ? { ...e, symmetric: false, cmap: "magma" } : null;
  }
  if (fld === "errFDTD") {
    const f = getFrame("Ez_fdtd");
    if (!f) return null;
    const out = new Float32Array(f.arr.length);
    for (let i = 0; i < f.Nx; i++)
      for (let j = 0; j < f.Ny; j++)
        out[i * f.Ny + j] = Math.abs(f.arr[i * f.Ny + j] -
                                     analyticalEz(i, j, f.Nx, f.Ny, t));
    return { arr: out, Nx: f.Nx, Ny: f.Ny, symmetric: false, cmap: "magma" };
  }
  if (!Ez) return null;
  if (fld === "E") return { ...Ez, symmetric: true, cmap: "rdbu" };
  if (!Bx || !By) return { ...Ez, symmetric: true, cmap: "rdbu" };

  const out = new Float32Array(Ez.arr.length);
  if (fld === "H") {
    for (let n = 0; n < out.length; n++)
      out[n] = Math.hypot(Bx.arr[n], By.arr[n]);
  } else if (fld === "energy") {
    for (let n = 0; n < out.length; n++)
      out[n] = 0.5 * (Ez.arr[n] ** 2 + Bx.arr[n] ** 2 + By.arr[n] ** 2);
  } else if (fld === "poynting") {
    // TM: S = (−Ez·By, Ez·Bx) → |S| = |Ez|·|B|
    for (let n = 0; n < out.length; n++)
      out[n] = Math.abs(Ez.arr[n]) * Math.hypot(Bx.arr[n], By.arr[n]);
  }
  return { arr: out, Nx: Ez.Nx, Ny: Ez.Ny, symmetric: false, cmap: "magma" };
}

/* Muestreo bilineal del campo en (x, y) ∈ [0,1]² */
function sampleField(f, x, y) {
  const fx = x * (f.Nx - 1), fy = y * (f.Ny - 1);
  const i0 = Math.min(Math.floor(fx), f.Nx - 2);
  const j0 = Math.min(Math.floor(fy), f.Ny - 2);
  const dx = fx - i0, dy = fy - j0;
  const v = (i, j) => f.arr[i * f.Ny + j];
  return (v(i0, j0) * (1 - dx) * (1 - dy) + v(i0 + 1, j0) * dx * (1 - dy) +
          v(i0, j0 + 1) * (1 - dx) * dy + v(i0 + 1, j0 + 1) * dx * dy);
}

/* ═══ Refresco de la superficie del campo + colorbar + isolíneas ═══ */
let pinnLiveTimer = null;

export function refreshField() {
  const f = computeDisplayField();
  const srcLabel = document.getElementById("vp-source-label");
  if (!f) {
    // Sin resultados: modo PINN en vivo (throttle)
    srcLabel.textContent = "PINN (en vivo)";
    if (!pinnLiveTimer) {
      pinnLiveTimer = setTimeout(async () => {
        pinnLiveTimer = null;
        try {
          const t = frameTimes()[state.frame] ?? 0;
          const r = await api.slice({ axis: "xy", t, N: RES });
          const arr = new Float32Array(RES * RES);
          for (let i = 0; i < RES; i++)
            for (let j = 0; j < RES; j++)
              arr[i * RES + j] = r.Ez[j][i];       // Ez[row=j][col=i]
          paintSurface({ arr, Nx: RES, Ny: RES, symmetric: true, cmap: "rdbu" });
        } catch { /* backend sin modelo */ }
      }, 60);
    }
    return;
  }
  srcLabel.textContent = state.resultMeta?.mode === "pinn" ? "Resultado: PINN"
    : state.resultMeta?.mode === "fdtd" ? "Resultado: FDTD"
    : "Resultado: PINN + FDTD";
  paintSurface(f);
}

function paintSurface(f) {
  const pos = fieldGeo.attributes.position;
  const col = fieldGeo.attributes.color;

  let vmax = 0;
  for (let n = 0; n < f.arr.length; n++) {
    const a = Math.abs(f.arr[n]);
    if (a > vmax) vmax = a;
  }
  vmax = vmax || 1;
  const lo = f.symmetric ? -vmax : 0;
  const span = f.symmetric ? 2 * vmax : vmax;
  const zScale = 0.34 / vmax;

  // PlaneGeometry (rotada): vértices en fila-major sobre (x, z=y físico)
  const n1 = RES;
  for (let r = 0; r < n1; r++) {
    for (let c = 0; c < n1; c++) {
      const idx = r * n1 + c;
      const x = pos.getX(idx);          // 0..1
      const yPhys = pos.getZ(idx);      // 0..1
      const v = sampleField(f, x, yPhys);
      pos.setY(idx, 0.02 + (f.symmetric ? (v + vmax) / 2 : v) * zScale *
                    (f.symmetric ? 1 : 1.4));
      const tNorm = (v - lo) / span;
      const [cr, cg, cb] = colormap(tNorm, f.cmap);
      col.setXYZ(idx, cr / 255, cg / 255, cb / 255);
    }
  }
  pos.needsUpdate = true;
  col.needsUpdate = true;
  fieldGeo.computeVertexNormals();

  // Colorbar
  document.getElementById("cb-max").textContent = fmtCb(f.symmetric ? vmax : vmax);
  document.getElementById("cb-min").textContent = fmtCb(f.symmetric ? -vmax : 0);
  const grad = document.getElementById("cb-grad");
  grad.style.background = f.cmap === "magma"
    ? "linear-gradient(to top, #000004, #651573, #d44842, #fac127, #fcfdbf)"
    : "linear-gradient(to top, #3b4cc0, #dddddd, #b40426)";

  if (state.show.iso) rebuildIsolines(f, lo, span, vmax);
}

function fmtCb(v) {
  return (v >= 0 ? "+" : "−") + Math.abs(v).toPrecision(3);
}

function updateColorbarLabel() {
  document.getElementById("cb-field").textContent = FIELD_LABEL[state.field] ?? state.field;
}

/* ═══ Isolíneas (marching squares elevadas) ═══ */
function rebuildIsolines(f, lo, span, vmax) {
  isoGroup.clear();
  const levels = 7;
  const zScale = 0.34 / vmax;
  const pts = [];
  const N = 48;
  for (let li = 1; li < levels; li++) {
    const lvl = lo + (span * li) / levels;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const x0 = i / N, x1 = (i + 1) / N, y0 = j / N, y1 = (j + 1) / N;
        const v00 = sampleField(f, x0, y0), v10 = sampleField(f, x1, y0);
        const v01 = sampleField(f, x0, y1), v11 = sampleField(f, x1, y1);
        const edges = [];
        const check = (va, vb, ax, ay, bx, by) => {
          if ((va - lvl) * (vb - lvl) < 0) {
            const t = (lvl - va) / (vb - va);
            edges.push([ax + (bx - ax) * t, ay + (by - ay) * t]);
          }
        };
        check(v00, v10, x0, y0, x1, y0);
        check(v10, v11, x1, y0, x1, y1);
        check(v11, v01, x1, y1, x0, y1);
        check(v01, v00, x0, y1, x0, y0);
        if (edges.length >= 2) {
          const h = 0.025 + (f.symmetric ? (lvl + vmax) / 2 : lvl) * zScale *
                    (f.symmetric ? 1 : 1.4);
          pts.push(edges[0][0], h, edges[0][1], edges[1][0], h, edges[1][1]);
        }
      }
    }
  }
  if (pts.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    isoGroup.add(new THREE.LineSegments(
      g, new THREE.LineBasicMaterial({ color: 0xd7dde6, transparent: true, opacity: 0.55 })));
  }
}

/* ═══ Cortes espacio-tiempo XZ / YZ (texturas del PINN) ═══ */
async function refreshSpaceTimeSlices() {
  if (!needsSliceRefresh) return;
  needsSliceRefresh = false;
  for (const [mesh, axis] of [[sliceXZ, "xz"], [sliceYZ, "yz"]]) {
    try {
      const N = 64;
      const r = await api.slice({ axis, position: 0.5, N });
      const data = new Uint8Array(N * N * 4);
      let vmax = 0;
      for (const row of r.Ez) for (const v of row) vmax = Math.max(vmax, Math.abs(v));
      vmax = vmax || 1;
      for (let j = 0; j < N; j++)
        for (let i = 0; i < N; i++) {
          const v = (r.Ez[j][i] / vmax + 1) / 2;
          const [cr, cg, cb] = colormap(v, "rdbu");
          const p = (j * N + i) * 4;
          data[p] = cr; data[p + 1] = cg; data[p + 2] = cb; data[p + 3] = 235;
        }
      const tex = new THREE.DataTexture(data, N, N);
      tex.needsUpdate = true;
      mesh.material.map?.dispose();
      mesh.material.map = tex;
      mesh.material.needsUpdate = true;
    } catch { /* sin modelo */ }
  }
}

/* ═══ Selección y medición ═══ */
function onPointerDown(ev) {
  if (ev.button !== 0) return;
  const rect = canvasEl.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -((ev.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);

  if (measureState.active) {
    const hit = raycaster.intersectObject(domainBox, false)[0] ??
                raycaster.intersectObject(fieldMesh, false)[0];
    if (hit) addMeasurePoint(hit.point, ev);
    return;
  }

  const targets = [];
  for (const id in objectMeshes) {
    objectMeshes[id].traverse((o) => { if (o.userData?.pick || o.isMesh) targets.push(o); });
  }
  targets.push(domainBox);
  const hits = raycaster.intersectObjects(targets, false);
  for (const h of hits) {
    let o = h.object;
    while (o && !o.userData?.pick) o = o.parent;
    if (o?.userData?.pick) {
      select(o.userData.pick.kind, o.userData.pick.id);
      return;
    }
  }
}

function toggleMeasure() {
  measureState.active = !measureState.active;
  measureState.pts = [];
  document.getElementById("vp-btn-measure").classList.toggle("active", measureState.active);
  controls.enabled = true;
  if (!measureState.active) {
    labelEl.hidden = true;
    if (measureState.line) { scene.remove(measureState.line); measureState.line = null; }
  }
}

function addMeasurePoint(pt, ev) {
  measureState.pts.push(pt.clone());
  if (measureState.pts.length === 2) {
    const [a, b] = measureState.pts;
    if (measureState.line) scene.remove(measureState.line);
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    measureState.line = new THREE.Line(
      g, new THREE.LineBasicMaterial({ color: 0x4cc2ff, depthTest: false }));
    scene.add(measureState.line);
    // distancia física en el plano XY (x, z de three)
    const d = Math.hypot(b.x - a.x, b.z - a.z);
    labelEl.textContent = `d = ${d.toFixed(4)} (unid. de L)`;
    labelEl.style.left = `${ev.clientX - canvasEl.getBoundingClientRect().left + 12}px`;
    labelEl.style.top = `${ev.clientY - canvasEl.getBoundingClientRect().top - 8}px`;
    labelEl.hidden = false;
    measureState.pts = [];
  } else {
    labelEl.textContent = "Clic en el segundo punto…";
    labelEl.style.left = `${ev.clientX - canvasEl.getBoundingClientRect().left + 12}px`;
    labelEl.style.top = `${ev.clientY - canvasEl.getBoundingClientRect().top - 8}px`;
    labelEl.hidden = false;
  }
}

/* ═══ Exportar imagen del viewport ═══ */
export function viewportPNG() {
  return canvasEl.toDataURL("image/png");
}
export function viewportCanvas() { return canvasEl; }
