"use strict";
/* tools/design_gui/static/app.js —— 页面状态机与 UI 绑定。 */

const $ = (id) => document.getElementById(id);
const LOCAL_PREFIX = "design_gui_v1:";   // 每个电机一份草稿
const LOCAL_LAST = "design_gui_last_motor";
//: 这些预设走 ``run_auto_schedule.py --ladder <name>``，而不是 --stages 文本。
const LADDER_PRESETS = ["night", "stress", "rip"];

const state = {
  motor: "6s4p",
  geo: null,
  editor: null,
  renderer: null,
  view: null,
  needsFit: true,

  radiusMm: 2.0,
  identityMm: 0.2,
  maxDensity: 0.65,
  tool: "brush",
  brushMm: 1.5,
  value: 1.0,

  previewEnabled: false,
  previewRho: null,
  previewStats: null,
  lost: null,

  editSlaves: true,
  showSlaves: true,
  xPreimage: false,

  pointer: null,          // [x,y] world mm (hover)
  hover: -1,
  dragging: null,
  panning: null,
  space: false,
  polygon: null,          // {pts:[[x,y],...], cursor:[x,y]|null}
  shape: null,

  importedX: null,
  importedRadiusMm: null,
  edited: false,

  importCtx: null,
  loadGen: 0,
  jobOffset: 0,
  jobTimer: null,
  previewTimer: null,
  autosaveTimer: null,
  raf: 0,
  dirty: {base: false, fg: false, ov: false},
};

/* ------------------------------------------------------------------ 事件 */
document.addEventListener("DOMContentLoaded", async () => {
  bindControls();
  try {
    await refreshState();
    const last = localStorage.getItem(LOCAL_LAST);
    const motor = (last && $("motor").querySelector(`option[value="${last}"]`))
      ? last : ($("motor").value || "6s4p");
    $("motor").value = motor;
    await loadMotor(motor, loadLocal(motor));
    refreshDesigns();
    maybeResumeJob();
  } catch (e) {
    console.error(e);
    showHint("初始化失败：" + e.message, true);
  }
});

function showHint(text, bad = false) {
  const el = $("canvas-hint");
  if (!text) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  el.textContent = text;
  el.style.color = bad ? "var(--bad)" : "var(--muted)";
}

async function refreshState() {
  if ($("motor").options.length === 0) {
    $("motor").innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "6s4p"; opt.textContent = "6s4p";
    $("motor").appendChild(opt);
  }
  const st = await apiGet("/api/state");
  if (st.motors) {
    const cur = $("motor").value;
    $("motor").innerHTML = "";
    for (const [k, label] of Object.entries(st.motors)) {
      const o = document.createElement("option");
      o.value = k; o.textContent = label;
      $("motor").appendChild(o);
    }
    if (cur) $("motor").value = cur;
  }
  if (st.job) updateJobChip(st.job);
  updateBuildChip(st.build);
  return st;
}

/* -------------------------------------------------------------- 加载几何 */
async function loadMotor(motor, saved) {
  state.loadGen += 1;
  clearTimeout(state.previewTimer);
  state.motor = motor;
  state.editor = null;
  state.geo = null;
  state.previewRho = null;
  state.previewStats = null;
  state.lost = null;
  state.dragging = null;
  state.polygon = null;
  state.shape = null;
  state.hover = -1;
  state.importedX = null;
  state.importedRadiusMm = null;
  state.edited = false;
  showHint("几何构建中，请稍候…（约 8 s）");
  setBuildChip("几何构建中…", "busy");

  let payload = null;
  for (let i = 0; i < 300; i++) {
    let r;
    try {
      r = await apiGet(`/api/geometry?motor=${encodeURIComponent(motor)}`);
    } catch (e) {
      setBuildChip("几何失败", "bad");
      showHint("几何构建失败：" + e.message, true);
      throw e;
    }
    if (r.geometry) { payload = r.geometry; break; }
    if (r.n_master) { payload = r; break; }   // 兼容“直接返回 geometry”的旧服务
    if (r.building === false) {
      setBuildChip("几何失败", "bad");
      showHint("几何构建失败：" + (r.error || "未知错误"), true);
      throw new Error("geometry build failed");
    }
    await sleep(500);
  }
  if (!payload) throw new Error("几何构建超时");

  const geo = new Geometry(payload);
  state.geo = geo;
  state.editor = new Editor(geo);
  state.identityMm = geo.identityRadiusMm;
  if (!state.view) state.view = new View();
  state.needsFit = true;

  const canvases = {bg: $("bg"), fg: $("fg"), ov: $("ov")};
  if (!state.renderer) {
    state.renderer = new Renderer(canvases, geo, state.view);
  } else {
    state.renderer.setGeometry(geo);
    state.renderer.view = state.view;
  }
  fitView();
  setBuildChip("几何就绪", "good");
  showHint("");

  // 恢复本地草稿
  if (saved && saved.sha === geo.sha && saved.motor === motor &&
      Array.isArray(saved.rho) && saved.rho.length === geo.nMaster) {
    state.editor.rho = Float32Array.from(saved.rho);
    state.radiusMm = Number(saved.radiusMm) || geo.defaultRadiusMm;
    state.tool = saved.tool || "brush";
    state.brushMm = Number(saved.brushMm) || 1.5;
    state.value = Number(saved.value) || 1;
    state.editSlaves = saved.editSlaves !== false;
    state.showSlaves = saved.showSlaves !== false;
    state.previewEnabled = !!saved.previewEnabled;
    state.xPreimage = !!saved.xPreimage;
    state.maxDensity = Number(saved.maxDensity) || 0.65;
  } else {
    state.radiusMm = geo.defaultRadiusMm;
    state.maxDensity = Number(geo.p.cfg_hint ? geo.p.cfg_hint.max_density : 0.65) || 0.65;
  }
  syncAllControls();
  redrawAll(true);
  updateStats();
  if (state.previewEnabled) schedulePreview(0);
}

function fitView() {
  if (!state.geo || !state.renderer) return;
  const [w, h] = state.renderer.resize();
  state.view.w = w; state.view.h = h;
  state.view.fit(state.geo.bbox, w, h);
  state.needsFit = false;
  requestDraw({base: true, fg: true, ov: true});
  $("zoom-info").textContent = `缩放 ${state.view.scale.toFixed(2)} px/mm`;
}

function focusMaster() {
  if (!state.geo) return;
  const g = state.geo;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < g.nMaster; i++) {
    const x = g.cMaster[2 * i], y = g.cMaster[2 * i + 1];
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  const pad = 3.0;
  state.view.fitRange(x0 - pad, y0 - pad, x1 + pad, y1 + pad);
  requestDraw({base: true, fg: true, ov: true});
  $("zoom-info").textContent = `缩放 ${state.view.scale.toFixed(2)} px/mm`;
}

/* -------------------------------------------------------------- 绘制调度 */
function requestDraw(flags) {
  const d = state.dirty;
  if (flags.base) d.base = true;
  if (flags.fg) d.fg = true;
  if (flags.ov) d.ov = true;
  if (state.raf) return;
  state.raf = requestAnimationFrame(() => {
    state.raf = 0;
    const f = state.dirty;
    state.dirty = {base: false, fg: false, ov: false};
    if (f.base && state.renderer) state.renderer.drawBase();
    if (f.fg) drawDesign();
    if (f.ov) drawOverlay();
  });
}

function activeRho() {
  if (!state.editor) return null;
  if (state.previewEnabled && state.previewRho) return state.previewRho;
  return state.editor.rho;
}

function drawDesign() {
  if (!state.renderer || !state.editor) return;
  const rho = activeRho();
  if (!rho) return;
  state.renderer.drawDesign(rho, {showSlaves: state.showSlaves});
}

function drawOverlay() {
  if (!state.renderer) return;
  const brush = (state.pointer && (state.tool === "brush" || state.tool === "eraser"))
    ? {x: state.pointer[0], y: state.pointer[1], r: state.brushMm} : null;
  const poly = state.polygon
    ? state.polygon.pts.concat(state.polygon.cursor ? [state.polygon.cursor] : [])
    : null;
  state.renderer.drawOverlay({
    hover: state.hover,
    brush,
    shape: state.shape,
    polygon: poly,
    lost: state.previewEnabled ? state.lost : null,
  });
}

function redrawAll(base = false) {
  requestDraw({base, fg: true, ov: true});
}

/* -------------------------------------------------------------- 画布交互 */
function eventWorld(e) {
  const r = $("ov").getBoundingClientRect();
  return state.view.toWorld(e.clientX - r.left, e.clientY - r.top);
}

function currentValue(e) {
  let v = state.value;
  if (e && e.button === 2) v = 1 - v;
  return Math.max(0, Math.min(1, v));
}

function bindControls() {
  // 工具
  for (const b of document.querySelectorAll("#tools .tool")) {
    b.addEventListener("click", () => setTool(b.dataset.tool));
  }
  $("quick-solid").addEventListener("click", () => setValue(1));
  $("quick-air").addEventListener("click", () => setValue(0));
  $("undo").addEventListener("click", doUndo);
  $("redo").addEventListener("click", doRedo);
  $("fill-all").addEventListener("click", () => {
    if (!state.editor) return;
    state.editor.setAll(1); afterEdit();
  });
  $("clear-all").addEventListener("click", () => {
    if (!state.editor) return;
    state.editor.setAll(0); afterEdit();
  });
  $("fit-view").addEventListener("click", fitView);
  $("focus-master").addEventListener("click", focusMaster);

  // 笔刷 / 密度
  $("brush-radius").addEventListener("input", (e) => setBrush(Number(e.target.value)));
  $("brush-radius-num").addEventListener("input", (e) => setBrush(Number(e.target.value)));
  $("paint-value").addEventListener("input", (e) => setValue(Number(e.target.value)));
  $("paint-value-num").addEventListener("input", (e) => setValue(Number(e.target.value)));

  // 滤波
  $("filter-radius").addEventListener("input", (e) => setRadius(Number(e.target.value)));
  $("filter-radius-num").addEventListener("input", (e) => setRadius(Number(e.target.value)));
  for (const b of document.querySelectorAll("#radius-presets button")) {
    b.addEventListener("click", () => {
      if (b.dataset.r === "ident") setRadius(state.identityMm);
      else setRadius(Number(b.dataset.r));
    });
  }
  $("preview-filter").addEventListener("change", (e) => {
    state.previewEnabled = e.target.checked;
    if (state.previewEnabled) schedulePreview(0);
    else { state.previewRho = null; state.previewStats = null; state.lost = null; }
    redrawAll(); updateStats(); autosave();
  });
  $("edit-slaves").addEventListener("change", (e) => {
    state.editSlaves = e.target.checked; autosave();
  });
  $("show-slaves").addEventListener("change", (e) => {
    state.showSlaves = e.target.checked; redrawAll(); autosave();
  });
  $("x-preimage").addEventListener("change", (e) => {
    state.xPreimage = e.target.checked;
    if (state.previewEnabled) schedulePreview(0);
    autosave();
  });

  // 保存 / 导入
  $("save").addEventListener("click", saveDesign);
  $("refresh-designs").addEventListener("click", refreshDesigns);
  $("load-design").addEventListener("click", loadSelectedDesign);
  $("import-browse").addEventListener("click", browseImport);
  $("import-apply").addEventListener("click", applyImport);
  $("import-cancel").addEventListener("click", closeImport);
  $("import-key").addEventListener("change", updateImportNote);

  // 优化
  $("stage-preset").addEventListener("change", (e) => {
    const v = e.target.value;
    if (LADDER_PRESETS.includes(v)) $("stages").value = "";
    else if (v) $("stages").value = v;
  });
  $("start-opt").addEventListener("click", startOptimization);
  $("stop-opt").addEventListener("click", stopOptimization);

  // 网格切换（每个电机一份本地草稿）
  $("motor").addEventListener("change", async (e) => {
    await loadMotor(e.target.value, loadLocal(e.target.value));
    refreshDesigns();
  });

  // 画布
  const ov = $("ov");
  ov.addEventListener("pointerdown", onPointerDown);
  ov.addEventListener("pointermove", onPointerMove);
  ov.addEventListener("pointerup", onPointerUp);
  ov.addEventListener("pointercancel", onPointerUp);
  ov.addEventListener("pointerleave", () => {
    state.pointer = null; state.hover = -1;
    requestDraw({ov: true});
  });
  ov.addEventListener("wheel", onWheel, {passive: false});
  ov.addEventListener("contextmenu", (e) => e.preventDefault());
  ov.addEventListener("dblclick", onDoubleClick);
  window.addEventListener("resize", onResize);

  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("keyup", onKeyUp);
}

function onResize() {
  if (!state.renderer) return;
  const [w, h] = state.renderer.resize();
  state.view.w = w; state.view.h = h;
  if (state.needsFit) fitView();
  else requestDraw({base: true, fg: true, ov: true});
}

function onWheel(e) {
  e.preventDefault();
  if (!state.geo) return;
  const r = $("ov").getBoundingClientRect();
  const px = e.clientX - r.left, py = e.clientY - r.top;
  state.view.zoomAt(px, py, Math.exp(-e.deltaY * 0.0015));
  requestDraw({base: true, fg: true, ov: true});
  $("zoom-info").textContent = `缩放 ${state.view.scale.toFixed(2)} px/mm`;
}

function onPointerDown(e) {
  if (!state.geo) return;
  try { $("ov").setPointerCapture(e.pointerId); } catch (_) { /* 合成事件无捕获 */ }
  if (e.button === 1 || state.space) {
    state.panning = {x: e.clientX, y: e.clientY};
    return;
  }
  const p = eventWorld(e);
  if (state.tool === "polygon") {
    if (!state.polygon) state.polygon = {pts: [], cursor: p};
    state.polygon.pts.push(p);
    state.polygon.cursor = p;
    requestDraw({ov: true});
    return;
  }
  const value = currentValue(e);
  if (state.tool === "brush" || state.tool === "eraser") {
    const v = state.tool === "eraser" ? (e.button === 2 ? 1 : 0) : value;
    state.editor.snapshot();
    const changed = state.editor.applyBrush(p[0], p[1], state.brushMm, v,
      {slaves: state.editSlaves});
    state.dragging = {mode: "brush", last: p, value: v, changed};
    requestDraw({fg: true, ov: true});
    updateStats();
  } else {
    state.dragging = {mode: state.tool, a: p, b: p, value};
    state.shape = shapeOf(state.dragging);
    requestDraw({ov: true});
  }
}

function onPointerMove(e) {
  if (!state.geo) return;
  const r = $("ov").getBoundingClientRect();
  if (state.panning) {
    const dx = e.clientX - state.panning.x;
    const dy = e.clientY - state.panning.y;
    state.panning = {x: e.clientX, y: e.clientY};
    state.view.panBy(dx, dy);
    requestDraw({base: true, fg: true, ov: true});
    return;
  }
  const p = eventWorld(e);
  state.pointer = p;
  const k = state.geo.hitTest(p[0], p[1]);
  state.hover = k;
  updateCursorInfo(p, k);

  if (state.polygon) {
    state.polygon.cursor = p;
  }
  if (state.dragging) {
    if (state.dragging.mode === "brush") {
      paintStrokeTo(p, state.dragging.value);
    } else {
      state.dragging.b = p;
      state.shape = shapeOf(state.dragging);
    }
  }
  requestDraw({fg: false, ov: true});
}

function onPointerUp(e) {
  if (state.panning) { state.panning = null; return; }
  if (!state.dragging) return;
  const d = state.dragging;
  state.dragging = null;
  state.shape = null;
  if (d.mode === "brush") {
    if (d.changed) afterEdit();
    else {
      // 空操作：把按下时的快照弹掉，避免污染撤销栈、也不清 importedX
      if (state.editor.undoStack.length) state.editor.undoStack.pop();
      requestDraw({ov: true});
    }
    return;
  }
  if (!state.editor) return;
  state.editor.snapshot();
  const opts = {slaves: state.editSlaves};
  let changed = 0;
  if (d.mode === "line") {
    changed = state.editor.applySegment(d.a[0], d.a[1], d.b[0], d.b[1],
      state.brushMm, d.value, opts);
  } else if (d.mode === "rect") {
    changed = state.editor.applyRect(d.a[0], d.a[1], d.b[0], d.b[1], d.value, opts);
  } else if (d.mode === "circle") {
    const rr = Math.hypot(d.b[0] - d.a[0], d.b[1] - d.a[1]);
    changed = state.editor.applyCircle(d.a[0], d.a[1], rr, d.value, opts);
  } else if (d.mode === "ring") {
    changed = state.editor.applyRing(d.a[0], d.a[1], d.b[0], d.b[1], 45,
      d.value, opts);
  }
  if (changed) afterEdit();
  else {
    if (state.editor.undoStack.length) state.editor.undoStack.pop();
    requestDraw({ov: true});
  }
}

function onDoubleClick(e) {
  if (state.tool === "polygon" && state.polygon && state.polygon.pts.length >= 3) {
    finishPolygon();
  } else {
    fitView();
  }
}

function shapeOf(d) {
  if (d.mode === "line") return {type: "line", a: d.a, b: d.b, r: state.brushMm};
  if (d.mode === "rect") return {type: "rect", a: d.a, b: d.b};
  if (d.mode === "circle") return {type: "circle", a: d.a, b: d.b};
  if (d.mode === "ring") return {type: "ring", a: d.a, b: d.b};
  return null;
}

function paintStrokeTo(p, value) {
  const d = state.dragging;
  if (!d) return;
  const last = d.last;
  const dist = Math.hypot(p[0] - last[0], p[1] - last[1]);
  const step = Math.min(state.brushMm / 2, 0.2);
  const n = Math.max(1, Math.ceil(dist / Math.max(step, 1e-6)));
  for (let i = 1; i <= n; i++) {
    const t = i / n;
    const x = last[0] + (p[0] - last[0]) * t;
    const y = last[1] + (p[1] - last[1]) * t;
    d.changed += state.editor.applyBrush(x, y, state.brushMm, value,
      {slaves: state.editSlaves});
  }
  d.last = p;
  updateStats();
}

function finishPolygon() {
  const poly = state.polygon;
  state.polygon = null;
  if (!poly || poly.pts.length < 3) { requestDraw({ov: true}); return; }
  state.editor.snapshot();
  const changed = state.editor.applyPolygon(poly.pts, state.value,
    {slaves: state.editSlaves});
  state.shape = null;
  if (changed) afterEdit();
  else {
    if (state.editor.undoStack.length) state.editor.undoStack.pop();
    requestDraw({ov: true});
  }
}

function cancelShape() {
  state.dragging = null;
  state.shape = null;
  state.polygon = null;
  requestDraw({ov: true});
}

function onKeyDown(e) {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "select" || tag === "textarea") return;
  if (e.code === "Space") { state.space = true; e.preventDefault(); return; }
  if (e.ctrlKey || e.metaKey) {
    if (e.key.toLowerCase() === "z") {
      e.preventDefault();
      if (e.shiftKey) doRedo(); else doUndo();
    } else if (e.key.toLowerCase() === "y") {
      e.preventDefault(); doRedo();
    } else if (e.key.toLowerCase() === "s") {
      e.preventDefault(); saveDesign();
    }
    return;
  }
  const map = {b: "brush", e: "eraser", l: "line", r: "rect",
               c: "circle", o: "ring", p: "polygon"};
  const t = map[e.key.toLowerCase()];
  if (t) { setTool(t); return; }
  if (e.key === "Escape") { cancelShape(); return; }
  if (e.key === "Enter" && state.polygon) {
    e.preventDefault(); finishPolygon();
  }
}

function onKeyUp(e) {
  if (e.code === "Space") state.space = false;
}

/* -------------------------------------------------------------- 编辑提交 */
function afterEdit() {
  state.edited = true;
  state.importedX = null;
  state.importedRadiusMm = null;
  redrawAll();
  updateStats();
  autosave();
  schedulePreview();
}

function doUndo() {
  if (!state.editor || !state.editor.undo()) return;
  afterEdit();
}
function doRedo() {
  if (!state.editor || !state.editor.redo()) return;
  afterEdit();
}

/* -------------------------------------------------------------- UI 同步 */
function setTool(tool) {
  state.tool = tool;
  if (tool !== "polygon") state.polygon = null;
  for (const b of document.querySelectorAll("#tools .tool")) {
    b.classList.toggle("active", b.dataset.tool === tool);
  }
  updateCursorStyle();
  requestDraw({ov: true});
  autosave();
}

function updateCursorStyle() {
  $("canvas-wrap").style.cursor =
    (state.tool === "brush" || state.tool === "eraser") ? "none" : "crosshair";
}

function setBrush(mm) {
  const v = Math.max(0.3, Math.min(8, Number(mm) || 0.3));
  state.brushMm = v;
  $("brush-radius").value = String(v);
  $("brush-radius-num").value = String(v);
  $("brush-label").textContent = v.toFixed(1);
  requestDraw({ov: true});
  autosave();
}

function setValue(v) {
  const x = Math.max(0, Math.min(1, Number(v)));
  state.value = x;
  $("paint-value").value = String(x);
  $("paint-value-num").value = String(x);
  $("value-label").textContent = x.toFixed(2);
  autosave();
}

function setRadius(mm) {
  const v = Number(mm);
  if (!Number.isFinite(v) || v <= 0) return;
  state.radiusMm = v;
  $("filter-radius").value = String(v);
  $("filter-radius-num").value = String(v);
  $("radius-label").textContent = v.toFixed(2);
  if (state.previewEnabled) schedulePreview(80);
  autosave();
}

function syncAllControls() {
  setTool(state.tool);
  setBrush(state.brushMm);
  setValue(state.value);
  setRadius(state.radiusMm);
  $("preview-filter").checked = state.previewEnabled;
  $("edit-slaves").checked = state.editSlaves;
  $("show-slaves").checked = state.showSlaves;
  $("x-preimage").checked = state.xPreimage;
  $("max-density").value = String(state.maxDensity);
  if (state.geo) {
    $("preset-ident").textContent = `恒等 (~${state.identityMm.toFixed(2)})`;
  }
}

function updateCursorInfo(p, k) {
  if (!p) { $("cursor-info").textContent = "—"; return; }
  const ang = (Math.atan2(p[1], p[0]) * 180 / Math.PI + 360) % 360;
  let text = `r=${Math.hypot(p[0], p[1]).toFixed(2)} mm  θ=${ang.toFixed(1)}°`;
  if (k >= 0 && state.geo) {
    const i = state.geo.designMaster[k];
    const val = state.editor ? state.editor.rho[i] : 0;
    const where = k < state.geo.nMaster ? "主域" : "镜像副本";
    text += `  单元 ${k}（${where}）→ 主 ${i}  ρ=${val.toFixed(3)}`;
  }
  $("cursor-info").textContent = text;
}

function updateStats() {
  if (!state.geo || !state.editor) return;
  const g = state.geo, rho = state.editor.rho, area = g.areaMaster;
  let sum = 0, tot = 0, solid = 0, mn = Infinity, mx = -Infinity;
  for (let i = 0; i < g.nMaster; i++) {
    const v = rho[i];
    sum += v * area[i]; tot += area[i];
    if (v > 0.5) solid++;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const mean = tot > 0 ? sum / tot : 0;
  let text = `主单元 ${g.nMaster}，镜像 ${g.nSlave}\n`
    + `平均密度（面积加权） ${mean.toFixed(4)}\n`
    + `实心单元（ρ>0.5） ${solid}    范围 ${mn.toFixed(3)} ~ ${mx.toFixed(3)}\n`
    + `滤波半径 ${state.radiusMm.toFixed(3)} mm（恒等 ≈ ${state.identityMm.toFixed(3)}）`;
  if (state.previewStats) {
    const s = state.previewStats;
    text += `\n实际起点 RMSE ${s.rmse.toFixed(4)}  max|Δ| ${s.max_abs.toFixed(3)}`
      + `  |Δ|>0.2 ${(s.frac_lost * 100).toFixed(1)}%`
      + `\n实际平均密度 ${s.mean_density.toFixed(4)}`;
  }
  $("design-stats").textContent = text;

  const warn = $("density-warn");
  const lim = Number($("max-density").value) || state.maxDensity;
  if (mean > lim + 1e-9) {
    warn.classList.remove("hidden");
    warn.textContent = `⚠ 起点平均密度 ${mean.toFixed(3)} > 上限 ${lim.toFixed(2)}：`
      + "优化会先削材料，手绘拓扑可能被抹掉。可降低手绘填充或调高密度上限。";
  } else {
    warn.classList.add("hidden");
  }
}

/* -------------------------------------------------------------- 预览 */
function schedulePreview(delay = 220) {
  clearTimeout(state.previewTimer);
  if (!state.previewEnabled) return;
  state.previewTimer = setTimeout(runPreview, delay);
}

async function runPreview() {
  if (!state.geo || !state.editor || !state.previewEnabled) return;
  const gen = state.loadGen;
  try {
    const body = {
      motor: state.motor,
      rho_master: Array.from(state.editor.rho),
      radius_mm: state.radiusMm,
      x_mode: state.xPreimage ? "preimage" : "direct",
      x0_override: x0Override(),
    };
    const r = await apiPost("/api/preview", body);
    if (gen !== state.loadGen) return;     // 切换网格后丢弃旧响应
    state.previewRho = Float32Array.from(r.effective_master);
    state.previewStats = r.stats;
    state.lost = new Uint8Array(state.geo.nMaster);
    for (let i = 0; i < state.geo.nMaster; i++) {
      state.lost[i] = Math.abs(state.previewRho[i] - state.editor.rho[i]) > 0.2 ? 1 : 0;
    }
    state.identityMm = r.identity_filter_radius_mm || state.identityMm;
    redrawAll();
    updateStats();
  } catch (e) {
    console.error(e);
    $("design-stats").textContent = "预览失败：" + e.message;
  }
}

function x0Override() {
  if (state.importedX && !state.edited
      && state.importedRadiusMm !== null
      && Math.abs(state.radiusMm - state.importedRadiusMm) < 1e-9) {
    return state.importedX;
  }
  return null;
}

/* -------------------------------------------------------------- 本地草稿 */
function autosave() {
  clearTimeout(state.autosaveTimer);
  state.autosaveTimer = setTimeout(saveLocal, 450);
}

function saveLocal() {
  if (!state.geo || !state.editor) return;
  try {
    localStorage.setItem(LOCAL_PREFIX + state.motor, JSON.stringify({
      motor: state.motor,
      sha: state.geo.sha,
      rho: Array.from(state.editor.rho),
      radiusMm: state.radiusMm,
      tool: state.tool,
      brushMm: state.brushMm,
      value: state.value,
      editSlaves: state.editSlaves,
      showSlaves: state.showSlaves,
      previewEnabled: state.previewEnabled,
      xPreimage: state.xPreimage,
      maxDensity: Number($("max-density").value) || state.maxDensity,
    }));
    localStorage.setItem(LOCAL_LAST, state.motor);
  } catch (e) { /* localStorage 满了就忽略 */ }
}

function loadLocal(motor) {
  try {
    const s = localStorage.getItem(LOCAL_PREFIX + (motor || state.motor));
    return s ? JSON.parse(s) : null;
  } catch (e) { return null; }
}

/* -------------------------------------------------------------- 保存/导入 */
async function saveDesign() {
  if (!state.geo || !state.editor) return;
  const name = $("design-name").value.trim() || "design";
  try {
    const r = await apiPost("/api/export", {
      motor: state.motor,
      name,
      rho_master: Array.from(state.editor.rho),
      radius_mm: state.radiusMm,
      x_mode: state.xPreimage ? "preimage" : "direct",
      x0_override: x0Override(),
      ladder: stagesText(),
    });
    const s = r.stats;
    $("save-msg").textContent =
      `已保存：${r.path}\n实际起点 RMSE=${s.rmse.toFixed(4)}，`
      + `max|Δ|=${s.max_abs.toFixed(3)}，平均密度=${s.mean_density.toFixed(4)}`;
    refreshDesigns();
  } catch (e) {
    $("save-msg").textContent = "保存失败：" + e.message;
  }
}

async function refreshDesigns() {
  try {
    const r = await apiGet(`/api/designs?motor=${encodeURIComponent(state.motor)}`);
    const sel = $("designs");
    sel.innerHTML = "";
    for (const d of r.designs) {
      const o = document.createElement("option");
      o.value = d.path;
      let extra = "";
      if (d.mean_density_master !== undefined) {
        extra = `  ρ̄=${Number(d.mean_density_master).toFixed(3)}`;
      }
      o.textContent = d.name + extra;
      sel.appendChild(o);
    }
    if (!r.designs.length) {
      const o = document.createElement("option");
      o.value = ""; o.textContent = "（暂无保存的设计）";
      o.disabled = true;
      sel.appendChild(o);
    }
  } catch (e) {
    console.error(e);
  }
}

async function loadSelectedDesign() {
  const p = $("designs").value;
  if (!p) return;
  await openImport(p, true);
}

async function browseImport() {
  try {
    const r = await apiPost("/api/browse-native", {
      title: "选择 NPZ（设计/优化结果）",
      dir: "tools/_out",
    });
    if (r.path) await openImport(r.path, false);
  } catch (e) {
    alert("选择文件失败：" + e.message);
  }
}

async function openImport(path, autoApply) {
  try {
    const insp = await apiPost("/api/import/inspect", {path, motor: state.motor});
    state.importCtx = {path, insp};
    const sel = $("import-key");
    sel.innerHTML = "";
    const keys = (insp.keys || []).filter((k) => k.selectable);
    for (const k of keys) {
      const o = document.createElement("option");
      o.value = k.key;
      const dims = (k.shape || []).join("×");
      o.textContent = `${k.key}  [${dims}]  ${k.kind}`
        + (k.looks_like_x ? "  (x)" : "");
      sel.appendChild(o);
    }
    const prefer = ["x_best", "x_selected", "rho_final", "rho_master", "x0", "rho"];
    let pick = keys[0] ? keys[0].key : "";
    for (const p of prefer) {
      if (keys.some((k) => k.key === p)) { pick = p; break; }
    }
    if (pick) sel.value = pick;
    $("import-path").textContent = path;
    const srcR = insp.source_filter_radius_mm;
    $("import-radius").value = String(srcR !== null && srcR !== undefined
      ? Number(srcR).toFixed(3) : state.radiusMm.toFixed(3));
    syncImportAs();
    updateImportNote();
    $("modal-backdrop").classList.remove("hidden");
    if (autoApply && pick) await applyImport();
  } catch (e) {
    alert("读取 NPZ 失败：" + e.message);
  }
}

function syncImportAs() {
  const key = $("import-key").value || "";
  $("import-as").value = /(^|_)x($|_)|^x/.test(key) && !/rho/.test(key)
    ? "x" : "auto";
}

function updateImportNote() {
  const ctx = state.importCtx;
  const key = $("import-key").value;
  const k = ctx && ctx.insp.keys.find((z) => z.key === key);
  if (!k) { $("import-note").textContent = ""; return; }
  let note = `形状 ${(k.shape || []).join("×")}，类型 ${k.kind}`;
  if (k.looks_like_x) note += "；按设计变量 x 处理时用源滤波半径";
  if ((k.shape || []).length === 2) {
    note += `；2D 数组取第 ${$("import-row").value} 行/列（-1=最后）`;
  }
  if (ctx.insp.source_filter_radius_mm != null) {
    note += `；源滤波半径 ${Number(ctx.insp.source_filter_radius_mm).toFixed(3)} mm`;
  }
  $("import-note").textContent = note;
  syncImportAs();
}

async function applyImport() {
  const ctx = state.importCtx;
  if (!ctx) return;
  const key = $("import-key").value;
  if (!key) return;
  const gen = state.loadGen;
  try {
    const res = await apiPost("/api/import/apply", {
      path: ctx.path,
      motor: state.motor,
      key,
      row: Number($("import-row").value),
      as: $("import-as").value,
      fallback_radius_mm: Number($("import-radius").value),
    });
    if (gen !== state.loadGen || !state.editor) return;
    state.editor.rho = Float32Array.from(res.rho_master);
    state.edited = false;
    state.importedX = res.imported_x || null;
    state.importedRadiusMm = res.source_radius_mm;
    if (res.source_radius_mm) setRadius(Number(res.source_radius_mm));
    closeImport();
    redrawAll();
    updateStats();
    if (state.previewEnabled) schedulePreview(0);
    autosave();
    const warn = (res.warnings || []).join("\n");
    $("save-msg").textContent = `已导入 ${key}` + (warn ? "\n" + warn : "");
  } catch (e) {
    alert("导入失败：" + e.message);
  }
}

function closeImport() {
  $("modal-backdrop").classList.add("hidden");
  state.importCtx = null;
}

/* -------------------------------------------------------------- 优化 */
function stagesText() {
  return $("stages").value.trim();
}

async function startOptimization() {
  if (!state.geo || !state.editor) return;
  const preset = $("stage-preset").value;
  const useLadder = LADDER_PRESETS.includes(preset);
  const body = {
    motor: state.motor,
    name: $("design-name").value.trim() || "design",
    rho_master: Array.from(state.editor.rho),
    radius_mm: state.radiusMm,
    x_mode: state.xPreimage ? "preimage" : "direct",
    x0_override: x0Override(),
    stages: useLadder ? null : stagesText(),
    ladder: useLadder ? preset : null,
    max_eval_per_stage: Number($("max-eval").value) || 100,
    mma_move: Number($("mma-move").value) || 0.05,
    n_train: Number($("n-train").value) || 11,
    n_eval: Number($("n-eval").value) || 6,
    max_density: Number($("max-density").value) || 0.65,
    strict: $("strict").checked,
    extra_args: $("extra-args").value,
  };
  try {
    $("job-status").textContent = "启动中…";
    const r = await apiPost("/api/job/start", body);
    state.jobOffset = 0;
    $("job-log").textContent = "";
    $("monitor-cmd").textContent = (r.monitor_cmd || []).map(quoteArg).join(" ");
    updateJobChip(r.job);
    startJobPolling();
    refreshDesigns();
  } catch (e) {
    $("job-status").textContent = "启动失败：" + e.message;
  }
}

function quoteArg(a) {
  return /\s/.test(a) ? `"${a}"` : a;
}

async function stopOptimization() {
  try {
    const r = await apiPost("/api/job/stop", {});
    updateJobChip(r.state);
  } catch (e) {
    $("job-status").textContent = "停止失败：" + e.message;
  }
}

function startJobPolling() {
  if (state.jobTimer) clearInterval(state.jobTimer);
  state.jobTimer = setInterval(pollJob, 1000);
  pollJob();
}

async function maybeResumeJob() {
  try {
    const st = await apiGet("/api/state");
    if (st.job && (st.job.running || st.job.status !== "idle")) {
      state.jobOffset = 0;
      startJobPolling();
    }
  } catch (e) { /* ignore */ }
}

async function pollJob() {
  try {
    const r = await apiGet(`/api/job/log?offset=${state.jobOffset}`);
    state.jobOffset = r.next_offset;
    if (r.text) appendLog(r.text);
    if (r.state) {
      updateJobChip(r.state);
      $("job-status").textContent = r.state.msg
        + `  （评价 ${r.state.n_evals}，${Math.round(r.state.elapsed_s)} s）`;
    }
    if (r.done) {
      clearInterval(state.jobTimer);
      state.jobTimer = null;
    }
  } catch (e) {
    $("job-status").textContent = "日志读取失败：" + e.message;
  }
}

function appendLog(text) {
  const el = $("job-log");
  const lines = (el.textContent + text).split("\n");
  el.textContent = lines.slice(-220).join("\n");
  el.scrollTop = el.scrollHeight;
}

function updateJobChip(job) {
  if (!job) return;
  const chip = $("job-chip");
  chip.className = "chip";
  const map = {
    idle: ["idle", "优化空闲"],
    starting: ["busy", "优化启动中"],
    preparing: ["busy", "预组装中"],
    running: ["running", "优化运行中"],
    done: ["good", "优化完成"],
    failed: ["bad", "优化失败"],
    stopped: ["bad", "优化已停止"],
    unknown: ["bad", "优化状态未知"],
  };
  const [cls, text] = map[job.status] || ["idle", job.status || "未知"];
  chip.classList.add(cls);
  chip.textContent = text;
  chip.title = job.msg || "";
}

function setBuildChip(text, cls) {
  const chip = $("build-chip");
  chip.className = "chip" + (cls ? " " + cls : "");
  chip.textContent = text;
}

function updateBuildChip(build) {
  if (!build) return;
  const chip = $("build-chip");
  chip.className = "chip";
  if (build.ready) {
    chip.classList.add("good");
    chip.textContent = "几何就绪";
  } else if (build.error) {
    chip.classList.add("bad");
    chip.textContent = "几何失败";
    chip.title = build.error;
  } else {
    chip.classList.add("busy");
    chip.textContent = `几何构建中… ${Math.round(build.elapsed_s || 0)} s`;
  }
}
