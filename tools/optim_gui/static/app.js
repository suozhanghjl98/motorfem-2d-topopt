"use strict";
/* tools/optim_gui/static/app.js —— 前端逻辑：轮询状态、渲染配置表单、
   画转子几何（三角填充 + viridis(ρ)）与目标/约束/变化轨迹、控制按钮。 */

const VIRIDIS = [
[68,1,84],
[68,2,86],
[69,4,87],
[69,5,89],
[70,7,90],
[70,8,92],
[70,10,93],
[70,11,94],
[71,13,96],
[71,14,97],
[71,16,99],
[71,17,100],
[71,19,101],
[72,20,103],
[72,22,104],
[72,23,105],
[72,24,106],
[72,26,108],
[72,27,109],
[72,28,110],
[72,29,111],
[72,31,112],
[72,32,113],
[72,33,115],
[72,35,116],
[72,36,117],
[72,37,118],
[72,38,119],
[72,40,120],
[72,41,121],
[71,42,122],
[71,44,122],
[71,45,123],
[71,46,124],
[71,47,125],
[70,48,126],
[70,50,126],
[70,51,127],
[70,52,128],
[69,53,129],
[69,55,129],
[69,56,130],
[68,57,131],
[68,58,131],
[68,59,132],
[67,61,132],
[67,62,133],
[66,63,133],
[66,64,134],
[66,65,134],
[65,66,135],
[65,68,135],
[64,69,136],
[64,70,136],
[63,71,136],
[63,72,137],
[62,73,137],
[62,74,137],
[62,76,138],
[61,77,138],
[61,78,138],
[60,79,138],
[60,80,139],
[59,81,139],
[59,82,139],
[58,83,139],
[58,84,140],
[57,85,140],
[57,86,140],
[56,88,140],
[56,89,140],
[55,90,140],
[55,91,141],
[54,92,141],
[54,93,141],
[53,94,141],
[53,95,141],
[52,96,141],
[52,97,141],
[51,98,141],
[51,99,141],
[50,100,142],
[50,101,142],
[49,102,142],
[49,103,142],
[49,104,142],
[48,105,142],
[48,106,142],
[47,107,142],
[47,108,142],
[46,109,142],
[46,110,142],
[46,111,142],
[45,112,142],
[45,113,142],
[44,113,142],
[44,114,142],
[44,115,142],
[43,116,142],
[43,117,142],
[42,118,142],
[42,119,142],
[42,120,142],
[41,121,142],
[41,122,142],
[41,123,142],
[40,124,142],
[40,125,142],
[39,126,142],
[39,127,142],
[39,128,142],
[38,129,142],
[38,130,142],
[38,130,142],
[37,131,142],
[37,132,142],
[37,133,142],
[36,134,142],
[36,135,142],
[35,136,142],
[35,137,142],
[35,138,141],
[34,139,141],
[34,140,141],
[34,141,141],
[33,142,141],
[33,143,141],
[33,144,141],
[33,145,140],
[32,146,140],
[32,146,140],
[32,147,140],
[31,148,140],
[31,149,139],
[31,150,139],
[31,151,139],
[31,152,139],
[31,153,138],
[31,154,138],
[30,155,138],
[30,156,137],
[30,157,137],
[31,158,137],
[31,159,136],
[31,160,136],
[31,161,136],
[31,161,135],
[31,162,135],
[32,163,134],
[32,164,134],
[33,165,133],
[33,166,133],
[34,167,133],
[34,168,132],
[35,169,131],
[36,170,131],
[37,171,130],
[37,172,130],
[38,173,129],
[39,173,129],
[40,174,128],
[41,175,127],
[42,176,127],
[44,177,126],
[45,178,125],
[46,179,124],
[47,180,124],
[49,181,123],
[50,182,122],
[52,182,121],
[53,183,121],
[55,184,120],
[56,185,119],
[58,186,118],
[59,187,117],
[61,188,116],
[63,188,115],
[64,189,114],
[66,190,113],
[68,191,112],
[70,192,111],
[72,193,110],
[74,193,109],
[76,194,108],
[78,195,107],
[80,196,106],
[82,197,105],
[84,197,104],
[86,198,103],
[88,199,101],
[90,200,100],
[92,200,99],
[94,201,98],
[96,202,96],
[99,203,95],
[101,203,94],
[103,204,92],
[105,205,91],
[108,205,90],
[110,206,88],
[112,207,87],
[115,208,86],
[117,208,84],
[119,209,83],
[122,209,81],
[124,210,80],
[127,211,78],
[129,211,77],
[132,212,75],
[134,213,73],
[137,213,72],
[139,214,70],
[142,214,69],
[144,215,67],
[147,215,65],
[149,216,64],
[152,216,62],
[155,217,60],
[157,217,59],
[160,218,57],
[162,218,55],
[165,219,54],
[168,219,52],
[170,220,50],
[173,220,48],
[176,221,47],
[178,221,45],
[181,222,43],
[184,222,41],
[186,222,40],
[189,223,38],
[192,223,37],
[194,223,35],
[197,224,33],
[200,224,32],
[202,225,31],
[205,225,29],
[208,225,28],
[210,226,27],
[213,226,26],
[216,226,25],
[218,227,25],
[221,227,24],
[223,227,24],
[226,228,24],
[229,228,25],
[231,228,25],
[234,229,26],
[236,229,27],
[239,229,28],
[241,229,29],
[244,230,30],
[246,230,32],
[248,230,33],
[251,231,35],
[253,231,37]
];

const $ = (id) => document.getElementById(id);
const STATUS_LABEL = {
  idle: "空闲", building: "构建中", loading: "加载中", running: "运行中",
  paused: "已暂停", stopping: "停止中", stopped: "已停止",
  done: "完成", error: "出错",
};
const CON_COLORS = ["#dc2626", "#d97706", "#2563eb", "#16a34a"];
const CON_NAMES = ["con1 Arkkio", "con2 密度", "con3 转矩", "con4 VD"];

let state = null;
let hist = { step: [], obj: [], cons: [], fun_ms: [], em_ms: [], mech_ms: [],
             step_ms: [], dx_rel: [] };
let geom = null;
let geomVersion = -1;
let runId = -1;
let paths = null;
let rhoIndex = null;
let geomReady = false;
let formCfgKey = null;

// ---------------------------------------------------------------- utils
function fetchJSON(url, opts) {
  return fetch(url, Object.assign({ headers: { "Accept": "application/json" } }, opts))
    .then((r) => r.json());
}

function fmtNum(v) {
  if (v == null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-4 || a > 1e5)) return v.toExponential(2);
  return v.toFixed(4);
}

function fmtMs(ms) {
  return (ms == null || !Number.isFinite(ms)) ? "—" : ms.toFixed(0) + " ms";
}

function fmtTime(s) {
  if (s == null || !Number.isFinite(s) || s < 0) return "—";
  if (s < 90) return Math.round(s) + " s";
  if (s < 5400) return (s / 60).toFixed(1) + " min";
  return (s / 3600).toFixed(1) + " h";
}

function viridis(t) {
  const x = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return "rgb(" + Math.round(a[0] + (b[0] - a[0]) * f) + ","
    + Math.round(a[1] + (b[1] - a[1]) * f) + ","
    + Math.round(a[2] + (b[2] - a[2]) * f) + ")";
}

function getPath(obj, path) {
  let o = obj;
  for (const k of path) { if (o == null) return undefined; o = o[k]; }
  return o;
}
function setPath(obj, path, v) {
  let o = obj;
  for (let i = 0; i < path.length - 1; i++) {
    const k = path[i];
    if (typeof o[k] !== "object" || o[k] === null) o[k] = {};
    o = o[k];
  }
  o[path[path.length - 1]] = v;
}

// 预设下拉应选的值：某预设的全部 values 与当前 cfg 逐项相等 → 选中它；
// 无匹配（手动微调过）→ 返回 null（调用方显示『自定义』）。
function presetValueFor(g, cfg) {
  if (!g.preset || !g.preset.options) return null;
  for (const opt of g.preset.options) {
    const vs = opt.values || {};
    if (Object.keys(vs).every((key) => {
      const got = getPath(cfg, key.split("."));
      return String(got) === String(vs[key]);
    })) return opt.value;
  }
  return null;
}

// 预设下拉的 id（与 renderForm 里生成的一致）
function presetSelectId(g) {
  return "preset_" + g.group.replace(/\s+/g, "_");
}

// 给某个预设下拉设置选中值：匹配 → 选中该预设；无匹配 → 追加『自定义』并选中。
// removeCustom=true 时先清掉上次追加的『自定义』占位（同一个下拉被重复更新时用）。
function setPresetSelectValue(psel, g, cfg, removeCustom) {
  if (removeCustom) {
    const old = psel.querySelector('option[value="__custom__"]');
    if (old) old.remove();
  }
  const pv = presetValueFor(g, cfg);
  if (pv !== null) {
    psel.value = pv;
  } else {
    const custom = document.createElement("option");
    custom.value = "__custom__";
    custom.textContent = "自定义（手动微调，不匹配任何预设）";
    psel.appendChild(custom);
    psel.value = "__custom__";
  }
}

// ---------------------------------------------------------------- 配置表单
function renderForm(schema, cfg) {
  const root = $("config-form");
  root.innerHTML = "";
  for (const g of schema) {
    const div = document.createElement("div");
    div.className = "cfg-group";
    const h = document.createElement("h3");
    h.textContent = g.group;
    div.appendChild(h);

    // 组内预设下拉（一键应用一组设置；仅性能优化组有）
    if (g.preset) {
      const prow = document.createElement("div");
      prow.className = "cfg-row";
      const pl = document.createElement("label");
      pl.textContent = g.preset.label;
      pl.title = "一键应用一组性能设置。具体数值在下方『详细设置』里可再微调。";
      prow.appendChild(pl);
      const psel = document.createElement("select");
      psel.id = presetSelectId(g);
      for (const opt of g.preset.options) {
        const op = document.createElement("option");
        op.value = opt.value;
        op.textContent = opt.label;
        op.title = opt.hint || "";
        psel.appendChild(op);
      }
      // 用当前 cfg 回填选中值，避免重渲染（开始/加载检查点后 poll 触发）
      // 把预设下拉永远钉回第一项『生产默认』。
      setPresetSelectValue(psel, g, cfg, false);
      psel.onchange = () => applyPreset(g.preset.options, psel.value);
      prow.appendChild(psel);
      div.appendChild(prow);
    }

    // 折叠组：详细字段放进 <details>（默认收起）
    let body = div;
    if (g.collapsible) {
      const det = document.createElement("details");
      det.className = "cfg-details";
      const sum = document.createElement("summary");
      sum.textContent = "详细设置（默认隐藏）";
      sum.title = "展开后可逐个微调性能参数；改动立即反映在表单里。";
      det.appendChild(sum);
      div.appendChild(det);
      body = det;
    }

    for (const f of g.fields) {
      const row = document.createElement("div");
      row.className = "cfg-row";
      const lab = document.createElement("label");
      lab.textContent = f.label;
      lab.title = f.hint || f.label;   // 悬浮提示 = hint（含含义 + 生产默认值）
      row.appendChild(lab);
      const id = f.path.join("__");
      lab.htmlFor = id;
      const val = getPath(cfg, f.path);
      let el;
      if (f.type === "select") {
        el = document.createElement("select");
        for (const o of f.options) {
          const op = document.createElement("option");
          // 选项可以是字符串（值=显示名）或 {value, label}（值=内部名，
          // 显示=含义）。选 {value,label} 时 title 悬停显示内部名。
          const v = (o && typeof o === "object") ? o.value : o;
          const lbl = (o && typeof o === "object") ? o.label : o;
          op.value = v; op.textContent = lbl;
          if (o && typeof o === "object" && o.tip) op.title = o.tip;
          if (String(val) === String(v)) op.selected = true;
          el.appendChild(op);
        }
      } else if (f.type === "bool") {
        el = document.createElement("input");
        el.type = "checkbox";
        el.checked = !!val;
      } else if (f.type === "text") {
        // 文本字段（如位置表）：数组显示为逗号分隔
        el = document.createElement("input");
        el.type = "text";
        el.value = (val == null) ? "" :
          (Array.isArray(val) ? val.join(", ") : String(val));
      } else {
        el = document.createElement("input");
        el.type = "number";
        el.step = (f.type === "int" || f.type === "int_none") ? "1" : "any";
        el.value = (val == null) ? "" : String(val);
      }
      el.id = id;
      el.title = f.hint || f.label;   // 悬浮在输入框上也显示提示
      row.appendChild(el);
      body.appendChild(row);
    }
    root.appendChild(div);
  }
}

// 应用性能预设：把预设里的键值对写进对应表单控件（不立即开始）
function applyPreset(options, value) {
  const opt = options.find((o) => o.value === value);
  if (!opt) return;
  for (const key in opt.values) {
    const path = key.split(".");
    const el = $(path.join("__"));
    if (!el) continue;
    const v = opt.values[key];
    if (el.type === "checkbox") el.checked = !!v;
    else if (el.tagName === "SELECT") el.value = String(v);
    else el.value = (v == null) ? "" : String(v);
  }
}

function gatherConfig() {
  if (!state || !state.schema) return {};
  const out = {};
  for (const g of state.schema) {
    for (const f of g.fields) {
      const el = $(f.path.join("__"));
      if (!el) continue;
      let v;
      if (f.type === "bool") v = el.checked;
      else if (f.type === "select") v = el.value;
      else if (f.type === "text") v = el.value;   // 字符串（服务端解析成 List）
      else {
        const s = el.value.trim();
        if (f.type === "int_none" && s === "") v = null;
        else if (f.type === "int" || f.type === "int_none") {
          v = (s === "" && f.type === "int_none") ? null : parseInt(s, 10);
        } else v = s === "" ? 0 : parseFloat(s);
        if (Number.isNaN(v)) v = null;
      }
      setPath(out, f.path, v);
    }
  }
  return out;
}

function applyCfgToForm(cfg) {
  if (!state || !state.schema) return;
  for (const g of state.schema) {
    for (const f of g.fields) {
      const el = $(f.path.join("__"));
      if (!el) continue;
      const val = getPath(cfg, f.path);
      if (f.type === "bool") el.checked = !!val;
      else if (f.type === "select") el.value = String(val);
      else if (f.type === "text")
        el.value = (val == null) ? "" :
          (Array.isArray(val) ? val.join(", ") : String(val));
      else el.value = (val == null) ? "" : String(val);
    }
    if (g.preset) {
      const psel = $(presetSelectId(g));
      if (psel) setPresetSelectValue(psel, g, cfg, true);
    }
  }
}

// ---------------------------------------------------------------- 几何绘制
function buildPaths() {
  const pts = geom.xy, tri = geom.tri;
  paths = new Array(geom.n_elem);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of pts) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  for (let e = 0; e < geom.n_elem; e++) {
    const [a, b, c] = tri[e];
    const p = new Path2D();
    p.moveTo(pts[a][0], pts[a][1]);
    p.lineTo(pts[b][0], pts[b][1]);
    p.lineTo(pts[c][0], pts[c][1]);
    p.closePath();
    paths[e] = p;
  }
  rhoIndex = new Map();
  for (let k = 0; k < geom.design_idx.length; k++) rhoIndex.set(geom.design_idx[k], k);
  geom.bounds = { minX, maxX, minY, maxY };
}

function fitCanvas(cv, aspect) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth;
  if (!w) return;
  const h = Math.max(120, Math.round(w / aspect));
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.style.height = h + "px";
}

function drawGeometry(rho) {
  if (!geomReady || !paths) return;
  const rhoArr = rho || new Array(geom.n_topo).fill(0);
  const cv = $("geom"), ctx = cv.getContext("2d");
  fitCanvas(cv, 1.0);
  const b = geom.bounds;
  const pad = 12 * (window.devicePixelRatio || 1);
  const S = Math.max((cv.width - 2 * pad) / (b.maxX - b.minX),
                     (cv.height - 2 * pad) / (b.maxY - b.minY));
  const cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2;
  ctx.setTransform(S, 0, 0, -S, cv.width / 2 - S * cx, cv.height / 2 + S * cy);
  ctx.clearRect(b.minX, b.minY, b.maxX - b.minX, b.maxY - b.minY);
  ctx.lineWidth = 0;
  const base = geom.base_colors || {};
  for (let e = 0; e < geom.n_elem; e++) {
    const cid = geom.color_id[e];
    if (cid === 3) {
      const k = rhoIndex.get(e);
      ctx.fillStyle = viridis(k == null ? 0 : Math.max(0, Math.min(1, rhoArr[k])));
    } else {
      ctx.fillStyle = base[cid] || "#ffffff";
    }
    ctx.fill(paths[e]);
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

// ---------------------------------------------------------------- 轨迹绘制
function drawPlot(ctx, x0, y0, w, h, series, opts) {
  ctx.save();
  ctx.fillStyle = "#fff";
  ctx.fillRect(x0, y0, w, h);
  ctx.strokeStyle = "#dde1e6";
  ctx.lineWidth = 1;
  ctx.strokeRect(x0, y0, w, h);

  let ymin = Infinity, ymax = -Infinity;
  for (const s of series) {
    for (const v of s.data) {
      if (v == null || !Number.isFinite(v)) continue;
      if (opts.logY && !(v > 0)) continue;    // logY 下非正数（含首步 dx_rel=0）无对数，跳过
      if (opts.logY) {
        const l = Math.log10(v);
        if (l < ymin) ymin = l; if (l > ymax) ymax = l;
      } else {
        if (v < ymin) ymin = v; if (v > ymax) ymax = v;
      }
    }
  }
  if (!Number.isFinite(ymin)) { ctx.restore(); return; }
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const pad = (ymax - ymin) * 0.05;
  ymin -= pad; ymax += pad;

  const t0 = hist.step[0], t1 = hist.step[hist.step.length - 1];
  const X = (i) => x0 + (hist.step[i] - t0) / Math.max(t1 - t0, 1) * w;
  const Y = (v) => {
    if (opts.logY && !(v > 0)) return null;    // logY 下非正数不画点（断线）
    const y = opts.logY ? Math.log10(v) : v;
    return y0 + h * (1 - (y - ymin) / (ymax - ymin));
  };

  ctx.strokeStyle = "#eef0f3";
  for (let g = 1; g < 4; g++) {
    const gy = y0 + h * g / 4;
    ctx.beginPath(); ctx.moveTo(x0, gy); ctx.lineTo(x0 + w, gy); ctx.stroke();
  }
  if (opts.yZeroLine !== undefined && !opts.logY) {
    const zy = Y(0);
    if (zy > y0 - 1 && zy < y0 + h + 1) {
      ctx.strokeStyle = "#94a3b8";
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x0, zy); ctx.lineTo(x0 + w, zy); ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  ctx.font = "10px sans-serif";
  ctx.fillStyle = "#6b7280";
  ctx.textAlign = "right";
  ctx.fillText(opts.logY ? "1e" + ymax.toFixed(0) : fmtNum(ymax), x0 + w - 2, y0 + 10);
  ctx.fillText(opts.logY ? "1e" + ymin.toFixed(0) : fmtNum(ymin), x0 + w - 2, y0 + h - 2);
  ctx.textAlign = "left";

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < s.data.length; i++) {
      const v = s.data[i];
      if (v == null || !Number.isFinite(v)) continue;
      const px = X(i), py = Y(v);
      if (py == null) { started = false; continue; }   // logY 断线点
      if (!started) { ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawTrajectory() {
  const cv = $("traj"), ctx = cv.getContext("2d");
  fitCanvas(cv, 1.7);
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  if (!hist.step.length) {
    ctx.font = "12px sans-serif";
    ctx.fillStyle = "#9ca3af";
    ctx.textAlign = "center";
    ctx.fillText(state && state.read_only ? "等待外部优化评价日志（只读）" : "还没有数据 —— 点『开始』跑起来", W / 2, H / 2);
    return;
  }
  const x0 = 8, w = W - 16;
  const hObj = Math.round(H * 0.36);
  const hCon = Math.round(H * 0.30);
  const hDx = H - hObj - hCon - 16;
  drawPlot(ctx, x0, 2, w, hObj, [{ data: hist.obj, color: "#2563eb" }],
           { yZeroLine: false, logY: false });
  const consSeries = CON_COLORS.map((c, i) => ({
    data: hist.cons.map((row) => row[i]), color: c,
  }));
  drawPlot(ctx, x0, hObj + 8, w, hCon, consSeries,
           { yZeroLine: 0, logY: false });
  drawPlot(ctx, x0, hObj + hCon + 16, w, hDx,
           [{ data: hist.dx_rel, color: "#111827" }],
           { yZeroLine: false, logY: true });

  ctx.font = "10px sans-serif";
  ctx.fillStyle = "#6b7280";
  ctx.textAlign = "left";
  ctx.fillText("obj", x0 + 2, 2 + hObj - 4);
  ctx.fillText("约束（≤0 可行）", x0 + 2, hObj + 8 + hCon - 4);
  ctx.fillText("dx_rel（log）", x0 + 2, hObj + hCon + 16 + hDx - 4);
}

// ---------------------------------------------------------------- 状态渲染
function renderStatus(s) {
  const chip = $("status-chip");
  chip.className = "chip " + (STATUS_LABEL[s.status] ? s.status : "idle");
  chip.textContent = STATUS_LABEL[s.status] || s.status;
  const busy = s.status === "running" || s.status === "building"
    || s.status === "loading" || s.status === "stopping";
  $("btn-start").disabled = busy;
  $("btn-pause").disabled = s.status !== "running";
  $("btn-resume").disabled = s.status !== "paused";
  $("btn-step").disabled = (s.status !== "paused" && s.status !== "running");
  $("btn-stop").disabled = (s.status !== "running" && s.status !== "paused");
  $("btn-save").disabled = (s.step < 1);
  $("btn-load").disabled = busy;
  $("run-msg").textContent = s.msg || "";
  if (s.power) $("power").textContent = "电源：" + s.power;
  $("geom-note").textContent = s.geometry_version >= 0
    ? (s.n_topo ? "（设计区 " + s.n_topo + " 单元）" : "") : "";
  if (s.cfg) $("traj-note").textContent = "后端 " + (s.cfg.solver.solver_backend || "superlu")
    + (s.cfg.runtime.n_jobs ? " / n_jobs=" + s.cfg.runtime.n_jobs : " / 串行");
  if (s.read_only) {
    document.querySelectorAll("button,input,select,textarea").forEach(el => { el.disabled = true; });
    $("geom-note").textContent = s.snapshot_note;
    $("traj-note").textContent = "横轴为累计评价次数（不是 MMA 迭代）；约束顺序 ArkBE / 密度 / flux70 / VD；未启用项为空";
    chip.textContent = ({waiting:"等待",running:"运行中",done:"完成",error:"失败",unknown:"状态未知"})[s.status] || s.status;
  }
}

function renderStats(s) {
  const t = s.times || {};
  // [标签, 数值, 悬浮提示]
  const items = [
    [s.read_only ? "阶段 / 评价次数" : "步", s.read_only ? ((s.stage || "—") + " / " + s.step + "（非迭代）") : s.step + " / " + s.target,
      s.read_only ? "当前阶段已记录的目标函数评价次数，不是优化迭代数。" : "已完成步数 / 目标步数。达到目标自动结束（done）。"],
    ["目标 obj", fmtNum(s.obj),
      "当前目标函数值 = 弹性应变能 U + 应力 AL 罚项。越小越好。"],
    ["con1 Arkkio", s.cons ? fmtNum(s.cons[0]) : "—",
      "转矩波动约束（Arkkio 法）= (Arkkio 波动/上限 − 1)·权重。要求 ≤ 0。"],
    ["con2 密度", s.cons ? fmtNum(s.cons[1]) : "—",
      "体积约束 = 平均密度/max_density − 1。要求 ≤ 0。"],
    ["con3 转矩", s.cons ? fmtNum(s.cons[2]) : "—",
      "转矩约束 = (1 − 平均转矩/mean_torque_obj)·权重。要求 ≤ 0（转矩 ≥ 目标）。"],
    ["con4 VD", s.cons ? fmtNum(s.cons[3]) : "—",
      "转矩波动约束（虚功法 VD）= (VD 波动/上限 − 1)·权重。要求 ≤ 0。"],
    ["dx_rel", s.dx_rel == null ? "—" : s.dx_rel.toExponential(2),
      "相邻两步设计变量的相对变化 ||Δx||/||x||。持续减小 = 收敛趋势。"],
    ["fun", fmtMs(t.fun_ms),
      "单步总耗时（EM + 力学 + 聚合 + 滤波）。"],
    ["EM", fmtMs(t.em_ms),
      "电磁段耗时（Newton 求解 + 冻结分析，通常占整步 90%+）。"],
    ["力学", fmtMs(t.mech_ms),
      "力学段耗时（应力/应变能求解）。"],
    ["已用", fmtTime(s.elapsed_s),
      "本轮从启动到现在的墙钟时间。"],
    ["ETA", fmtTime(s.eta_s),
      "按当前平均步速估算的剩余时间。"],
    ["AL μ", s.al_mu == null ? "—" : fmtNum(s.al_mu),
      "应力约束的增广拉格朗日罚因子。约束被违反次数越多越大（从 μ0=0.01 起）。"],
    ["AL n_call", s.al_n_call == null ? "—" : s.al_n_call,
      "应力约束被违反（>0）的累计次数。越大说明应力约束越紧。"],
  ];
  $("stats").innerHTML = items.map(([k, v, hint]) =>
    "<div class=stat title='" + (hint || "").replace(/'/g, "&#39;")
      .replace(/"/g, "&quot;") + "'><span>" + k + "</span><b>" + v + "</b></div>").join("");
}

function renderSaves(saves) {
  const sel = $("sel-saves");
  if (sel.options.length === saves.length
      && sel.options[0] && sel.options[0].value === saves[0]) return;
  sel.innerHTML = "";
  for (const name of saves) {
    const op = document.createElement("option");
    op.value = name; op.textContent = name;
    sel.appendChild(op);
  }
}

// ---------------------------------------------------------------- 数据拉取
async function loadGeometry() {
  const r = await fetchJSON("/api/geometry");
  geomVersion = r.version;
  geom = r.geometry;
  if (geom) {
    buildPaths();
    geomReady = true;
    drawGeometry(null);
  }
}

async function loadHistory(start) {
  const h = await fetchJSON("/api/history?start=" + start);
  hist.step.push.apply(hist.step, h.step);
  hist.obj.push.apply(hist.obj, h.obj);
  for (const c of h.cons) hist.cons.push(c);
  hist.fun_ms.push.apply(hist.fun_ms, h.fun_ms);
  hist.em_ms.push.apply(hist.em_ms, h.em_ms);
  hist.mech_ms.push.apply(hist.mech_ms, h.mech_ms);
  hist.step_ms.push.apply(hist.step_ms, h.step_ms);
  hist.dx_rel.push.apply(hist.dx_rel, h.dx_rel);
}

let _pollBusy = false;
async function poll() {
  if (_pollBusy) return;   // 防重入：启动 poll() 与 setInterval 并发会重复 loadHistory
  _pollBusy = true;
  try {
    const s = await fetchJSON("/api/state");
    state = s;
    renderStatus(s);
    renderStats(s);
    if (s.run_id !== runId) {
      runId = s.run_id;
      hist = { step: [], obj: [], cons: [], fun_ms: [], em_ms: [],
               mech_ms: [], step_ms: [], dx_rel: [] };
      if (s.history_len > 0) await loadHistory(0);
      drawTrajectory();
    } else if (s.history_len > hist.step.length) {
      await loadHistory(hist.step.length);
      drawTrajectory();
    }
    if (s.cfg && JSON.stringify(s.cfg) !== formCfgKey) {
      formCfgKey = JSON.stringify(s.cfg);
      renderForm(s.schema, s.cfg);
    }
    if (s.geometry_version !== geomVersion) await loadGeometry();
    const rho = s.rho || (!s.read_only && s.n_topo ? new Array(s.n_topo).fill(0) : null);
    if (rho && geomReady) drawGeometry(rho);
    else if (s.read_only) $("geom").getContext("2d").clearRect(0, 0, $("geom").width, $("geom").height);
    renderSaves(s.saves || []);
  } catch (err) {
    console.error("poll 失败：", err);
  } finally {
    _pollBusy = false;
  }
}

// ---------------------------------------------------------------- 按钮
function wireButtons() {
  const post = (path, body) => fetchJSON(path, {
    method: "POST",
    body: JSON.stringify(body || {}),
    headers: { "Content-Type": "application/json" },
  });

  $("btn-start").onclick = async () => {
    const body = {
      config: gatherConfig(),
      mesh_path: $("in-mesh").value,
      csv_path: $("in-csv").value,
    };
    const steps = parseInt($("in-steps").value, 10);
    if (Number.isFinite(steps) && steps > 0) body.steps = steps;
    const r = await post("/api/start", body);
    if (!r.ok) alert(r.msg);
  };
  $("btn-pause").onclick = async () => {
    const r = await post("/api/pause");
    if (!r.ok) alert(r.msg);
  };
  $("btn-resume").onclick = async () => {
    const r = await post("/api/resume");
    if (!r.ok) alert(r.msg);
  };
  $("btn-step").onclick = async () => {
    const r = await post("/api/step");
    if (!r.ok) alert(r.msg);
  };
  $("btn-stop").onclick = async () => {
    const r = await post("/api/stop");
    if (!r.ok) alert(r.msg);
  };
  $("btn-save").onclick = async () => {
    const r = await post("/api/save", { name: $("in-save-name").value });
    alert(r.msg);
    await poll();
  };
  $("btn-load").onclick = async () => {
    const name = $("sel-saves").value;
    if (!name) { alert("先在列表里选一个检查点"); return; }
    const r = await post("/api/load", { name });
    alert(r.msg);
    await poll();
  };
  $("btn-reload").onclick = () => poll();

  $("btn-json").onclick = () => {
    const box = $("json-box");
    box.classList.toggle("hidden");
    if (!box.classList.contains("hidden") && state && state.cfg) {
      $("in-json").value = JSON.stringify(state.cfg, null, 1);
    }
  };
  $("btn-json-apply").onclick = () => {
    try {
      applyCfgToForm(JSON.parse($("in-json").value));
    } catch (e) {
      alert("JSON 解析失败：" + e.message);
    }
  };

  // ---- 文件浏览（系统原生对话框，由 server 弹 tkinter 对话框）----
  $("btn-browse-mesh").onclick = () => pickFile("mesh",
    "选择电机网格文件 (.msh)");
  $("btn-browse-csv").onclick = () => pickFile("csv",
    "选择 B-H 磁化曲线 CSV");
}

// 用当前输入值所在目录作为系统对话框的初始目录
function initialDirOf(target) {
  const cur = ($("in-" + target).value || "").trim();
  const idx = Math.max(cur.lastIndexOf("\\"), cur.lastIndexOf("/"));
  return idx >= 0 ? cur.slice(0, idx) : "";
}

async function pickFile(target, title) {
  const btn = $("btn-browse-" + target);
  const old = btn.textContent;
  btn.textContent = "选择中…";
  btn.disabled = true;
  try {
    const r = await fetchJSON("/api/browse-native", {
      method: "POST",
      body: JSON.stringify({ title, dir: initialDirOf(target) }),
      headers: { "Content-Type": "application/json" },
    });
    if (!r.ok) { alert(r.error); return; }
    if (!r.cancelled && r.path) $("in-" + target).value = r.path;
  } catch (e) {
    alert("文件选择失败：" + e.message);
  } finally {
    btn.textContent = old;
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------- 初始化
function initInputs() {
  $("in-mesh").value = state && state.mesh_path ? state.mesh_path
    : "legacy/pytorch/my_gmsh/model_mirror4.msh";
  $("in-csv").value = state && state.csv_path ? state.csv_path
    : "legacy/pytorch/50CS470.csv";
  if (state && state.cfg && !$("in-steps").value) {
    $("in-steps").value = state.cfg.mma.max_step;
  }
}

wireButtons();
initInputs();
poll();
setInterval(poll, 700);
window.addEventListener("resize", () => {
  if (geomReady) drawGeometry(state && state.rho);
  drawTrajectory();
});
