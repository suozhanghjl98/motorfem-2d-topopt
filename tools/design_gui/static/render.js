"use strict";
/* tools/design_gui/static/render.js —— 视图变换、几何渲染、叠加层。 */

/* viridis 近似（12 个采样点线性插值；视觉上与 matplotlib viridis 接近） */
const VIRIDIS_STOPS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
  [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
  [180, 222, 44], [253, 231, 37],
];
const VIRIDIS = (() => {
  const out = new Array(256);
  for (let i = 0; i < 256; i++) {
    const t = i / 255 * (VIRIDIS_STOPS.length - 1);
    const j = Math.min(Math.floor(t), VIRIDIS_STOPS.length - 2);
    const f = t - j;
    const a = VIRIDIS_STOPS[j], b = VIRIDIS_STOPS[j + 1];
    out[i] = [
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f),
    ];
  }
  return out;
})();

function viridisCss(v) {
  let t = Number(v);
  if (!Number.isFinite(t)) t = 0;
  t = Math.max(0, Math.min(1, t));
  const c = VIRIDIS[Math.round(t * 255)];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

class View {
  constructor() {
    this.scale = 4;   // px / mm（CSS 像素）
    this.tx = 0;
    this.ty = 0;
    this.w = 1;
    this.h = 1;
  }
  fit(bbox, w, h) {
    this.w = Math.max(1, w);
    this.h = Math.max(1, h);
    const [x0, y0, x1, y1] = bbox;
    const bw = Math.max(x1 - x0, 1e-9);
    const bh = Math.max(y1 - y0, 1e-9);
    this.scale = Math.min(this.w * 0.92 / bw, this.h * 0.92 / bh);
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    this.tx = this.w / 2 - this.scale * cx;
    this.ty = this.h / 2 + this.scale * cy;
  }
  fitRange(x0, y0, x1, y1) {
    this.fit([x0, y0, x1, y1], this.w, this.h);
  }
  toScreen(x, y) {
    return [this.tx + this.scale * x, this.ty - this.scale * y];
  }
  toWorld(px, py) {
    return [(px - this.tx) / this.scale, (this.ty - py) / this.scale];
  }
  zoomAt(px, py, factor) {
    const [wx, wy] = this.toWorld(px, py);
    this.scale = Math.max(0.4, Math.min(400, this.scale * factor));
    this.tx = px - this.scale * wx;
    this.ty = py + this.scale * wy;
  }
  panBy(dx, dy) {
    this.tx += dx;
    this.ty += dy;
  }
}

class Renderer {
  constructor(canvases, geo, view) {
    this.bg = canvases.bg;
    this.fg = canvases.fg;
    this.ov = canvases.ov;
    this.geo = geo;
    this.view = view || new View();
    this.dpr = Math.max(1, window.devicePixelRatio || 1);
    this.basePaths = null;
    this._buildBasePaths();
  }

  setGeometry(geo) {
    this.geo = geo;
    this._buildBasePaths();
  }

  _buildBasePaths() {
    const g = this.geo;
    const paths = {
      0: new Path2D(), 1: new Path2D(), 2: new Path2D(),
    };
    const xy = g.xy, tri = g.tri;
    for (let e = 0; e < g.nElem; e++) {
      const cid = g.colorId[e];
      if (cid === 3) continue;
      const p = paths[cid] || paths[0];
      const a = tri[3 * e], b = tri[3 * e + 1], c = tri[3 * e + 2];
      p.moveTo(xy[2 * a], xy[2 * a + 1]);
      p.lineTo(xy[2 * b], xy[2 * b + 1]);
      p.lineTo(xy[2 * c], xy[2 * c + 1]);
      p.closePath();
    }
    this.basePaths = paths;
  }

  resize() {
    const wrap = this.bg.parentElement;
    const w = Math.max(1, wrap.clientWidth);
    const h = Math.max(1, wrap.clientHeight);
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    this.dpr = dpr;
    for (const c of [this.bg, this.fg, this.ov]) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    this.view.w = w;
    this.view.h = h;
    return [w, h];
  }

  _transform(ctx) {
    const v = this.view, d = this.dpr;
    ctx.setTransform(d * v.scale, 0, 0, -d * v.scale, d * v.tx, d * v.ty);
    ctx.imageSmoothingEnabled = false;
  }

  drawBase() {
    const ctx = this.bg.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.bg.width, this.bg.height);
    this._transform(ctx);
    const colors = this.geo.baseColors;
    for (const cid of [0, 1, 2]) {
      const p = this.basePaths[cid];
      ctx.fillStyle = colors[String(cid)] || "rgb(140,140,148)";
      ctx.fill(p);
    }
    // 外轮廓：轻描一下，避免深色背景下边界发虚
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 0.2 / this.view.scale;
    for (const cid of [0, 1, 2]) ctx.stroke(this.basePaths[cid]);
  }

  /**
   * 设计层（主 + 镜像副本）。
   * @param {Float32Array} rho 主单元密度（长度 nMaster）
   * @param {{showSlaves?:boolean, alpha?:number}} opts
   */
  drawDesign(rho, opts = {}) {
    const g = this.geo;
    const ctx = this.fg.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.fg.width, this.fg.height);
    this._transform(ctx);
    const showSlaves = opts.showSlaves !== false;
    const alpha = opts.alpha === undefined ? 1 : opts.alpha;
    const n = showSlaves ? g.nDesign : g.nMaster;
    const lw = 0.45 / this.view.scale;
    ctx.globalAlpha = alpha;
    for (let k = 0; k < n; k++) {
      const v = rho[g.designMaster[k]];
      const col = viridisCss(v);
      const o = 6 * k;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.moveTo(g.dtri[o], g.dtri[o + 1]);
      ctx.lineTo(g.dtri[o + 2], g.dtri[o + 3]);
      ctx.lineTo(g.dtri[o + 4], g.dtri[o + 5]);
      ctx.closePath();
      ctx.fill();
      // 同色描边消除三角形之间的发丝缝
      if (lw > 0.01) {
        ctx.strokeStyle = col;
        ctx.lineWidth = lw;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  /**
   * 叠加层：主扇区虚线、悬停轮廓、笔刷圆、形状预览、被滤波抹掉的单元。
   */
  drawOverlay(opts = {}) {
    const g = this.geo;
    const v = this.view;
    const ctx = this.ov.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.ov.width, this.ov.height);
    this._transform(ctx);
    const px = 1.0 / v.scale;      // 1 CSS px 对应的世界长度

    // 主扇区边界（0° 与 45° 径向虚线）
    ctx.save();
    ctx.setLineDash([6 * px, 5 * px]);
    ctx.strokeStyle = "rgba(120,200,255,0.55)";
    ctx.lineWidth = 1.1 * px;
    for (const ang of g.sector) {
      const r = g.maxRadius * 1.02;
      const t = ang * Math.PI / 180;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(r * Math.cos(t), r * Math.sin(t));
      ctx.stroke();
    }
    ctx.restore();

    // 被滤波抹掉的细节
    if (opts.lost && opts.lost.length) {
      ctx.strokeStyle = "rgba(255,80,90,0.85)";
      ctx.lineWidth = 1.2 * px;
      ctx.beginPath();
      for (let i = 0; i < g.nMaster; i++) {
        if (!opts.lost[i]) continue;
        const o = 6 * i;
        ctx.moveTo(g.dtri[o], g.dtri[o + 1]);
        ctx.lineTo(g.dtri[o + 2], g.dtri[o + 3]);
        ctx.lineTo(g.dtri[o + 4], g.dtri[o + 5]);
        ctx.closePath();
      }
      ctx.stroke();
    }

    // 悬停单元
    if (opts.hover >= 0) {
      const o = 6 * opts.hover;
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 1.6 * px;
      ctx.beginPath();
      ctx.moveTo(g.dtri[o], g.dtri[o + 1]);
      ctx.lineTo(g.dtri[o + 2], g.dtri[o + 3]);
      ctx.lineTo(g.dtri[o + 4], g.dtri[o + 5]);
      ctx.closePath();
      ctx.stroke();
    }

    // 形状预览
    const sh = opts.shape;
    if (sh) this._drawShape(ctx, sh, px);
    if (opts.polygon) this._drawPolygon(ctx, opts.polygon, px);

    // 笔刷光标
    if (opts.brush) {
      const {x, y, r} = opts.brush;
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineWidth = 1.2 * px;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fill();
    }
  }

  _drawShape(ctx, sh, px) {
    ctx.save();
    ctx.setLineDash([5 * px, 4 * px]);
    ctx.strokeStyle = "rgba(255,230,120,0.95)";
    ctx.lineWidth = 1.4 * px;
    if (sh.type === "line") {
      ctx.beginPath();
      ctx.moveTo(sh.a[0], sh.a[1]);
      ctx.lineTo(sh.b[0], sh.b[1]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = "rgba(255,230,120,0.45)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(sh.a[0], sh.a[1], sh.r, 0, Math.PI * 2);
      ctx.stroke();
    } else if (sh.type === "rect") {
      const x0 = Math.min(sh.a[0], sh.b[0]), x1 = Math.max(sh.a[0], sh.b[0]);
      const y0 = Math.min(sh.a[1], sh.b[1]), y1 = Math.max(sh.a[1], sh.b[1]);
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    } else if (sh.type === "circle") {
      const r = Math.hypot(sh.b[0] - sh.a[0], sh.b[1] - sh.a[1]);
      ctx.beginPath();
      ctx.arc(sh.a[0], sh.a[1], r, 0, Math.PI * 2);
      ctx.stroke();
    } else if (sh.type === "ring") {
      const [r1, a1] = polar(sh.a[0], sh.a[1]);
      const [r2, a2] = polar(sh.b[0], sh.b[1]);
      const rin = Math.min(r1, r2), rout = Math.max(r1, r2);
      let start = Math.min(a1, a2), end = Math.max(a1, a2);
      if (Math.abs(end - start) < 1e-6) { start = a1 - 22.5; end = a1 + 22.5; }
      ctx.beginPath();
      ctx.arc(0, 0, rout, start * Math.PI / 180, end * Math.PI / 180);
      ctx.arc(0, 0, rin, end * Math.PI / 180, start * Math.PI / 180, true);
      ctx.closePath();
      ctx.stroke();
    }
    ctx.restore();
  }

  _drawPolygon(ctx, pts, px) {
    if (!pts || !pts.length) return;
    ctx.save();
    ctx.setLineDash([5 * px, 4 * px]);
    ctx.strokeStyle = "rgba(255,230,120,0.95)";
    ctx.lineWidth = 1.4 * px;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.restore();
  }
}

function polar(x, y) {
  return [Math.hypot(x, y), (Math.atan2(y, x) * 180 / Math.PI + 360) % 360];
}
