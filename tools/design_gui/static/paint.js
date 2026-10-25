"use strict";
/* tools/design_gui/static/paint.js —— 密度编辑模型（画笔/形状/撤销重做）。 */

class Editor {
  constructor(geo) {
    this.geo = geo;
    this.rho = new Float32Array(geo.nMaster).fill(1.0);
    this.undoStack = [];
    this.redoStack = [];
    this.maxUndo = 120;
  }

  reset(geo) {
    this.geo = geo;
    this.rho = new Float32Array(geo.nMaster).fill(1.0);
    this.undoStack = [];
    this.redoStack = [];
  }

  snapshot() {
    if (this.undoStack.length >= this.maxUndo) this.undoStack.shift();
    this.undoStack.push(new Float32Array(this.rho));
    this.redoStack.length = 0;
  }

  undo() {
    if (!this.undoStack.length) return false;
    this.redoStack.push(new Float32Array(this.rho));
    this.rho = this.undoStack.pop();
    return true;
  }

  redo() {
    if (!this.redoStack.length) return false;
    this.undoStack.push(new Float32Array(this.rho));
    this.rho = this.redoStack.pop();
    return true;
  }

  setAll(v) {
    this.snapshot();
    this.rho.fill(v);
  }

  /* ------------------------------------------------ 内部：遍历设计单元 */
  _apply(test, value, opts) {
    const g = this.geo;
    const n = (opts && opts.slaves === false) ? g.nMaster : g.nDesign;
    let changed = 0;
    for (let k = 0; k < n; k++) {
      const x = g.cDesign[2 * k], y = g.cDesign[2 * k + 1];
      if (!test(x, y)) continue;
      const idx = g.designMaster[k];
      if (this.rho[idx] !== value) {
        this.rho[idx] = value;
        changed++;
      }
    }
    return changed;
  }

  applyBrush(cx, cy, r, value, opts) {
    const r2 = r * r;
    return this._apply(
      (x, y) => (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2,
      value, opts);
  }

  applySegment(ax, ay, bx, by, r, value, opts) {
    const dx = bx - ax, dy = by - ay;
    const L2 = dx * dx + dy * dy;
    return this._apply((x, y) => {
      let t = L2 > 0 ? ((x - ax) * dx + (y - ay) * dy) / L2 : 0;
      t = Math.max(0, Math.min(1, t));
      const px = ax + t * dx, py = ay + t * dy;
      return (x - px) * (x - px) + (y - py) * (y - py) <= r * r;
    }, value, opts);
  }

  applyRect(ax, ay, bx, by, value, opts) {
    const x0 = Math.min(ax, bx), x1 = Math.max(ax, bx);
    const y0 = Math.min(ay, by), y1 = Math.max(ay, by);
    return this._apply((x, y) => x >= x0 && x <= x1 && y >= y0 && y <= y1,
      value, opts);
  }

  applyCircle(cx, cy, r, value, opts) {
    const r2 = r * r;
    return this._apply(
      (x, y) => (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2,
      value, opts);
  }

  /**
   * 圆环扇区：a 为内半径/起始角上的点，b 为外半径/终止角上的点。
   * 两角几乎相同时用 ``sweepDeg`` 在 a 的角两侧展开。
   */
  applyRing(ax, ay, bx, by, sweepDeg, value, opts) {
    const [r1, a1] = polar(ax, ay);
    const [r2, a2] = polar(bx, by);
    const rin = Math.min(r1, r2), rout = Math.max(r1, r2);
    let s = Math.min(a1, a2), e = Math.max(a1, a2);
    if (e - s < 1e-6) {
      const half = Math.max(1, sweepDeg) / 2;
      s = a1 - half; e = a1 + half;
    }
    const wrap = (e - s) > 180;   // 跨 0° 时取补角
    return this._apply((x, y) => {
      const r = Math.hypot(x, y);
      if (r < rin || r > rout) return false;
      let ang = (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
      if (wrap) {
        ang = (ang + 180) % 360;
        const ss = (s + 180) % 360, ee = (e + 180) % 360;
        return ang >= Math.min(ss, ee) && ang <= Math.max(ss, ee);
      }
      return ang >= s && ang <= e;
    }, value, opts);
  }

  applyPolygon(points, value, opts) {
    const n = points.length;
    if (n < 3) return 0;
    return this._apply((x, y) => {
      let inside = false;
      for (let i = 0, j = n - 1; i < n; j = i++) {
        const xi = points[i][0], yi = points[i][1];
        const xj = points[j][0], yj = points[j][1];
        if (((yi > y) !== (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
          inside = !inside;
        }
      }
      return inside;
    }, value, opts);
  }
}
