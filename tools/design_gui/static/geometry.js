"use strict";
/* tools/design_gui/static/geometry.js —— 几何解码、质心、命中测试（空间索引）。 */

class Geometry {
  constructor(p) {
    this.p = p;
    this.motor = p.motor;
    this.sha = p.mesh && p.mesh.sha256 ? p.mesh.sha256 : "";
    this.nNode = p.n_node;
    this.nElem = p.n_elem;
    this.nMaster = p.n_master;
    this.nSlave = p.n_slave;
    this.nDesign = p.n_design;

    // xy: [[x,y],...] mm；tri: [[a,b,c],...]
    this.xy = new Float32Array(this.nNode * 2);
    for (let i = 0; i < this.nNode; i++) {
      this.xy[2 * i] = p.xy[i][0];
      this.xy[2 * i + 1] = p.xy[i][1];
    }
    this.tri = new Int32Array(this.nElem * 3);
    for (let e = 0; e < this.nElem; e++) {
      this.tri[3 * e] = p.tri[e][0];
      this.tri[3 * e + 1] = p.tri[e][1];
      this.tri[3 * e + 2] = p.tri[e][2];
    }
    this.colorId = Int32Array.from(p.color_id);

    this.masterCells = Int32Array.from(p.master_cells);
    this.slaveCells = Int32Array.from(p.slave_cells);
    this.designCells = Int32Array.from(p.design_cells);
    this.designMaster = Int32Array.from(p.design_master);

    this.areaMaster = Float32Array.from(p.area_master);
    this.radiiMaster = Float32Array.from(p.radii_master);
    this.anglesMasterDeg = Float32Array.from(p.angles_master_deg);
    this.sector = Float64Array.from(p.master_sector_deg || [0, 45]);
    this.identityRadiusMm = Number(p.identity_filter_radius_mm);
    this.defaultRadiusMm = Number(p.default_filter_radius_mm);
    this.baseColors = p.base_colors || {};

    // 主单元质心（payload 直接给，单位 mm）
    this.cMaster = new Float32Array(this.nMaster * 2);
    for (let i = 0; i < this.nMaster; i++) {
      this.cMaster[2 * i] = p.centroids_master[i][0];
      this.cMaster[2 * i + 1] = p.centroids_master[i][1];
    }
    // 设计单元（含镜像）三角形坐标与质心
    this.dtri = new Float32Array(this.nDesign * 6);
    this.cDesign = new Float32Array(this.nDesign * 2);
    for (let k = 0; k < this.nDesign; k++) {
      const e = this.designCells[k];
      const a = this.tri[3 * e], b = this.tri[3 * e + 1], c = this.tri[3 * e + 2];
      const x1 = this.xy[2 * a], y1 = this.xy[2 * a + 1];
      const x2 = this.xy[2 * b], y2 = this.xy[2 * b + 1];
      const x3 = this.xy[2 * c], y3 = this.xy[2 * c + 1];
      this.dtri.set([x1, y1, x2, y2, x3, y3], 6 * k);
      this.cDesign[2 * k] = (x1 + x2 + x3) / 3;
      this.cDesign[2 * k + 1] = (y1 + y2 + y3) / 3;
    }

    // 全模型包围盒（mm）
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    for (let n = 0; n < this.nNode; n++) {
      const x = this.xy[2 * n], y = this.xy[2 * n + 1];
      if (x < minx) minx = x;
      if (x > maxx) maxx = x;
      if (y < miny) miny = y;
      if (y > maxy) maxy = y;
    }
    this.bbox = [minx, miny, maxx, maxy];
    this.maxRadius = Math.max(
      Math.hypot(minx, miny), Math.hypot(minx, maxy),
      Math.hypot(maxx, miny), Math.hypot(maxx, maxy));

    this._buildGrid();
  }

  designCentroid(k) {
    return [this.cDesign[2 * k], this.cDesign[2 * k + 1]];
  }

  masterCentroid(i) {
    return [this.cMaster[2 * i], this.cMaster[2 * i + 1]];
  }

  /* -------------------------------------------------- 均匀网格空间索引 */
  _buildGrid() {
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    const n = this.nDesign;
    for (let k = 0; k < n; k++) {
      const o = 6 * k;
      for (let v = 0; v < 3; v++) {
        const x = this.dtri[o + 2 * v], y = this.dtri[o + 2 * v + 1];
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
      }
    }
    const span = Math.max(maxx - minx, maxy - miny, 1e-9);
    this.cell = Math.max(span / 96, 0.4);
    this.gx0 = minx;
    this.gy0 = miny;
    this.nx = Math.max(1, Math.ceil((maxx - minx) / this.cell) + 1);
    this.ny = Math.max(1, Math.ceil((maxy - miny) / this.cell) + 1);
    this.grid = new Array(this.nx * this.ny);
    for (let k = 0; k < n; k++) {
      const o = 6 * k;
      let x1 = this.dtri[o], y1 = this.dtri[o + 1];
      let x2 = this.dtri[o + 2], y2 = this.dtri[o + 3];
      let x3 = this.dtri[o + 4], y3 = this.dtri[o + 5];
      const bx0 = Math.min(x1, x2, x3), bx1 = Math.max(x1, x2, x3);
      const by0 = Math.min(y1, y2, y3), by1 = Math.max(y1, y2, y3);
      const i0 = Math.max(0, Math.floor((bx0 - minx) / this.cell));
      const i1 = Math.min(this.nx - 1, Math.floor((bx1 - minx) / this.cell));
      const j0 = Math.max(0, Math.floor((by0 - miny) / this.cell));
      const j1 = Math.min(this.ny - 1, Math.floor((by1 - miny) / this.cell));
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          const idx = j * this.nx + i;
          (this.grid[idx] || (this.grid[idx] = [])).push(k);
        }
      }
    }
  }

  hitTest(x, y) {
    const i = Math.floor((x - this.gx0) / this.cell);
    const j = Math.floor((y - this.gy0) / this.cell);
    if (i < 0 || j < 0 || i >= this.nx || j >= this.ny) return -1;
    const arr = this.grid[j * this.nx + i];
    if (!arr) return -1;
    for (let t = 0; t < arr.length; t++) {
      const k = arr[t];
      const o = 6 * k;
      if (pointInTri(x, y,
        this.dtri[o], this.dtri[o + 1],
        this.dtri[o + 2], this.dtri[o + 3],
        this.dtri[o + 4], this.dtri[o + 5])) return k;
    }
    return -1;
  }

  /* 世界坐标 (mm) → 半径/角度（度，0~360） */
  polar(x, y) {
    return [Math.hypot(x, y), (Math.atan2(y, x) * 180 / Math.PI + 360) % 360];
  }
}

function pointInTri(px, py, x1, y1, x2, y2, x3, y3) {
  const d1 = (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2);
  const d2 = (px - x3) * (y2 - y3) - (x2 - x3) * (py - y3);
  const d3 = (px - x1) * (y3 - y1) - (x3 - x1) * (py - y1);
  const neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
  const pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
  return !(neg && pos);
}
