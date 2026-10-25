# -*- coding: utf-8 -*-
"""verify_gui.py —— 不打开浏览器，验证网页 GUI 的三个关键接口。

用途：CI / 远程机器上确认"GUI 真的能用"，判据依次是
  1. ``GET /``            返回页面（含标题，说明静态资源在）
  2. ``GET /api/geometry`` 几何已构建完成（单元数 > 0）
  3. ``POST /api/export``  把手绘密度导出成 NPZ，且 ``rho_master`` 回读一致

前置：目标机器上已经跑起手绘 GUI（另开一个终端）::

    python -u tools/design_gui/server.py --host 127.0.0.1 --port 8766

然后::

    python -u tools/scripts/verify_gui.py --url http://127.0.0.1:8766

默认（带 token）模式下脚本会自己从 ``GET /`` 的 ``<meta name="design-gui-token">``
里取 token；``--no-token`` 启动的服务与 ``--token`` 手填两种用法同样支持。

退出码 0 = 三项全过。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _get(base: str, path: str, token: str = "") -> tuple[int, bytes]:
    req = urllib.request.Request(base.rstrip("/") + path)
    if token:
        req.add_header("X-Design-GUI-Token", token)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.read()


def _post(base: str, path: str, body: dict, token: str = "") -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    if token:
        req.add_header("X-Design-GUI-Token", token)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8766")
    ap.add_argument("--npz", default=str(REPO / "examples" / "hand_design.npz"),
                    help="拿这个 NPZ 的 rho_master 当手绘输入")
    ap.add_argument("--radius-mm", type=float, default=0.9)
    ap.add_argument("--token", default="")
    ap.add_argument("--wait-s", type=float, default=120.0,
                    help="等几何构建完成的最长秒数")
    a = ap.parse_args()

    ok = True

    code, body = _get(a.url, "/")
    has_title = b"<title>" in body
    print(f"[1/3] GET /                -> {code}, {len(body)} bytes, title={has_title}")
    ok &= (code == 200 and has_title)

    # 默认（带 token）模式下，页面会把 token 写进 <meta name="design-gui-token">；
    # 这里自动取出来，省得再手动抄一次。
    token = a.token
    if not token:
        m = re.search(rb'design-gui-token"\s+content="([^"]*)"', body)
        if m:
            token = m.group(1).decode("utf-8", "replace")
    print(f"      token={'（无，--no-token 模式）' if not token else token[:8] + '…'}")

    t0 = time.time()
    n_elem = n_master = 0
    while True:
        _, raw = _get(a.url, "/api/geometry")
        g = json.loads(raw.decode("utf-8"))
        geo = g.get("geometry") or {}
        n_elem = int(geo.get("n_elem") or 0)
        n_master = int(geo.get("n_master") or 0)
        if n_elem or time.time() - t0 > a.wait_s:
            break
        time.sleep(2.0)
    print(f"[2/3] GET /api/geometry    -> 全场 {n_elem} 单元，可编辑主域 {n_master} 个"
          f"（{time.time() - t0:.1f} s 等待）")
    ok &= (n_elem > 0 and n_master > 0)

    z = np.load(a.npz, allow_pickle=True)
    rho = np.asarray(z["rho_master"], float)
    res = _post(a.url, "/api/export", {
        "motor": "6s4p", "rho_master": rho.tolist(), "radius_mm": a.radius_mm,
        "x_mode": "direct", "name": "verify_gui", "ladder": "S0:T=40,R=80",
    }, token=token)
    p = Path(res["path"])
    if not p.is_absolute():
        p = REPO / p
    back = np.load(p, allow_pickle=True)["rho_master"]
    same = np.allclose(back, rho)
    print(f"[3/3] POST /api/export     -> {p.name}, {p.stat().st_size/1024:.1f} KB, "
          f"rho_master 回读一致={same}")
    ok &= bool(res.get("ok")) and same

    print("=== GUI 验证通过 ===" if ok else "=== GUI 验证失败 ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
