#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/design_gui/server.py —— 手绘转子初始结构的网页 GUI（stdlib，零依赖）。

启动::

    PYTHONIOENCODING=utf-8 python -u tools/design_gui/server.py --port 8766

然后浏览器打开 http://127.0.0.1:8766 。

架构
----
- ``core.py``   ：拓扑/几何/滤波/NPZ（纯逻辑，可单测）
- ``jobs.py``   ：优化子进程（复用 ``run_auto_schedule.py``，不复制数值路径）
- 本文件        ：ThreadingHTTPServer + 静态页 + 后台构建线程

⚠ 几何构建（``build_field_topology``）约 8 s，放在后台线程；页面先显示
“构建中”，完成后自动拉取。本服务**不做任何 FEM 求解**；优化在独立子进程里跑，
所以 GUI 与优化不会同时争用 CHOLMOD。

⚠ 安全默认：只绑定 127.0.0.1；POST 需要页面注入的 token。若要 ``--host 0.0.0.0``，
启动时会打印警告。
"""

from __future__ import annotations

import argparse
import gzip
import json
import secrets
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

import core  # noqa: E402
import jobs  # noqa: E402

_STATIC = Path(__file__).parent / "static"
_DIALOG = _REPO / "tools" / "optim_gui" / "file_dialog.py"
_JSON_LIMIT = 8 << 20


# ---------------------------------------------------------------------------
# 后台几何构建
# ---------------------------------------------------------------------------
class BundleState:
    def __init__(self, motor: str):
        self.motor = motor
        self.bundle: Optional[core.Bundle] = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.thread: Optional[threading.Thread] = None
        self.version = 0

    def snapshot(self) -> Dict[str, Any]:
        building = (self.bundle is None and self.error is None
                    and self.thread is not None and self.thread.is_alive())
        end = self.finished_at if self.finished_at is not None else time.time()
        elapsed = (end - self.started_at) if self.started_at else 0.0
        return {
            "motor": self.motor,
            "building": bool(building),
            "ready": self.bundle is not None,
            "error": self.error,
            "elapsed_s": round(elapsed, 2),
            "version": self.version,
        }


class BundleStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: Dict[str, BundleState] = {}

    def state(self, motor: str) -> BundleState:
        with self._lock:
            st = self._states.get(motor)
            if st is None:
                st = BundleState(motor)
                self._states[motor] = st
            return st

    def ensure(self, motor: str) -> BundleState:
        st = self.state(motor)
        with self._lock:
            if st.bundle is not None:
                return st
            if st.thread is not None and st.thread.is_alive():
                return st
            st.error = None
            st.started_at = time.time()
            st.finished_at = None
            st.thread = threading.Thread(
                target=self._build, args=(motor,), name=f"build-{motor}",
                daemon=True)
            st.thread.start()
        return st

    def _build(self, motor: str) -> None:
        st = self.state(motor)
        try:
            b = core.build_bundle(motor)
            with self._lock:
                st.bundle = b
                st.error = None
                st.version += 1
        except Exception:  # noqa: BLE001
            with self._lock:
                st.error = traceback.format_exc()[-4000:]
        finally:
            st.finished_at = time.time()

    def get(self, motor: str) -> Optional[core.Bundle]:
        st = self.state(motor)
        return st.bundle


# ---------------------------------------------------------------------------
# 应用状态
# ---------------------------------------------------------------------------
class App:
    def __init__(self, *, host: str, port: int, motor: str,
                 out_dir: Path, token: str):
        self.host = host
        self.port = port
        self.default_motor = motor
        self.out_dir = Path(out_dir)
        self.token = token
        self.store = BundleStore()
        self.job = jobs.JobManager()

    # ---------------------------------------------------------------- 辅助
    def motor_of(self, q: Optional[Dict[str, str]] = None, body: Optional[dict] = None
                 ) -> str:
        if body and body.get("motor"):
            return str(body["motor"])
        if q and q.get("motor"):
            return str(q["motor"][0])
        return self.default_motor

    def require_bundle(self, motor: str) -> core.Bundle:
        b = self.store.get(motor)
        if b is None:
            st = self.store.state(motor)
            if st.error:
                raise RuntimeError("几何构建失败：\n" + st.error)
            self.store.ensure(motor)
            raise RuntimeError("几何仍在构建中（约 8 s），请稍后重试")
        return b


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    app: App = None  # type: ignore[assignment]
    server_version = "DesignGUI/1.0"

    # ---------------------------------------------------------------- 基础
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        # 静默访问日志（避免刷屏）；错误仍由 _error 输出。
        return

    def _send_json(self, code: int, obj: Dict[str, Any],
                   extra: Optional[Dict[str, str]] = None) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, msg: str, *, exc: Optional[BaseException] = None
               ) -> None:
        payload: Dict[str, Any] = {"ok": False, "error": msg}
        if exc is not None:
            payload["traceback"] = traceback.format_exc()[-3000:]
        self._send_json(code, payload)

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return b""
        if n > _JSON_LIMIT:
            raise ValueError(f"请求体过大：{n} bytes（上限 {_JSON_LIMIT}）")
        return self.rfile.read(n)

    def _json_body(self) -> Dict[str, Any]:
        raw = self._read_body()
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _check_token(self) -> bool:
        if not self.app.token:          # --no-token 调试模式
            return True
        tok = self.headers.get("X-Design-GUI-Token", "")
        return secrets.compare_digest(tok, self.app.token)

    # ---------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self._serve_index()
            elif u.path.startswith("/static/"):
                self._serve_static(u.path[len("/static/"):])
            elif u.path == "/api/geometry":
                self._api_geometry(q)
            elif u.path == "/api/state":
                self._api_state(q)
            elif u.path == "/api/designs":
                self._api_designs(q)
            elif u.path == "/api/job/log":
                off = int((q.get("offset") or ["0"])[0])
                self._send_json(200, self.app.job.log_delta(off))
            elif u.path == "/api/health":
                self._send_json(200, {"ok": True, "time": time.time()})
            else:
                self._error(404, f"未知路径 {u.path}")
        except Exception as exc:  # noqa: BLE001
            self._error(500, f"{type(exc).__name__}: {exc}", exc=exc)

    # ---------------------------------------------------------------- POST
    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if not self._check_token():
            self._send_json(403, {"ok": False, "error": "token 缺失或错误"})
            return
        try:
            body = self._json_body()
            if u.path == "/api/preview":
                self._api_preview(body, q)
            elif u.path == "/api/export":
                self._api_export(body)
            elif u.path == "/api/import/inspect":
                self._api_import_inspect(body)
            elif u.path == "/api/import/apply":
                self._api_import_apply(body)
            elif u.path == "/api/job/start":
                self._api_job_start(body)
            elif u.path == "/api/job/stop":
                self._send_json(200, self.app.job.stop())
            elif u.path == "/api/browse-native":
                self._api_browse(body)
            else:
                self._error(404, f"未知路径 {u.path}")
        except Exception as exc:  # noqa: BLE001
            self._error(400, f"{type(exc).__name__}: {exc}", exc=exc)

    # ---------------------------------------------------------------- 静态
    def _serve_index(self) -> None:
        p = _STATIC / "index.html"
        text = p.read_text(encoding="utf-8")
        text = text.replace("__DESIGN_GUI_TOKEN__", self.app.token)
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, rel: str) -> None:
        rel = rel.split("?", 1)[0]
        if not rel or rel.startswith(".") or ".." in Path(rel).parts:
            self._error(403, "非法静态路径")
            return
        p = (_STATIC / rel).resolve()
        try:
            p.relative_to(_STATIC.resolve())
        except ValueError:
            self._error(403, "非法静态路径")
            return
        if not p.is_file():
            self._error(404, f"文件不存在：{rel}")
            return
        mime = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".json": "application/json; charset=utf-8",
        }.get(p.suffix.lower(), "application/octet-stream")
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ---------------------------------------------------------------- API
    def _api_geometry(self, q: Dict[str, list]) -> None:
        motor = self.app.motor_of(q)
        st = self.app.store.state(motor)
        if st.bundle is None and not st.error:
            self.app.store.ensure(motor)
            snap = st.snapshot()
            self._send_json(200, {"ok": True, **snap, "geometry": None})
            return
        if st.bundle is None:
            self._send_json(500, {"ok": False, "error": "几何构建失败",
                                  "building": False, "traceback": st.error})
            return
        accept = (self.headers.get("Accept-Encoding") or "").lower()
        use_gz = "gzip" in accept
        raw = core.geometry_bytes(st.bundle, use_gzip=False)
        wrapped = (b'{"ok":true,"building":false,"ready":true,"version":'
                   + str(int(st.version)).encode()
                   + b',"geometry":' + raw + b'}')
        data = gzip.compress(wrapped, 6) if use_gz else wrapped
        etag = f'"{st.bundle.mesh_sha256[:16]}-{st.version}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("ETag", etag)
        if use_gz:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(data)

    def _api_state(self, q: Dict[str, list]) -> None:
        motor = self.app.motor_of(q)
        st = self.app.store.state(motor)
        b = st.bundle
        self._send_json(200, {
            "ok": True,
            "build": st.snapshot(),
            "job": self.app.job.state(),
            "defaults": {
                "motor": self.app.default_motor,
                "filter_radius_mm": (b.default_filter_radius_m * 1e3
                                     if b is not None else 2.0),
                "identity_filter_radius_mm": (b.identity_radius_m * 1e3
                                              if b is not None else 0.2),
                "max_density": (float(b.cfg.topology.max_density)
                                if b is not None else 0.65),
            },
            "motors": {k: v["label"] for k, v in core.MOTORS.items()},
        })

    def _api_preview(self, body: Dict[str, Any], q: Dict[str, list]) -> None:
        motor = self.app.motor_of(q, body)
        b = self.app.require_bundle(motor)
        res = core.compute_start(
            b, body.get("rho_master"), float(body["radius_mm"]) * 1e-3,
            x_mode=str(body.get("x_mode") or "direct"),
            x0_override=body.get("x0_override"))
        self._send_json(200, {
            "ok": True,
            "effective_master": res["rho_eff_master"].tolist(),
            "stats": res["stats"],
            "x_mode": res["x_mode"],
            "filter_radius_mm": res["filter_radius_m"],
            "identity_filter_radius_mm": b.identity_radius_m * 1e3,
        })

    def _api_export(self, body: Dict[str, Any]) -> None:
        motor = self.app.motor_of(body=body)
        b = self.app.require_bundle(motor)
        res = core.export_design(
            b, body.get("rho_master"), float(body["radius_mm"]) * 1e-3,
            x_mode=str(body.get("x_mode") or "direct"),
            name=str(body.get("name") or "design"),
            out_dir=self.app.out_dir,
            ladder=str(body.get("ladder") or "S0:T=50,R=80"),
            x0_override=body.get("x0_override"),
        )
        self._send_json(200, res)

    def _resolve_npz(self, body: Dict[str, Any]) -> Path:
        raw = str(body.get("path") or "").strip()
        if not raw:
            raise ValueError("缺少 path")
        p = Path(raw)
        if not p.is_absolute():
            p = (_REPO / p)
        p = p.resolve()
        if p.suffix.lower() != ".npz":
            raise ValueError("只接受 .npz 文件")
        if not p.is_file():
            raise FileNotFoundError(f"文件不存在：{p}")
        return p

    def _api_import_inspect(self, body: Dict[str, Any]) -> None:
        motor = self.app.motor_of(body=body)
        b = self.app.require_bundle(motor)
        p = self._resolve_npz(body)
        self._send_json(200, core.inspect_npz(b, p))

    def _api_import_apply(self, body: Dict[str, Any]) -> None:
        motor = self.app.motor_of(body=body)
        b = self.app.require_bundle(motor)
        p = self._resolve_npz(body)
        res = core.apply_import(
            b, p, str(body["key"]),
            row=(None if body.get("row") in (None, "", -1) else int(body["row"])),
            as_=str(body.get("as") or "auto"),
            fallback_radius_mm=(None if body.get("fallback_radius_mm") is None
                                else float(body["fallback_radius_mm"])),
        )
        out = dict(res)
        out["rho_master"] = res["rho_master"].tolist()
        if res.get("imported_x") is not None:
            out["imported_x"] = res["imported_x"].tolist()
        self._send_json(200, out)

    def _api_designs(self, q: Dict[str, list]) -> None:
        rows = core.list_designs(self.app.out_dir)
        motor = (q.get("motor") or [None])[0]
        if motor:
            rows = [r for r in rows if r.get("motor") in (None, motor)]
        self._send_json(200, {"ok": True, "designs": rows,
                              "dir": str(self.app.out_dir)})

    def _api_browse(self, body: Dict[str, Any]) -> None:
        title = str(body.get("title") or "选择 NPZ")
        initial = str(body.get("dir") or self.app.out_dir)
        cmd = [sys.executable, "-u", str(_DIALOG),
               "--title", title, "--dir", initial,
               "--filetypes", "NPZ 文件;*.npz;所有文件;*.*"]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        path = (r.stdout or "").strip().splitlines()
        self._send_json(200, {"ok": True, "path": (path[-1] if path else "")})

    def _api_job_start(self, body: Dict[str, Any]) -> None:
        motor = self.app.motor_of(body=body)
        b = self.app.require_bundle(motor)
        radius_mm = float(body.get("radius_mm") or b.default_filter_radius_m * 1e3)
        name = str(body.get("name") or "design")
        # 先保存最新设计（与预览同一套 core 逻辑）
        exp = core.export_design(
            b, body.get("rho_master"), radius_mm * 1e-3,
            x_mode=str(body.get("x_mode") or "direct"),
            name=name, out_dir=self.app.out_dir,
            ladder=str(body.get("stages") or "S0:T=50,R=80"),
            x0_override=body.get("x0_override"),
        )
        spec = jobs.JobSpec(
            npz_path=exp["path"],
            motor=motor,
            stages=(str(body["stages"]) if body.get("stages") else None),
            ladder=(str(body["ladder"]) if body.get("ladder") else None),
            n_train=int(body.get("n_train") or 11),
            n_eval=int(body.get("n_eval") or 6),
            max_eval_per_stage=int(body.get("max_eval_per_stage") or 100),
            mma_move=float(body.get("mma_move") or 0.05),
            filter_radius_mm=radius_mm,
            max_density=(None if body.get("max_density") in (None, "")
                         else float(body["max_density"])),
            strict=bool(body.get("strict")),
            tag=core.safe_slug(name),
            extra_args=str(body.get("extra_args") or ""),
        )
        state = self.app.job.start(spec, slug=core.safe_slug(name))
        self._send_json(200, {
            "ok": True,
            "export": exp,
            "job": state,
            "monitor_cmd": [
                sys.executable, "-u", "tools/optim_gui/server.py",
                "--read-only", "--port", "8765",
                "--log", state["log"], "--npz", str(Path(state["out"]).parent / "run"),
            ],
        })


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--motor", default=core.DEFAULT_MOTOR,
                    choices=sorted(core.MOTORS))
    ap.add_argument("--out-dir", default=str(core.DEFAULT_OUT_DIR),
                    help="导出 NPZ 的目录（默认 tools/_out/design_gui/designs）")
    ap.add_argument("--no-token", action="store_true",
                    help="关闭 POST token 校验（仅本机调试；不推荐）")
    args = ap.parse_args()

    token = "" if args.no_token else secrets.token_urlsafe(24)
    app = App(host=args.host, port=args.port, motor=args.motor,
              out_dir=Path(args.out_dir), token=token)
    Handler.app = app

    # 启动即后台构建默认网格（不阻塞端口）。
    app.store.ensure(args.motor)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"⚠ 警告：绑定 {args.host}，POST 端点可被局域网访问。")
    print(f"手绘设计 GUI：http://{args.host}:{args.port}")
    print(f"默认网格：{core.MOTORS[args.motor]['mesh']}")
    print(f"导出目录：{args.out_dir}")
    print("几何在后台构建（约 8 s）；构建完成后页面自动加载。")
    print("按 Ctrl-C 退出（不会终止已启动的优化子进程）。")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        srv.server_close()
        app.job.shutdown(kill=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
