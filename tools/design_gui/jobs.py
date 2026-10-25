# -*- coding: utf-8 -*-
"""tools/design_gui/jobs.py —— 手绘设计 → 优化子进程的管理。

设计原则
--------
- **不复制优化逻辑**：启动的是既有 ``tools/scripts/run_auto_schedule.py``，
  通过 ``--x0-npz <export.npz> --x0-key x0`` 把手绘结构作为起点。
- **日志契约**：``<out>.log`` 由 ``RunLog`` 按
  ``docs/GUI只读监控产物契约.md`` 写；本模块只按契约增量读取，不猜成功。
- 同一时刻只允许一个 job；stdout/stderr 重定向到 ``<out>.console.log``
  （PIPE 不读会阻塞子进程）。
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
_RUNNER = _REPO / "tools" / "scripts" / "run_auto_schedule.py"
RUN_ROOT = _REPO / "tools" / "_out" / "design_gui" / "runs"

_WIN = (os.name == "nt")

_EVAL_RE = re.compile(r"^EVAL\s+(.*)$")
_KV_RE = re.compile(r"(\w+)=([^\s]+)")


def _split_extra(text: str) -> List[str]:
    """解析自由参数文本（不经过 shell；去掉包裹引号）。"""
    if not text or not text.strip():
        return []
    toks = shlex.split(text, posix=False)
    return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t
            for t in toks]


@dataclass
class JobSpec:
    """启动一次优化的全部参数（由前端/HTTP 层填充）。"""

    npz_path: str
    motor: str = "6s4p"
    stages: Optional[str] = None          # "S0:T=50,R=80|S1:..."
    ladder: Optional[str] = None          # 与 stages 二选一
    n_train: int = 11
    n_eval: int = 6
    max_eval_per_stage: int = 100
    mma_move: float = 0.05
    filter_radius_mm: Optional[float] = None
    max_density: Optional[float] = None
    strict: bool = False
    tag: str = "design"
    out_prefix: Optional[str] = None
    extra_args: str = ""
    csv: Optional[str] = None

    def resolved_out(self, *, slug: str) -> Path:
        if self.out_prefix:
            return Path(self.out_prefix)
        ts = time.strftime("%Y%m%d_%H%M%S")
        return RUN_ROOT / f"{ts}_{slug}" / "run"


def build_command(spec: JobSpec, *, slug: str, python: Optional[str] = None,
                  repo: Path = _REPO) -> Dict[str, Any]:
    """拼出 ``subprocess.Popen`` 的 argv 与输出前缀。"""
    py = python or sys.executable
    out = spec.resolved_out(slug=slug)
    cmd: List[str] = [
        py, "-u", str(repo / "tools" / "scripts" / "run_auto_schedule.py"),
        "--motor", str(spec.motor),
        "--x0-npz", str(spec.npz_path),
        "--x0-key", "x0",
        "--n-train", str(int(spec.n_train)),
        "--n-eval", str(int(spec.n_eval)),
        "--max-eval-per-stage", str(int(spec.max_eval_per_stage)),
        "--mma-move", str(float(spec.mma_move)),
        "--tag", str(spec.tag),
        "--out", str(out),
    ]
    if spec.stages:
        cmd += ["--stages", str(spec.stages)]
    elif spec.ladder:
        cmd += ["--ladder", str(spec.ladder)]
    if spec.filter_radius_mm is not None:
        cmd += ["--filter-radius-mm", str(float(spec.filter_radius_mm))]
    if spec.max_density is not None:
        cmd += ["--max-density", str(float(spec.max_density))]
    if spec.csv:
        cmd += ["--csv", str(spec.csv)]
    if spec.strict:
        cmd += ["--strict"]
    cmd += _split_extra(spec.extra_args)
    return {
        "cmd": cmd,
        "out": out,
        "log": Path(str(out) + ".log"),
        "console": Path(str(out) + ".console.log"),
    }


def _parse_eval_line(line: str) -> Optional[Dict[str, Any]]:
    m = _EVAL_RE.match(line.strip())
    if not m:
        return None
    kv = dict(_KV_RE.findall(m.group(1)))
    row: Dict[str, Any] = {}
    for k in ("stage", "n"):
        if k in kv:
            try:
                row[k] = int(float(kv[k]))
            except ValueError:
                pass
    for k in ("obj", "seconds", "T_flux"):
        if k in kv:
            try:
                row[k] = float(kv[k])
            except ValueError:
                pass
    cons = []
    for i in range(1, 5):
        v = kv.get(f"con{i}")
        try:
            cons.append(None if v in (None, "-", "") else float(v))
        except ValueError:
            cons.append(None)
    row["cons"] = cons
    return row


class JobManager:
    """单实例优化子进程 + 增量日志读取。"""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.spec: Optional[JobSpec] = None
        self.slug: str = ""
        self.paths: Dict[str, Any] = {}
        self.started_at: float = 0.0
        self.console_file = None
        self._offset = 0
        self._tail: List[str] = []
        self._terminal: Optional[str] = None
        self._n_evals = 0
        self._last_eval: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._stop_requested = False

    # ---------------------------------------------------------------- 启动
    def start(self, spec: JobSpec, *, slug: str) -> Dict[str, Any]:
        if self.proc is not None and self.proc.poll() is None:
            raise RuntimeError("已有优化在运行；先停止再启动")
        paths = build_command(spec, slug=slug)
        out: Path = paths["out"]
        out.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        })
        console = paths["console"]
        console.parent.mkdir(parents=True, exist_ok=True)
        self.console_file = open(console, "wb")
        creationflags = 0
        if _WIN:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            self.proc = subprocess.Popen(
                paths["cmd"],
                cwd=str(_REPO),
                env=env,
                stdout=self.console_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except Exception:
            self.console_file.close()
            self.console_file = None
            raise

        self.spec = spec
        self.slug = slug
        self.paths = paths
        self.started_at = time.time()
        self._offset = 0
        self._tail = []
        self._terminal = None
        self._n_evals = 0
        self._last_eval = None
        self._error = None
        self._stop_requested = False
        return self.state()

    # ---------------------------------------------------------------- 状态
    def _read_log_bytes(self) -> bytes:
        p: Optional[Path] = self.paths.get("log")
        if p is None or not p.exists():
            return b""
        with open(p, "rb") as f:
            return f.read()

    def _scan_terminal(self) -> Optional[str]:
        if self._terminal is not None:
            return self._terminal
        try:
            text = self._read_log_bytes().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return None
        for line in text.splitlines():
            if line.startswith("COMPLETE"):
                self._terminal = "complete"
            elif line.startswith("FAILED"):
                self._terminal = "failed"
                self._error = line.strip()[:500]
            elif line.startswith("STOP:"):
                self._terminal = "stopped"
                self._error = line.strip()[:500]
        return self._terminal

    def state(self) -> Dict[str, Any]:
        if self.proc is None:
            return {"running": False, "status": "idle", "msg": "空闲"}
        rc = self.proc.poll()
        running = rc is None
        terminal = self._scan_terminal()
        log_path: Optional[Path] = self.paths.get("log")
        has_log = bool(log_path and log_path.exists())
        if running:
            if not has_log:
                status, msg = "starting", "已启动，等待日志出现"
            elif self._n_evals == 0:
                status, msg = "preparing", "构建 FieldSystem / 预组装（首个 EVAL 前可能数十秒）"
            else:
                status, msg = "running", "运行中"
        else:
            if terminal == "complete":
                status, msg = "done", "完成（COMPLETE）"
            elif terminal == "failed":
                status, msg = "failed", f"失败：{self._error or '见日志'}"
            elif terminal == "stopped":
                status, msg = "stopped", f"已停止：{self._error or 'STOP'}"
            elif self._stop_requested:
                # 用户点了停止：我们**知道**是被终止的，但仍如实标注“无终态标记”。
                status, msg = "stopped", "已停止（用户请求；进程未写终态标记）"
            else:
                status, msg = "unknown", "进程已退出，但未见终态标记（不判定成功）"
        return {
            "running": running,
            "status": status,
            "msg": msg,
            "pid": (self.proc.pid if running else None),
            "returncode": rc,
            "elapsed_s": (time.time() - self.started_at) if self.started_at else 0.0,
            "cmd": list(self.paths.get("cmd", [])),
            "out": str(self.paths.get("out", "")),
            "log": str(self.paths.get("log", "")),
            "console": str(self.paths.get("console", "")),
            "n_evals": self._n_evals,
            "last_eval": self._last_eval,
        }

    # ---------------------------------------------------------------- 日志
    def log_delta(self, offset: int = 0) -> Dict[str, Any]:
        """按字节 offset 返回新增的**完整行**。"""
        p: Optional[Path] = self.paths.get("log")
        data = self._read_log_bytes()
        off = int(offset or 0)
        if off < 0 or off > len(data):
            off = 0                       # 文件被截断/重开
        chunk = data[off:]
        end = chunk.rfind(b"\n")
        if end < 0:
            complete = b""
        else:
            complete = chunk[: end + 1]
        text = complete.decode("utf-8", errors="replace")
        for line in text.splitlines():
            ev = _parse_eval_line(line)
            if ev is not None:
                self._n_evals += 1
                self._last_eval = ev
                self._tail.append(line)
                if len(self._tail) > 400:
                    del self._tail[:-400]
        self._scan_terminal()
        return {
            "ok": True,
            "offset": off,
            "next_offset": off + len(complete),
            "text": text,
            "done": bool(self._terminal) or (self.proc is not None
                                             and self.proc.poll() is not None),
            "state": self.state(),
        }

    def tail(self, n: int = 120) -> List[str]:
        return self._tail[-int(n):]

    # ---------------------------------------------------------------- 停止
    def stop(self, timeout_s: float = 5.0) -> Dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            return {"ok": False, "msg": "没有运行中的优化", "state": self.state()}
        proc = self.proc
        self._stop_requested = True
        try:
            if _WIN:
                try:
                    proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
                except Exception:  # noqa: BLE001
                    proc.terminate()
            else:
                proc.terminate()
            proc.wait(timeout=timeout_s)
        except Exception:  # noqa: BLE001
            if _WIN:
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        check=False, capture_output=True,
                    )
                except Exception:  # noqa: BLE001
                    pass
            else:
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._close_console()
        return {"ok": True, "msg": "停止请求已处理", "state": self.state()}

    def _close_console(self) -> None:
        if self.console_file is not None:
            try:
                self.console_file.flush()
                self.console_file.close()
            except Exception:  # noqa: BLE001
                pass
            self.console_file = None

    def shutdown(self, *, kill: bool = False) -> None:
        """服务退出时调用；默认**不杀**子进程（文档写明）。"""
        if kill:
            self.stop()
        self._close_console()
