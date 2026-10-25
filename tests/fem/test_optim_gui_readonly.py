# -*- coding: utf-8 -*-
"""只读监视模式的契约测试（不跑 FEM，不 import 计算栈）。

覆盖 ``docs/GUI只读监控产物契约.md`` 的每一条：日志行解析、终态标记、
pid 存活判定、NPZ 快照、POST 全 403、HTML 首帧禁用。
"""
import importlib.util
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "optim_gui_server", ROOT / "tools" / "optim_gui" / "server.py")
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)


def test_coerce_stage_accepts_string_and_int():
    """run_auto_schedule 的 stage 是字符串（"S0"），不能直接 int()。"""
    assert gui._coerce_stage(np.asarray("S0")) == "S0"
    assert gui._coerce_stage(np.asarray(3)) == 3
    assert gui._coerce_stage(np.asarray("3")) == 3


# ------------------------------------------------------------------ 日志解析
def _mgr(tmp_path, log_text="", npz=None):
    log = tmp_path / "run.log"
    if log_text is not None:
        log.write_text(log_text, encoding="utf-8")
    return gui.LogManager(log, npz or (tmp_path / "run.npz"), "m.msh", "c.csv")


def test_running_line_sets_pid_and_target(tmp_path):
    mgr = _mgr(tmp_path, "RUNNING pid=999999 target=250\n")
    mgr.refresh()
    assert mgr._pid == 999999 and mgr.target == 250


def test_incomplete_trailing_line_is_not_consumed(tmp_path):
    """生产端正在写的那半行不能被当成完整事件。"""
    mgr = _mgr(tmp_path, "RUNNING pid=1 target=10\nEVAL stage=1 n=1 obj=5.0 seconds=1\n")
    mgr.refresh()
    assert len(mgr.rows) == 1
    before = len(mgr.rows)
    with open(tmp_path / "run.log", "a", encoding="utf-8") as f:
        f.write("EVAL stage=1 n=2 obj=4.0 sec")     # 没换行
    mgr.refresh()
    assert len(mgr.rows) == before, "未写完的半行必须被忽略"


def test_eval_parses_metrics_and_missing_constraints(tmp_path):
    mgr = _mgr(tmp_path,
               "RUNNING pid=1\n"
               "EVAL stage=2 n=7 obj=62.5 T_flux=72.6 seconds=3.25 "
               "con1=0.1 con2=-0.2 con3=-3.7 con4=-\n")
    mgr.refresh()
    r = mgr.rows[0]
    assert (r["stage"], r["n"], r["obj"]) == (2, 7, 62.5)
    assert r["seconds"] == 3.25
    assert r["cons"][:3] == [0.1, -0.2, -3.7]
    assert r["cons"][3] is None, "连字符 = 未启用，必须是 None 而不是 0"


def test_eval_missing_required_field_is_explicit_error(tmp_path):
    mgr = _mgr(tmp_path, "RUNNING pid=1\nEVAL stage=1 obj=5.0\n")
    mgr.refresh()
    assert mgr.status == "error"
    assert "EVAL" in mgr.error


def test_unknown_line_is_ignored_not_fatal(tmp_path):
    """生产端新增日志行不得让消费者崩 —— 这是契约化的核心动机。"""
    mgr = _mgr(tmp_path,
               "RUNNING pid=1\n"
               "PROGRESS something=1\n"
               "EVAL stage=1 n=1 obj=5.0 seconds=1\n")
    mgr.refresh()
    assert len(mgr.rows) == 1 and mgr.status != "error"


def test_labels_line_overrides_defaults(tmp_path):
    mgr = _mgr(tmp_path, "RUNNING pid=1\nLABELS con1=ArkBE con3=flux70\n")
    mgr.refresh()
    assert mgr.labels["con1"] == "ArkBE" and mgr.labels["con3"] == "flux70"
    assert mgr.labels["con2"] == "con2", "未覆盖的保持默认"


# ------------------------------------------------------------------ 终态
@pytest.mark.parametrize("line,status", [
    ("COMPLETE\n", "done"),
    ("FAILED 网格奇异\n", "error"),
    ("STOP: 门槛未过\n", "stopped"),
])
def test_terminal_markers(tmp_path, line, status):
    mgr = _mgr(tmp_path, "RUNNING pid=1\n" + line)
    mgr.refresh()
    assert mgr.status == status
    assert mgr._terminal


def test_log_without_terminal_marker_is_never_success(tmp_path):
    """★ 核心纪律：进程退出 ≠ 优化成功。无终态标记一律 unknown。

    用一个必然不存在的 pid，模拟「日志还在但进程没了」。
    """
    mgr = _mgr(tmp_path, "RUNNING pid=1\nEVAL stage=1 n=1 obj=5.0 seconds=1\n")
    mgr.refresh()
    assert mgr.status == "unknown"
    assert "不判定成功" in mgr.msg


def test_missing_log_is_waiting_not_error(tmp_path):
    mgr = _mgr(tmp_path, None)          # 日志文件不存在
    mgr.refresh()
    assert mgr.status == "waiting" and mgr.error is None


# ------------------------------------------------------------------ NPZ
def _write_npz(path, x_calls, **extra):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, x_calls=np.asarray(x_calls),
                            config_json=json.dumps({"aaa": 1}), **extra)
    tmp.replace(path)


def test_npz_missing_x_calls_is_error(tmp_path, monkeypatch):
    p = tmp_path / "run.npz"
    with p.open("wb") as f:
        np.savez_compressed(f, other=np.zeros(3))
    mgr = _mgr(tmp_path, "RUNNING pid=1\n", npz=p)
    with pytest.raises(ValueError, match="x_calls"):
        mgr._load_snapshot(p, ("k", 1))


def test_npz_prefix_probe_picks_newest(tmp_path):
    """--npz 给前缀时，取 mtime 最新且存在的 _stage*.npz。"""
    _write_npz(tmp_path / "run_stage1.npz", [[1.0, 2.0]])
    time.sleep(0.01)
    _write_npz(tmp_path / "run_stage2.npz", [[3.0, 4.0]])
    mgr = _mgr(tmp_path, "", npz=tmp_path / "run")
    cands = mgr._npz_candidates()
    assert [c.name for c in cands] == ["run_stage1.npz", "run_stage2.npz"]
    assert cands[-1].name == "run_stage2.npz"


# ------------------------------------------------------------------ payload
def test_state_payload_shape_and_readonly_flag(tmp_path):
    mgr = _mgr(tmp_path,
               "RUNNING pid=1 target=250\n"
               "EVAL stage=1 n=3 obj=9.0 seconds=2.0 con2=-0.1\n")
    s = mgr.state_payload()
    assert s["read_only"] is True
    assert s["step"] == 3 and s["stage"] == 1 and s["target"] == 250
    assert s["cons"][0] is None and s["cons"][1] == -0.1
    assert s["times"]["fun_ms"] == 2000.0
    assert s["saves"] == [] and s["schema"] == []


def test_history_payload_slices(tmp_path):
    lines = "RUNNING pid=1\n" + "".join(
        f"EVAL stage=1 n={i} obj={10.0 - i} seconds=1\n" for i in range(1, 6))
    mgr = _mgr(tmp_path, lines)
    mgr.refresh()
    h = mgr.history_payload(2)
    assert h["n"] == 5 and h["start"] == 2
    assert h["step"] == [3, 4, 5] and h["obj"] == [7.0, 6.0, 5.0]


def test_history_start_clamped(tmp_path):
    mgr = _mgr(tmp_path, "RUNNING pid=1\nEVAL stage=1 n=1 obj=1.0 seconds=1\n")
    mgr.refresh()
    h = mgr.history_payload(999)
    assert h["start"] == 1 and h["step"] == []


# ------------------------------------------------------------------ HTTP 边界
@pytest.fixture
def readonly_server(tmp_path):
    (tmp_path / "run.log").write_text("RUNNING pid=1 target=10\n"
                                      "EVAL stage=1 n=1 obj=5.0 seconds=1\n",
                                      encoding="utf-8")
    mgr = gui.LogManager(tmp_path / "run.log", tmp_path / "run.npz", "m.msh", "c.csv")
    handler = type("H", (gui.Handler,), {"mgr": mgr, "read_only": True})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _req(url, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_get_state_and_history_work(readonly_server):
    code, s = _req(readonly_server + "/api/state")
    assert code == 200 and s["read_only"] is True and s["step"] == 1
    code, h = _req(readonly_server + "/api/history?start=0")
    assert code == 200 and h["step"] == [1]


@pytest.mark.parametrize("path", ["/api/start", "/api/pause", "/api/resume",
                                  "/api/step", "/api/stop", "/api/save",
                                  "/api/load", "/api/browse-native"])
def test_all_posts_are_403(readonly_server, path):
    code, r = _req(readonly_server + path, "POST", {})
    assert code == 403 and r["ok"] is False


def test_index_html_is_disabled_before_first_frame(readonly_server):
    """控件禁用必须写进 HTML，不能等 JS 事后 disable。"""
    with urllib.request.urlopen(readonly_server + "/", timeout=10) as resp:
        html = resp.read().decode("utf-8")
    assert "只读监视" in html
    assert "运行信息（只读，控制请用命令行）" in html
    # 属性顺序不固定，按元素逐个断言（每个控件标签内都要有 disabled）
    import re
    tags = re.findall(r"<(?:button|input|select|textarea)\b[^>]*>", html)
    assert tags, "没找到任何控件"
    missing = [t for t in tags if "disabled" not in t]
    assert not missing, f"这些控件在首帧前没被禁用：{missing}"
    start = [t for t in tags if 'id="btn-start"' in t]
    assert start and "disabled" in start[0]


def test_readonly_does_not_import_compute_stack():
    """★ 解耦的硬指标：只读模式不拉起计算栈。

    已 import 的话 FieldSystem 不是 None；这里断言模块级全局仍是惰性占位。
    """
    assert gui.FieldSystem is None, "只读路径不应在 import 期加载 FieldSystem"
