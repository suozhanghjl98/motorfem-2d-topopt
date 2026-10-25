#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/optim_gui/_flow_test.py —— GUI 全流程冒烟测试（暂停/保存/停止/加载/续跑）。

直接打本机 server 的 REST 接口（urllib，无第三方依赖）。跑法：

    python -u tools/optim_gui/_flow_test.py [--port 8765]

脚本要求 server 已在同一端口运行。流程：
  1. start（steps=6）→ 等 step>=2
  2. pause → 等 status=paused
  3. save checkpoint 'flow_test'
  4. stop → 等 status=stopped
  5. load 'flow_test' → 等 status=paused
  6. start（续跑）→ 等 status=done
  7. 校验 history 长度 == 6、run_id 变化符合预期
"""
import argparse
import json
import sys
import time
import urllib.request

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


def req(method: str, path: str, body=None):  # body: Optional[dict]
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def state() -> dict:
    return req("GET", "/api/state")


def wait_for(pred, desc: str, timeout_s: float = 240.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = state()
        if pred(s):
            print(f"  ✓ {desc}（status={s['status']} step={s['step']}）")
            return s
        time.sleep(1.0)
    raise TimeoutError(f"超时等 {desc}；最后状态={state()['status']}")


def main() -> int:
    global PORT, BASE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    PORT = args.port
    BASE = f"http://127.0.0.1:{PORT}"

    print("== 1. 启动 6 步 ==")
    r = req("POST", "/api/start", {
        "steps": 6,
        "config": {"mma": {"max_step": 6}},
        "mesh_path": "legacy/pytorch/my_gmsh/model_mirror4.msh",
        "csv_path": "legacy/pytorch/50CS470.csv",
    })
    assert r["ok"], r
    s = wait_for(lambda st: st["step"] >= 2, "跑到第 2 步")

    print("== 2. 暂停 ==")
    r = req("POST", "/api/pause")
    assert r["ok"], r
    s = wait_for(lambda st: st["status"] == "paused", "暂停生效")

    print("== 3. 保存检查点 ==")
    r = req("POST", "/api/save", {"name": "flow_test"})
    assert r["ok"], r
    saved_path = r["msg"].strip()
    with open(saved_path, encoding="utf-8") as f:
        cp = json.load(f)
    saved_step = int(cp["step"])
    print(f"  已保存：{saved_path}（第 {saved_step} 步）")

    print("== 4. 停止 ==")
    r = req("POST", "/api/stop")
    assert r["ok"], r
    s = wait_for(lambda st: st["status"] == "stopped", "停止生效")
    assert s["step"] >= saved_step, s["step"]

    print("== 5. 加载检查点 ==")
    r = req("POST", "/api/load", {"name": "flow_test"})
    assert r["ok"], r
    s = wait_for(lambda st: st["status"] == "paused" and st["step"] == saved_step,
                 f"加载到第 {saved_step} 步")
    run_id_loaded = s["run_id"]
    hist_loaded = s["history_len"]
    assert hist_loaded == saved_step, f"加载后历史应保留前 {saved_step} 步，实得 {hist_loaded}"

    print("== 6. 续跑 ==")
    r = req("POST", "/api/start", {})   # 空 body ⇒ 用加载的配置/网格/检查点
    assert r["ok"], r
    s = wait_for(lambda st: st["status"] == "done", "续跑到完成")

    print("== 7. 校验 ==")
    hist = req("GET", "/api/history?start=0")
    print(f"  history_len={hist['n']}（应=6），obj 序列={[round(x,4) for x in hist['obj']]}")
    assert hist["n"] == 6, f"续跑后应 6 步，实得 {hist['n']}"
    assert s["run_id"] == run_id_loaded + 1, "续跑应使 run_id +1"
    assert hist_loaded == saved_step, f"加载后历史应保留前 {saved_step} 步"
    assert req("GET", "/api/geometry")["geometry"] is not None, "几何应在"
    print("  全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
