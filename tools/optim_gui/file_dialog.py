#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools/optim_gui/file_dialog.py —— 独立进程弹系统原生文件选择对话框。

网页无法通过 ``<input type=file>`` 拿到本地文件的绝对路径（浏览器安全限制，
只给 ``C:\\fakepath\\xxx``）。server（``server.py::browse_native``）用
``subprocess`` 启动本脚本：弹出 **tkinter 原生系统对话框**，把用户选中的
绝对路径打印到 stdout。GUI 线程与 server 线程因此互不干扰。

用法（由 server 调用）::

    python -u file_dialog.py --title "选择文件" --dir "D:\\..."

用户取消 → 无 stdout 输出，退出码 0；选中 → 打印一行绝对路径。
"""

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")   # 中文路径在 GBK 控制台也能打
except AttributeError:  # pragma: no cover
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--title", default="选择文件")
    ap.add_argument("--dir", default="")
    ap.add_argument("--filetypes", default="", help="'名称;*.ext' 分号分隔的 filetypes")
    args = ap.parse_args()

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)          # 对话框浮在浏览器之上
    root.update()

    initialdir = args.dir or os.getcwd()
    if not os.path.isdir(initialdir):
        initialdir = os.getcwd()

    filetypes = [("所有文件", "*.*")]
    if args.filetypes:
        for part in args.filetypes.split(";"):
            if part.strip():
                filetypes.insert(0, (part.strip(), "*.*"))

    try:
        path = filedialog.askopenfilename(
            title=args.title, initialdir=initialdir, parent=root,
            filetypes=filetypes)
    finally:
        root.destroy()

    if path:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
