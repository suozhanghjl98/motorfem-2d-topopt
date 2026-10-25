"""motorfem.config.resolved —— 阶段 2：ResolvedConfig（运行解析后的完整配置快照）

设计文档 §阶段 2 末段：新增 ``ResolvedConfig``，每次跑落一份 ``resolved_config.json``，
含 ``model_hash``（.msh 的 sha256）、``winding_factor``、各区域单元数、库版本。

与 ``RunConfig`` 的区别：RunConfig 是「输入参数」；ResolvedConfig 是「运行解析后
的快照」——把网格指纹、区域统计这些**运行后才知道**的量一并落盘，使任何一次运行的
配置可完整复现（模型换了 / 网格变了能立刻从 json 看出来）。

``winding_factor``：legacy 无此量（设计文档新增）。当前用整数槽全节距双层绕组的
基波**分布因数**解析公式（阶段 5 星形图算法会替换为通用计算）；只作落盘元数据，
不参与任何数值计算。
"""
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import motorfem

from .config import RunConfig

__all__ = ["ResolvedConfig", "integer_slot_winding_factor"]


def integer_slot_winding_factor(Z: int, pole_pairs: int, m: int = 3) -> float:
    """整数槽全节距双层绕组的基波分布因数。

    q = Z / (2·p·m)（每极每相槽数），槽距电角度 α = π/(m·q)，
    分布因数 kd = sin(q·α/2) / (q·sin(α/2))；全节距 ⇒ 短距系数 kp = 1，
    绕组因数 kw = kd·kp = kd。对 Z=36/p=2/m=3：q=3、α=20°、kd=0.95979…
    """
    q = Z / (2 * pole_pairs * m)
    alpha = math.pi / (m * q)
    return math.sin(q * alpha / 2) / (q * math.sin(alpha / 2))


def _region_size(value: Any) -> int:
    """区域统计口径：set → len；dict（对称合并）→ len(keys)；list（槽表）→ len。"""
    if isinstance(value, (set, list)):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    raise TypeError(f"无法统计的区域类型 {type(value)}")


@dataclass
class ResolvedConfig:
    """运行解析后的完整配置快照（JSON 可序列化，落盘 resolved_config.json 用）。"""

    run: RunConfig
    model_name: str = ""                       # 网格文件名（如 model_mirror4.msh）
    model_hash: str = ""                       # .msh 的 sha256（网格指纹）
    region_element_counts: Dict[str, int] = field(default_factory=dict)
    winding_factor: Optional[float] = None     # 基波绕组因数（阶段 5 前为解析值）
    lib_version: str = motorfem.__version__

    # ------------------------------------------------------------ 构造
    @classmethod
    def from_legacy(cls, g: Dict[str, Any], msh_path, regions,
                    winding_factor: Optional[float] = None) -> "ResolvedConfig":
        """从 legacy 命名空间快照 + 网格 + 区域映射构造。

        ``regions``：RegionMap 或 ``name → set/dict/list`` 的 dict
        （RegionMap.region 的取值都能统计）。``winding_factor`` 不传时用
        整数槽分布因数解析值。
        """
        run = RunConfig.from_legacy(g)
        msh_path = str(msh_path)
        return cls(
            run=run,
            model_name=msh_path.split("/")[-1].split("\\")[-1],
            model_hash=_sha256(msh_path),
            region_element_counts={
                name: _region_size(regions.region(name) if hasattr(regions, "region")
                                   else regions[name])
                for name in (regions.names() if hasattr(regions, "names") else regions)
            },
            winding_factor=winding_factor
            if winding_factor is not None
            else integer_slot_winding_factor(run.geometry.Z, run.geometry.pole_pairs),
        )

    # ------------------------------------------------------------ 序列化
    def to_dict(self) -> Dict[str, Any]:
        d = self.run.to_dict()
        return {
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "region_element_counts": dict(sorted(self.region_element_counts.items())),
            "winding_factor": self.winding_factor,
            "lib_version": self.lib_version,
            "run_config": d,
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "ResolvedConfig":
        import json
        d = json.loads(s)
        return cls(
            run=RunConfig.from_dict(d["run_config"]),
            model_name=d["model_name"],
            model_hash=d["model_hash"],
            region_element_counts=dict(d["region_element_counts"]),
            winding_factor=d["winding_factor"],
            lib_version=d["lib_version"],
        )

    # ------------------------------------------------------------ 校验（阶段 3，闸②）
    def verify_mesh(self, msh_path) -> None:
        """网格指纹校验：当前 .msh 的 sha256 vs 本快照落盘的 ``model_hash``。

        resume 流程用：网格被换 / 被改 / 路径指错，立刻 raise（阶段 3 三道闸之二）。
        """
        from ..mesh.gates import verify_fingerprint

        verify_fingerprint(msh_path, self.model_hash, model_name=self.model_name)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
