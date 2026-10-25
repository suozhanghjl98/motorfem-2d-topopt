"""star-of-slots 绕组生成器（阶段 5 核心）。

由 ``(Z, p, m, parts, 内/外层参考角)`` 生成 6 个相带的「槽 → 相带」映射，并给出
每扇区的线圈 → 相带归属。对账 legacy 相带表（``MMA10:270-277``）是硬门槛。

------------------------------------------------------------------------------
星形图（star of slots）算法（设计文档 §5 第 1~6 步）
------------------------------------------------------------------------------

::

    α_k = p·(参考机械角_k)              # 槽 k 的电角度
    sec_k = floor(α_k / (π/m)) mod 2m   # 60° 相带分区
    分区 → (相号, 符号)：0→A+ 1→C− 2→B+ 3→A− 4→C+ 5→B−
    双层：内层线圈 2k = 槽 k 的参考边 A 相带；外层 2k+1 = 槽 k 的参考边 B 相带

「参考机械角」由槽号 + 槽内偏移决定：``(k + off)·360/Z``，``off`` 取
``0``（起始边）/ ``0.5``（中心）/ ``1``（结束边）。

------------------------------------------------------------------------------
★ 两套 legacy 表 = 本函数的标准 60° 相带特例（实测逐位复现）
------------------------------------------------------------------------------

1. **SynRM**（Z=36, p=2, m=3, parts=4, 双层）：
   ``inner_ref='edge', outer_ref='edge', outer_shift=1`` —— 内层 = 槽**结束边**
   电角相带（相带边界落在槽中心线，槽 3/6/9 恰在边界归入下一相带）；外层 =
   全局槽 **k+1** 的结束边相带（双层 1 槽错位）。
2. **PMSM**（Z=6, p=2, m=3, parts=2）：``inner_ref='start', outer_ref='center',
   outer_shift=0`` —— q=0.5 分数槽，槽距 120° 电角 ⇒ 每槽跨 2 个相邻 60° 相带
   （内层 = 槽起始边、外层 = 槽中心相带）。⚠ 是**标准 60° 相带**（非 120° 相带
   绕组，参考资料里那种每相带 120° 宽与此无关）。

------------------------------------------------------------------------------
命名约定（与 RegionMap / golden 一致）
------------------------------------------------------------------------------

相带名 ``['Ap','An','Bp','Bn','Cp','Cn']``（legacy 全局名，顺序照抄
``MMA10:270-277``）。``slot_sec`` 返回 0..5 的 sec 索引，与 ``SEC_TO_NAME``
对照。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Dict, List, Sequence

import numpy as np

__all__ = [
    "WindingLayout",
    "star_of_slots",
    "SEC_TO_NAME",
    "coils_per_phase",
    "build_phase_element_sets",
    "winding_factors",
]

#: 分区 → 相带名（设计文档 §5 第 4 步：0→(A,+) 1→(C,−) 2→(B,+) 3→(A,−) 4→(C,+) 5→(B,−)）
SEC_TO_NAME: List[str] = ["Ap", "Cn", "Bp", "An", "Cp", "Bn"]

#: 槽参考位置 → 槽内偏移（槽宽倍数；start=0, center=0.5, edge=1）
_REF_OFFSET = {"start": 0.0, "center": 0.5, "edge": 1.0}


@dataclass(frozen=True)
class WindingLayout:
    """一份绕组布局：每全局槽的内/外层相带索引 + 每扇区线圈归属。

    Attributes
    ----------
    Z : int 总槽数
    pole_pairs : int 极对数
    phases : int 相数
    parts : int 周期份数（物理模型截取 Z/parts 个槽）
    layers : int 层数（本项目双层 = 2）
    inner_sec, outer_sec : (Z,) ndarray 每全局槽的内/外层相带索引（0..2m-1）
    slot_sec : (Z,) ndarray 内层相带索引（= inner_sec，向后兼容）
    slot_phase : (Z,) ndarray 内层相带名
    inner_ref, outer_ref : 内/外层槽参考位置（'start'/'center'/'edge'）
    outer_shift : 外层线圈的槽号偏移（全局）
    """

    Z: int
    pole_pairs: int
    phases: int
    parts: int
    layers: int
    inner_sec: np.ndarray
    outer_sec: np.ndarray
    inner_ref: str
    outer_ref: str
    outer_shift: int

    @property
    def slot_sec(self) -> np.ndarray:
        """内层相带索引（向后兼容旧字段名）。"""
        return self.inner_sec

    @property
    def slot_phase(self) -> np.ndarray:
        """内层相带名（= SEC_TO_NAME[inner_sec]）。"""
        return np.array([SEC_TO_NAME[s] for s in self.inner_sec])

    @property
    def slots_per_part(self) -> int:
        """每扇区槽数（物理模型截取宽度，设计文档 §5 第 6 步）。"""
        return self.Z // self.parts

    @property
    def n_coils_per_part(self) -> int:
        """每扇区的线圈实体数 = 槽数 × 层数。"""
        return self.slots_per_part * self.layers

    def coil_sec(self, coil_local: int) -> int:
        """扇区内线圈实体的相带索引（0..2m-1）。

        ``coil 2k`` = 槽 k 内层（``inner_sec[k]``）；
        ``coil 2k+1`` = 槽 k 外层（``outer_sec[k]`` —— ``outer_sec`` 已含
        ``outer_shift`` 的槽号偏移，见 ``star_of_slots``）。
        """
        k, layer = divmod(coil_local, self.layers)
        if layer:
            return int(self.outer_sec[k % self.Z])
        return int(self.inner_sec[k % self.Z])

    def coil_phase(self, coil_local: int) -> str:
        """扇区内线圈实体的相带名。"""
        return SEC_TO_NAME[self.coil_sec(coil_local)]


def star_of_slots(
    Z: int,
    pole_pairs: int,
    phases: int = 3,
    parts: int = 1,
    *,
    layers: int = 2,
    inner_ref: str = "edge",
    outer_ref: str = "edge",
    outer_shift: int = 1,
) -> WindingLayout:
    """star-of-slots 生成器。返回 ``WindingLayout``。

    Parameters
    ----------
    Z, pole_pairs, phases : 总槽数 / 极对数 / 相数。
    parts : 周期份数（物理模型截取 Z/parts 槽）。
    layers : 层数（2 = 双层；1 = 单层，此时只算内层，外层数组仍给但不用）。
    inner_ref, outer_ref : 内/外层线圈的槽参考位置，取 ``"start"``（槽起始边）/
        ``"center"``（槽中心）/ ``"edge"``（槽结束边）。参考机械角 =
        ``(k + off)·360/Z``，电角 = ``p·参考机械角``，相带 ``sec =
        floor(电角 / (180/m)) mod 2m``。
    outer_shift : 外层线圈的**槽号偏移**（槽 k 外层取槽 ``k + outer_shift`` 的
        参考角，跨扇区边界时全局取模）。

    Notes
    -----
    两套 legacy 表都是本函数的标准 60° 相带特例（模块 docstring 详述）：

    - **SynRM**（默认）：``inner_ref='edge', outer_ref='edge', outer_shift=1``
    - **PMSM**：``inner_ref='start', outer_ref='center', outer_shift=0``

    边界落在分区线上时（浮点精确）归入**下一**相带（``floor`` 语义）。
    """
    if Z % parts != 0:
        raise ValueError(f"parts={parts} 必须整除 Z={Z}")
    if inner_ref not in _REF_OFFSET or outer_ref not in _REF_OFFSET:
        raise ValueError(
            f"参考位置须为 start/center/edge，实得 inner={inner_ref} outer={outer_ref}")
    m = int(phases)
    slot_w = 360.0 / Z

    def _sec(slot: np.ndarray, ref: str) -> np.ndarray:
        mech = (slot + _REF_OFFSET[ref]) * slot_w
        elec = (pole_pairs * mech) % 360.0
        return np.floor(elec / (180.0 / m)).astype(int) % (2 * m)

    g = np.arange(Z)
    return WindingLayout(
        Z=Z, pole_pairs=pole_pairs, phases=m, parts=parts, layers=layers,
        inner_sec=_sec(g, inner_ref),
        outer_sec=_sec(g + outer_shift, outer_ref),
        inner_ref=inner_ref, outer_ref=outer_ref, outer_shift=outer_shift,
    )


def coils_per_phase(layout: WindingLayout) -> Dict[str, List[int]]:
    """每相带的扇区 0 线圈实体索引（与 legacy 相带表同构）。"""
    out: Dict[str, List[int]] = {n: [] for n in SEC_TO_NAME[: 2 * layout.phases]}
    for c in range(layout.n_coils_per_part):
        out[layout.coil_phase(c)].append(c)
    return out


def build_phase_element_sets(
    layout: WindingLayout,
    element_stator_coil: Sequence[set],
) -> Dict[str, set]:
    """由生成器推导 6 个相带的**单元集合**（对账 golden L1 的 element_phase*）。

    Parameters
    ----------
    layout : WindingLayout
    element_stator_coil : 长度 = 每扇区线圈数的列表，第 i 项 = 线圈实体 i 的
        单元集合（legacy 的 ``element_stator_coil``，RegionMap 提供）。

    Returns
    -------
    {相带名: 单元集合} —— 与 legacy ``element_phase{name}`` 同构。

    Notes
    -----
    这是「生成器 → 物理单元」的最后一跳：``coil_phase`` 给出每线圈的相带，
    相带单元集合 = 该相带所有线圈实体的单元并集。对账测试
    （``tests/winding/test_layout.py``）断言它与 golden L1 逐位相等。
    """
    if len(element_stator_coil) != layout.n_coils_per_part:
        raise ValueError(
            f"线圈实体数 {len(element_stator_coil)} != 生成器每扇区线圈数 "
            f"{layout.n_coils_per_part}（Z={layout.Z} parts={layout.parts} "
            f"layers={layout.layers}）"
        )
    out: Dict[str, set] = {}
    for name, coils in coils_per_phase(layout).items():
        s = set()
        for c in coils:
            s.update(element_stator_coil[c])
        out[name] = s
    return out


def winding_factors(
    layout: WindingLayout,
    pitch: int,
    *,
    sign: bool = True,
) -> tuple:
    """绕组系数 ``(k_p, k_d, k_w)``。对应设计文档 §5 第 7 步。

    Parameters
    ----------
    layout : WindingLayout
    pitch : int 节距（槽数）。整距 = 每极槽数 = Z/(2p)。
    sign : bool 分布因数是否带电流方向符号（双层回边取反）。默认 True。

    Returns
    -------
    ``(k_p, k_d, k_w)``：
        ``k_p = sin(p·π·pitch/Z)``（节距因数）；
        ``k_d = |Σ sign·exp(jα)|/n``（分布因数，对 A 相带去边槽求和）；
        ``k_w = k_p·k_d``。

    Notes
    -----
    - 分布因数只对**去边**槽求和（回边已在节距因数里体现）—— 与参考资料的
      MMF 谐波分析同构。
    - 对对称分布 ``k_d`` 与参考角取法无关（相量和模只取决于槽间角差）。
    - 生产配置（Z=36, p=2, 整距 pitch=9）：``k_p=1``、``k_d≈0.9598``
      （与 ``ResolvedConfig.integer_slot_winding_factor`` 一致）。
    """
    Z, p = layout.Z, layout.pole_pairs
    k_p = abs(np.sin(p * np.pi * pitch / Z))

    # A+ 相带的去边槽：取 phase 为 Ap 的槽，电角 = p·θ（center 参考）
    mech_deg = np.array([(g + 0.5) * 360.0 / Z for g in range(Z)])
    elec_deg = (p * mech_deg) % 360.0
    ap_mask = (elec_deg >= 0) & (elec_deg < 180.0 / layout.phases)
    alpha = np.deg2rad(elec_deg[ap_mask])
    if alpha.size == 0:
        raise ValueError("A+ 相带无槽 —— 检查 Z/p/m 组合")
    s = 1.0 if sign else 1.0
    k_d = float(np.abs(np.sum(s * np.exp(1j * alpha))) / alpha.size)
    k_w = k_p * k_d
    return k_p, k_d, k_w
