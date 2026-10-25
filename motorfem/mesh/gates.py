"""motorfem.mesh.gates —— 阶段 3：求解器侧三道闸（断掉「tag 恰好对齐」）

设计文档 §阶段 3 的求解器侧三道闸：

1. **启动断言**（``require_regions``）：每个 driver 声明必需区域，缺一个就 raise，
   并列出该模型实际拥有的组 —— 把「改几何就静默算错」（F1）变成启动即报错。
2. **网格指纹**（``verify_fingerprint``）：``model_hash``（.msh 的 sha256）落盘，
   resume 时校验。已由 ``ResolvedConfig.model_hash`` 落盘；这里的函数是校验侧。
3. **区域覆盖性检查**（``RegionMap.check_partition``，见 ``motorfem/mesh/mesh.py``）：
   所有组的并集 == 全部单元、两两不相交。今天做不到这件事正是因为物理组不存在。

三道闸都是纯校验（只 raise，不改任何数值）。
"""

import hashlib
from typing import Iterable

__all__ = ["require_regions", "verify_fingerprint"]


def require_regions(regions, required: Iterable[str], who: str = "driver") -> None:
    """启动断言（闸①）：``regions``（RegionMap 或 ``name in`` 容器）缺任一必需名即 raise。

    Parameters
    ----------
    regions :
        区域映射（``RegionMap`` 或任何支持 ``name in regions`` 与 ``regions.names()``
        的对象）。
    required :
        该 driver 声明必需的 legacy 区域名。
    who :
        driver 名（错误信息里标注，如 ``"FieldSystem(EM)"``）。
    """
    missing = [name for name in required if name not in regions]
    if missing:
        have = sorted(regions.names())
        raise KeyError(
            f"{who}: 必需区域缺失 {missing}；网格实际拥有的组: {have}。"
            f"（阶段 3 启动断言 —— 区域名契约被违反，可能是网格与配置不匹配）"
        )


def verify_fingerprint(msh_path, expected_sha256: str, model_name: str = "") -> None:
    """网格指纹（闸②）：.msh 实算 sha256 与期望不符即 raise。

    resume 流程：读回上次 ``ResolvedConfig`` 的 ``model_hash``，与当前 .msh 实算
    哈希比对 —— 网格被换 / 被改 / 路径指错立刻报错，而不是静默算一个错的结果。
    """
    actual = _sha256(msh_path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(
            f"网格指纹不匹配（{model_name or msh_path}）:\n"
            f"  期望 {expected_sha256}\n"
            f"  实算 {actual}\n"
            f"  网格文件与记录不一致 —— 禁止 resume / 继续。"
        )


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
