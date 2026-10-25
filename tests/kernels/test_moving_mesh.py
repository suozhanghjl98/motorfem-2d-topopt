"""内核 7 测试：动网格滑移层与旋转边界权重矩阵。

测试三级来源：

  来源 A 解析性质      —— ``pos = 0`` 时插值退化为恒等（静网格一致性）；逐扇区改号
                          的线性斜坡场是所有插值行的精确不动点；周期情形下常值场
                          满足全部行（行和为零）；权重与手算角度比一致；解析导数与
                          中心差分一致；接口角覆盖性前提；插入层的节点复制/单射/
                          定转子节点不相交/虚拟单元增量更新等价
  来源 B 与 legacy 一致 —— 照抄 ``MMA10:443-491``（插入层）、``743-806``
                          （``rotating_boundary``）、``815-903``（``rotating_boundary_v2``）
                          到本文件，在合成网格上逐位比对（``np.array_equal``）
  来源 C ref vs fast   —— 见本文件末节（``fast/`` 未落地时自动跳过）

运行::

    python -m pytest tests/kernels/test_moving_mesh.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse as sp_sparse

from motorfem.kernels.reference import boundaries as ref_bnd
from motorfem.kernels.reference import moving_mesh as ref_mm

try:  # 来源 C 的 fast 层（子任务 4 落地；未落地时 C 级测试自动跳过）
    from motorfem.kernels.fast import moving_mesh as fast_mm
except ImportError:  # pragma: no cover
    fast_mm = None

requires_fast = pytest.mark.skipif(fast_mm is None, reason="fast/ 层尚未落地")


# =============================================================================
# 合成网格
# =============================================================================

#: 扇区角（parts = 4 → 90°）
SECTOR = np.pi / 2

#: 四层同心圆：转子内 / 转子外 / **接口圆** / 定子外（单位 m）
R_RINGS = (0.030, 0.040, 0.050, 0.060)

#: 真实网格上的接口节点数（实测 236）—— 用于「同规模」的来源 B 复核
N_ANG_REAL = 236

#: legacy 的坐标判等容差（``MMA10:87`` 的 ``_coord_err_abs`` 实参）
COORD_ERR_ABS = 1e-6

#: pos 采样：覆盖偶/奇扇区、多个扇区序号、以及「恰好落在接口节点角上」（0）
POS_SAMPLES = (0.0, 0.3, 0.77, np.pi / 4, 1.0, 2.0, 3.0, 4.1, -0.4, 2 * SECTOR - 1e-9)


def make_interface_mesh(n_ang=7, radii=R_RINGS):
    """造一个含接口圆的极坐标环扇网格。

    节点编号 ``= 层序号 * n_ang + 角度序号``，角度取 ``linspace(0, 90°, n_ang)``
    —— **含两端点**，即接口角恰好覆盖 ``[0, w]``。θ=0 与 θ=90° 射线上因而有节点，
    内核 6 的主从配对才成立（θ=90° 的节点是从属，θ=0 的是它的主动节点）。

    Returns
    -------
    node : (N, 4) ndarray
        与 legacy 同构的节点表 ``[节点号, x, y, 0]``（坐标为米）。
    element_S : list[list]
        与 legacy 同构的连接表 ``[单元号, n1, n2, n3, 区域号]``；区域号
        0 = 转子铁芯、1 = 转子气隙、2 = 定子气隙。
    element_rotor_gap, element_stator, element_rotor : set[int]
    """
    angles = np.linspace(0.0, SECTOR, n_ang)
    xy = np.array(
        [[r * np.cos(t), r * np.sin(t)] for r in radii for t in angles], dtype=float
    )
    node = np.column_stack(
        [np.arange(xy.shape[0], dtype=float), xy[:, 0], xy[:, 1], np.zeros(xy.shape[0])]
    )

    element_S = []
    for k in range(len(radii) - 1):
        for l in range(n_ang - 1):
            a = k * n_ang + l
            b = (k + 1) * n_ang + l
            c = (k + 1) * n_ang + l + 1
            d = k * n_ang + l + 1
            element_S.append([len(element_S), a, b, c, k])
            element_S.append([len(element_S), a, c, d, k])

    element_rotor_gap = set(_e[0] for _e in element_S if _e[4] == 1)
    element_stator = set(_e[0] for _e in element_S if _e[4] == 2)
    element_rotor = set(_e[0] for _e in element_S if _e[4] in (0, 1))
    return node, element_S, element_rotor_gap, element_stator, element_rotor


def tri_of(element_S):
    """``element_S`` → ``(N_elem, 3)`` 连接表（本内核的输入约定）。"""
    return np.array([_e[1:4] for _e in element_S], dtype=int)


def interface_set_legacy(node, r_interface=0.050, coord_err_abs=COORD_ERR_ABS):
    """照抄 ``MMA10:339-340``：半径**平方差**判据挑出接口圆上的节点（set）。"""
    return set(
        [int(_node[0]) for _node in node
         if np.abs((_node[1] ** 2 + _node[2] ** 2) - r_interface ** 2) < coord_err_abs]
    )


# =============================================================================
# legacy 片段（照抄，不复用被测实现）
# =============================================================================

def legacy_rot_x(x, y, angle):
    """照抄 MMA10:177-178。"""
    return x * np.cos(angle) - y * np.sin(angle)


def legacy_rot_y(x, y, angle):
    """照抄 MMA10:181-182。"""
    return x * np.sin(angle) + y * np.cos(angle)


class LegacyState:
    """复刻 legacy 的模块级全局量（``node``/``node_slave``/``node_gap_*`` 等）。"""

    def __init__(self, node, element_S, element_rotor_gap, element_stator,
                 element_rotor, parts=4):
        self.node = node
        self.element_S = [list(_e) for _e in element_S]
        self.element_rotor_gap = element_rotor_gap
        self.element_stator = element_stator
        self.element_rotor = element_rotor
        self.parts = parts
        self.EM_MS_boundary_A_eq = False
        self.N_node = node.shape[0]

        # 主从配对（照抄 MMA10:304-321）
        self.node_master = [int(_node[0]) for _node in node if
                            np.abs(_node[2]) < COORD_ERR_ABS < np.abs(_node[1])]
        self.node_slave = {}
        for i_node_s in [int(_node[0]) for _node in node
                         if np.abs(legacy_rot_y(_node[1], _node[2], -2 * np.pi / parts))
                         < COORD_ERR_ABS < np.abs(
                             legacy_rot_x(_node[1], _node[2], -2 * np.pi / parts))]:
            for i_node_m in self.node_master:
                if (np.abs(legacy_rot_x(node[i_node_m][1], node[i_node_m][2],
                                        2 * np.pi / parts) - node[i_node_s][1])
                        < COORD_ERR_ABS
                        and np.abs(legacy_rot_y(node[i_node_m][1], node[i_node_m][2],
                                                2 * np.pi / parts) - node[i_node_s][2])
                        < COORD_ERR_ABS):
                    self.node_slave[i_node_s] = i_node_m
                    break
            assert i_node_s in self.node_slave
        self.node_master = dict((v, k) for (k, v) in self.node_slave.items())

        # 从属虚拟单元（照抄 MMA10:323-335）
        self.element_slave = set([_elem[0] for _elem in self.element_S
                                  if (_elem[1] in self.node_slave
                                      or _elem[2] in self.node_slave
                                      or _elem[3] in self.node_slave)])
        self.element_S_slave = {}
        for i_elem in self.element_slave:
            self.element_S_slave[i_elem] = self._virtual_row(i_elem)

        # 接口圆节点（照抄 MMA10:339-340）
        self.node_gap_stator = interface_set_legacy(self.node)

    def _virtual_row(self, i_elem):
        e = self.element_S[i_elem]
        ns = self.node_slave
        return [e[0],
                ns[e[1]] if e[1] in ns else e[1],
                ns[e[2]] if e[2] in ns else e[2],
                ns[e[3]] if e[3] in ns else e[3],
                e[4]]

    def insert_layer(self):
        """照抄 ``MMA10:442-491``（插入动网格节点层，就地改写全局量）。"""
        node = self.node
        N_node = self.N_node
        element_S = self.element_S
        node_gap_stator = self.node_gap_stator
        node_slave = self.node_slave
        node_master = self.node_master
        element_rotor_gap = self.element_rotor_gap

        N_node_gap = len(node_gap_stator)

        node_gap_rotor = dict((i + N_node, j) for (i, j) in enumerate(node_gap_stator))
        node_gap_stator = dict((j, i) for (i, j) in node_gap_rotor.items())
        _node_gap_rotor = []
        for i_node_gap_rotor, i_node_gap_stator in node_gap_rotor.items():
            _node_gap_rotor.append([i_node_gap_rotor,
                                    node[i_node_gap_stator][1],
                                    node[i_node_gap_stator][2], 0])

        _node_gap_rotor = np.array(_node_gap_rotor)
        node = np.concatenate([node, _node_gap_rotor], axis=0)
        N_node += N_node_gap

        for i_elem in element_rotor_gap:
            for _i in [1, 2, 3]:
                i_node = element_S[i_elem][_i]
                if i_node in node_gap_stator:
                    element_S[i_elem][_i] = node_gap_stator[i_node]

        for i_node_gap_rotor, i_node_gap_stator in node_gap_rotor.items():
            if i_node_gap_stator in node_slave:
                node_slave[i_node_gap_rotor] = node_gap_stator[node_slave[i_node_gap_stator]]
                node_master[node_gap_stator[node_slave[i_node_gap_stator]]] = i_node_gap_rotor

        for i_elem in element_rotor_gap.intersection(self.element_S_slave.keys()):
            self.element_S_slave[i_elem] = self._virtual_row(i_elem)

        node_stator = set()
        node_rotor = set()
        for i_elem in self.element_stator:
            node_stator.add(element_S[i_elem][1])
            node_stator.add(element_S[i_elem][2])
            node_stator.add(element_S[i_elem][3])
        for i_elem in self.element_rotor:
            node_rotor.add(element_S[i_elem][1])
            node_rotor.add(element_S[i_elem][2])
            node_rotor.add(element_S[i_elem][3])

        self.node = node
        self.N_node = N_node
        self.node_gap_stator = node_gap_stator
        self.node_gap_rotor = node_gap_rotor
        self.node_stator = node_stator
        self.node_rotor = node_rotor


def legacy_rotating_boundary(g, _pos):
    """照抄 ``MMA10:738-806``。"""
    N_node = g.N_node
    node = g.node
    node_gap_stator = g.node_gap_stator
    node_gap_rotor = g.node_gap_rotor
    node_slave = g.node_slave
    parts = g.parts

    _K_rotating_boundary = sp_sparse.lil_matrix((N_node, N_node))
    _node_rotating_boundary = {}

    _node_stator_th = [(np.arctan2(node[i_node_gap_stator, 2], node[i_node_gap_stator, 1]), i_node_gap_stator)
                       for i_node_gap_stator in node_gap_stator]
    _node_stator_th = sorted(_node_stator_th, key=lambda item: item[0])

    for i_node_gap_rotor in node_gap_rotor:
        _node_rotor_th = np.arctan2(node[i_node_gap_rotor, 2], node[i_node_gap_rotor, 1]) + _pos

        _node_rotor_th_mod = np.mod(_node_rotor_th, 2 * np.pi / parts)
        _other_region = np.mod(_node_rotor_th // (2 * np.pi / parts), 2) > 0.5  # 位于另一个周期区域内

        index_stator = 0
        while True:
            if _node_stator_th[index_stator][0] > _node_rotor_th_mod:
                break
            index_stator += 1

        assert index_stator > 0

        # 处理主从边界的问题
        _kp = (_node_rotor_th_mod - _node_stator_th[index_stator - 1][0]) / (
                _node_stator_th[index_stator][0] - _node_stator_th[index_stator - 1][0])
        _kn = 1 - _kp

        if not g.EM_MS_boundary_A_eq:
            # 反周期边界
            _K_rotating_boundary[i_node_gap_rotor, i_node_gap_rotor] = 1
            if _node_stator_th[index_stator][1] in node_slave:
                _K_rotating_boundary[i_node_gap_rotor,
                node_slave[_node_stator_th[index_stator][1]]] = -_kp if _other_region else _kp
            else:
                _K_rotating_boundary[i_node_gap_rotor, _node_stator_th[index_stator][1]] = (
                    _kp if _other_region else -_kp)

            if _node_stator_th[index_stator - 1][1] in node_slave:
                _K_rotating_boundary[i_node_gap_rotor,
                node_slave[_node_stator_th[index_stator - 1][1]]] = -_kn if _other_region else _kn
            else:
                _K_rotating_boundary[i_node_gap_rotor, _node_stator_th[index_stator - 1][1]] = (
                    _kn if _other_region else -_kn)

        else:
            # 周期边界
            _K_rotating_boundary[i_node_gap_rotor, i_node_gap_rotor] = 1

            if _node_stator_th[index_stator][1] in node_slave:
                _K_rotating_boundary[i_node_gap_rotor, node_slave[_node_stator_th[index_stator][1]]] = -_kp
            else:
                _K_rotating_boundary[i_node_gap_rotor, _node_stator_th[index_stator][1]] = -_kp

            if _node_stator_th[index_stator - 1][1] in node_slave:
                _K_rotating_boundary[i_node_gap_rotor, node_slave[_node_stator_th[index_stator - 1][1]]] = -_kn
            else:
                _K_rotating_boundary[i_node_gap_rotor, _node_stator_th[index_stator - 1][1]] = -_kn

        _node_rotating_boundary[i_node_gap_rotor] = (_node_stator_th[index_stator][1]
                                                     if _node_stator_th[index_stator][1] not in node_slave
                                                     else node_slave[_node_stator_th[index_stator][1]],
                                                     _node_stator_th[index_stator - 1][1]
                                                     if _node_stator_th[index_stator - 1][1] not in node_slave
                                                     else node_slave[_node_stator_th[index_stator - 1][1]])

    _K_rotating_boundary = sp_sparse.csr_matrix(_K_rotating_boundary)

    return _K_rotating_boundary, _node_rotating_boundary


def legacy_rotating_boundary_v2(g, _pos):
    """照抄 ``MMA10:809-903``（带 ∂K/∂θ）。"""
    N_node = g.N_node
    node = g.node
    node_gap_stator = g.node_gap_stator
    node_gap_rotor = g.node_gap_rotor
    node_slave = g.node_slave
    parts = g.parts

    _K_rotating_boundary = sp_sparse.lil_matrix((N_node, N_node))
    _dKrb_p_dth = sp_sparse.lil_matrix((N_node, N_node))
    _node_rotating_boundary = {}

    _node_stator_th = [(np.arctan2(node[i_node_gap_stator, 2], node[i_node_gap_stator, 1]), i_node_gap_stator)
                       for i_node_gap_stator in node_gap_stator]
    _node_stator_th = sorted(_node_stator_th, key=lambda item: item[0])

    for i_node_gap_rotor in node_gap_rotor:
        _node_rotor_th = np.arctan2(node[i_node_gap_rotor, 2], node[i_node_gap_rotor, 1]) + _pos

        _node_rotor_th_mod = np.mod(_node_rotor_th, 2 * np.pi / parts)
        _other_region = np.mod(_node_rotor_th // (2 * np.pi / parts), 2) > 0.5

        index_stator = 0
        while True:
            if _node_stator_th[index_stator][0] > _node_rotor_th_mod:
                break
            index_stator += 1

        assert index_stator > 0

        _kp = (_node_rotor_th_mod - _node_stator_th[index_stator - 1][0]) / (
                _node_stator_th[index_stator][0] - _node_stator_th[index_stator - 1][0])
        _kn = 1 - _kp

        _dkp = 1 / (
                _node_stator_th[index_stator][0] - _node_stator_th[index_stator - 1][0])
        _dkn = - _dkp

        if not g.EM_MS_boundary_A_eq:
            _K_rotating_boundary[i_node_gap_rotor, i_node_gap_rotor] = 1
            if _node_stator_th[index_stator][1] in node_slave:
                _K_rotating_boundary[i_node_gap_rotor,
                node_slave[_node_stator_th[index_stator][1]]] = -_kp if _other_region else _kp

                _dKrb_p_dth[i_node_gap_rotor,
                node_slave[_node_stator_th[index_stator][1]]] = -_dkp if _other_region else _dkp
            else:
                _K_rotating_boundary[i_node_gap_rotor, _node_stator_th[index_stator][1]] = (
                    _kp if _other_region else -_kp)

                _dKrb_p_dth[i_node_gap_rotor, _node_stator_th[index_stator][1]] = (
                    _dkp if _other_region else -_dkp)

            if _node_stator_th[index_stator - 1][1] in node_slave:
                _K_rotating_boundary[i_node_gap_rotor,
                node_slave[_node_stator_th[index_stator - 1][1]]] = -_kn if _other_region else _kn

                _dKrb_p_dth[i_node_gap_rotor,
                node_slave[_node_stator_th[index_stator - 1][1]]] = -_dkn if _other_region else _dkn
            else:
                _K_rotating_boundary[i_node_gap_rotor, _node_stator_th[index_stator - 1][1]] = (
                    _kn if _other_region else -_kn)

                _dKrb_p_dth[i_node_gap_rotor, _node_stator_th[index_stator - 1][1]] = (
                    _dkn if _other_region else -_dkn)

        else:
            _K_rotating_boundary[i_node_gap_rotor, i_node_gap_rotor] = 1

            if _node_stator_th[index_stator][1] in node_slave:
                _K_rotating_boundary[i_node_gap_rotor, node_slave[_node_stator_th[index_stator][1]]] = -_kp

                _dKrb_p_dth[i_node_gap_rotor, node_slave[_node_stator_th[index_stator][1]]] = -_dkp
            else:
                _K_rotating_boundary[i_node_gap_rotor, _node_stator_th[index_stator][1]] = -_kp

                _dKrb_p_dth[i_node_gap_rotor, _node_stator_th[index_stator][1]] = -_dkp

            if _node_stator_th[index_stator - 1][1] in node_slave:
                _K_rotating_boundary[i_node_gap_rotor, node_slave[_node_stator_th[index_stator - 1][1]]] = -_kn

                _dKrb_p_dth[i_node_gap_rotor, node_slave[_node_stator_th[index_stator - 1][1]]] = -_dkn
            else:
                _K_rotating_boundary[i_node_gap_rotor, _node_stator_th[index_stator - 1][1]] = -_kn

                _dKrb_p_dth[i_node_gap_rotor, _node_stator_th[index_stator - 1][1]] = -_dkn

        _node_rotating_boundary[i_node_gap_rotor] = (_node_stator_th[index_stator][1]
                                                     if _node_stator_th[index_stator][1] not in node_slave
                                                     else node_slave[_node_stator_th[index_stator][1]],
                                                     _node_stator_th[index_stator - 1][1]
                                                     if _node_stator_th[index_stator - 1][1] not in node_slave
                                                     else node_slave[_node_stator_th[index_stator - 1][1]])

    _K_rotating_boundary = sp_sparse.csr_matrix(_K_rotating_boundary)
    _dKrb_p_dth = sp_sparse.csr_matrix(_dKrb_p_dth)

    return _K_rotating_boundary, _node_rotating_boundary, _dKrb_p_dth


# =============================================================================
# 夹具
# =============================================================================

def _layer_of(node, element_S, element_rotor_gap, element_stator, element_rotor,
              interface_nodes):
    """用内核 7 插入动网格层（接口顺序由调用方给出）。"""
    pairing = ref_bnd.pair_master_slave(node[:, 1:3], parts=4)
    return ref_mm.insert_moving_mesh_layer(
        node[:, 1:3],
        tri_of(element_S),
        interface_nodes,
        element_rotor_gap,
        pairing,
        element_stator=element_stator,
        element_rotor=element_rotor,
    )


def _build_pair(n_ang):
    """同一网格上同时造出 legacy 版、内核 7 版，以及**插入前**的原始数组。

    接口顺序取 legacy 的 set 迭代顺序（``node_gap_stator``），这样新节点编号与
    legacy 一致；``raw`` 留给需要在「插入前」起点上重复调用的测试
    （``insert_layer`` 会就地改写 legacy 状态）。
    """
    node, element_S, e_gap, e_stator, e_rotor = make_interface_mesh(n_ang=n_ang)
    g = LegacyState(node, element_S, e_gap, e_stator, e_rotor)
    interface = tuple(g.node_gap_stator)
    layer = _layer_of(node, element_S, e_gap, e_stator, e_rotor, interface)
    raw = (node, element_S, e_gap, e_stator, e_rotor, interface)
    g.insert_layer()
    return g, layer, raw


@pytest.fixture(scope="module")
def small():
    """小网格（接口 7 个节点）：(legacy 已插入状态, 内核 7 层, 插入前原始数组)。"""
    return _build_pair(7)


@pytest.fixture(scope="module")
def dense():
    """与真实网格同规模（接口 236 个节点）。"""
    return _build_pair(N_ANG_REAL)


@pytest.fixture(scope="module")
def legacy(small):
    return small[0]


@pytest.fixture(scope="module")
def layer(small):
    return small[1]


# =============================================================================
# 来源 A：插入层（几何性质）
# =============================================================================

def test_new_nodes_are_bit_exact_copies(layer):
    """新节点的坐标与它的源节点**逐位**相同（legacy 直接复制 x/y）。"""
    for i_new, i_old in layer.rotor_to_stator.items():
        assert np.array_equal(layer.xy[i_new], layer.xy[i_old]), i_new


def test_node_remap_is_a_bijection_onto_the_appended_ids():
    node, element_S, e_gap, e_stator, e_rotor = make_interface_mesh(n_ang=7)
    n_before = node.shape[0]
    interface = interface_set_legacy(node)
    layer = _layer_of(node, element_S, e_gap, e_stator, e_rotor, tuple(interface))

    assert layer.n_node == n_before + len(interface)
    assert set(layer.stator_to_rotor) == interface
    assert set(layer.stator_to_rotor.values()) == set(
        range(n_before, n_before + len(interface))
    )
    assert all(layer.rotor_to_stator[v] == k for k, v in layer.stator_to_rotor.items())
    assert layer.rotor_nodes == tuple(layer.rotor_to_stator)


def test_rotor_gap_elements_no_longer_touch_interface_nodes(layer):
    """接口圆上的节点在转子气隙单元里被整体换成新节点号 —— 定转子不再共享节点。"""
    _, _, e_gap, _, _ = make_interface_mesh(n_ang=7)
    interface = set(layer.interface_nodes)
    assert not (interface & set(layer.rotor_nodes))
    for i_elem in e_gap:
        for n in layer.tri[i_elem]:
            assert int(n) not in interface, (i_elem, int(n))


def test_node_stator_and_node_rotor_are_disjoint(layer):
    """动网格层的意义：``node_stator ∩ node_rotor = ∅``。"""
    assert layer.node_stator is not None and layer.node_rotor is not None
    assert not (layer.node_stator & layer.node_rotor)
    assert set(layer.rotor_nodes) <= layer.node_rotor
    assert set(layer.interface_nodes) <= layer.node_stator


def test_pairing_is_rebound_through_the_layer(layer, legacy):
    """重绑后的配对表逐键逐值等于 legacy 就地更新后的结果。"""
    assert layer.pairing.slave_to_master == legacy.node_slave
    assert layer.pairing.master_to_slave == legacy.node_master
    # 新节点确实进了从属表：θ = 90° 的那个动节点
    assert set(layer.rotor_nodes) & set(layer.pairing.slave_to_master)


def test_virtual_elements_incremental_equals_recompute(layer, legacy):
    """「新 tri + 重绑 pairing 重算」≡ legacy 的增量更新（``MMA10:471-479``）。"""
    recomputed = ref_bnd.slave_virtual_elements(layer.tri, layer.pairing)
    assert set(recomputed.elements) == set(legacy.element_S_slave.keys())
    for i_elem, virtual in recomputed.virtual_tri.items():
        assert tuple(virtual) == tuple(legacy.element_S_slave[i_elem][1:4]), i_elem


# =============================================================================
# 来源 A：插值行的解析性质
# =============================================================================

def _is_odd_sector(theta):
    """legacy 的 ``_other_region``：``mod(θ // w, 2) > 0.5``。"""
    return bool(np.mod(theta // SECTOR, 2) > 0.5)


def _row_psi(layer, i, pos):
    return np.arctan2(layer.xy[i, 1], layer.xy[i, 0]) + pos


def test_pos_zero_row_reproduces_the_source_node_value(layer):
    """``pos = 0``：动节点与它的源节点角重合 ⇒ 插值退化为恒等（静网格一致性）。

    除 θ = 90° 那个动节点外（它落在奇扇区），每个动节点都精确继承「它复制自的那个
    接口节点」的值。θ = 90° 的动节点则由行给出 ``A_i = −A(θ=0° 的节点)`` —— 正是
    内核 6 主从驱动行的内容，两条驱动行同值，不冲突。
    """
    K = ref_mm.rotating_boundary(layer, parts=4, pos=0.0, a_eq=False).matrix
    rng = np.random.default_rng(0)
    A = np.zeros(layer.n_node)
    for n in layer.interface_nodes:
        A[n] = rng.standard_normal()

    odd = [i for i in layer.rotor_nodes if _is_odd_sector(_row_psi(layer, i, 0.0))]
    assert len(odd) == 1, "只有 θ=90° 那个动节点的扇区序号是奇数"
    A_identity = A.copy()
    for i in layer.rotor_nodes:
        if i not in odd:
            A_identity[i] = A[layer.rotor_to_stator[i]]
    A_identity[odd[0]] = -A[_zero_angle_interface_node(layer)]

    residual = K @ A_identity
    scale = float(np.abs(A_identity).max())
    assert np.all(np.abs(residual) <= 1e-14 * scale)


def _zero_angle_interface_node(layer):
    """接口节点里角度恰为 0 的那个（θ = 90° 动节点的主动节点）。"""
    hits = [
        n for n in layer.interface_nodes
        if np.arctan2(layer.xy[n, 1], layer.xy[n, 0]) == 0.0
    ]
    assert len(hits) == 1
    return hits[0]


def test_pos_zero_rows_have_exactly_two_nonzeros(layer):
    """``pos = 0`` 时每行只有对角 + 一个接口节点，系数为 ±1（无插值拖尾）。"""
    K = ref_mm.rotating_boundary(layer, parts=4, pos=0.0, a_eq=False).matrix
    for i in layer.rotor_nodes:
        row = K.getrow(i)
        assert row.nnz == 2, i
        assert row[0, i] == 1.0
        off = [c for c in row.indices if c != i]
        assert len(off) == 1
        assert abs(row[0, off[0]]) == 1.0


def _exact_field(layer, c, pos, a_eq):
    """沿角度轴定义的、**在每个括号区间上线性**的自洽场。

    对角度 ψ 令 ``k = floor(ψ/w)``、``φ = ψ − k·w``：

    ``a_eq = False``（反周期）：**逐扇区改号的斜坡**
        ``A(ψ) = (−1)^k · c · (1 − 2φ/w)``
        它在每个扇区内部线性、在扇区边界处跳变；跳变与
        ``A(θ+w) = −A(θ)``（从属关系，``σ = −1``）以及奇扇区符号 ``s`` 自洽。

    ``a_eq = True``（周期）：**连续三角波（帐篷）**
        ``A(ψ) = c · |1 − 2φ/w|``
        它在 φ = w/2 处有一个折点，且 ``A(0) = A(w) = c``（与 ``A_slave =
        +A_master`` 自洽）。⚠ 折点必须**落在接口节点上**，否则它落在某个括号区间
        内部、该区间上不再是线性函数，不动点性质不成立 —— 本函数的调用者负责
        保证这一点（n_ang = 7 的网格上 w/2 = 45° 是节点）。

    线性插值对线性函数是精确的，所以这两种场都是全部动节点行的精确不动点。
    """
    def value(psi):
        k = int(np.floor(psi / SECTOR))
        phi = psi - k * SECTOR
        if a_eq:
            return c * abs(1 - 2 * phi / SECTOR)
        sign = 1.0 if k % 2 == 0 else -1.0
        return sign * c * (1 - 2 * phi / SECTOR)

    A = np.zeros(layer.n_node)
    for n in layer.interface_nodes:
        A[n] = value(np.arctan2(layer.xy[n, 1], layer.xy[n, 0]))
    for i in layer.rotor_nodes:
        A[i] = value(np.arctan2(layer.xy[i, 1], layer.xy[i, 0]) + pos)
    return A


@pytest.mark.parametrize("pos", POS_SAMPLES)
@pytest.mark.parametrize("a_eq", [False, True])
def test_exact_field_is_a_fixed_point(layer, pos, a_eq):
    """自洽的分段线性场是全部插值行的精确不动点。

    本内核最强的解析判据：一次验证权重 ``kp/kn``、奇扇区反周期符号 ``s``、以及
    从属替换 ``σ``。容差按场的量级缩放（每一项都是「±1 乘权重乘场值」）。
    """
    if a_eq:
        # 折点必须在节点上，否则区间上不再是线性函数（见 _exact_field）
        mid = SECTOR / 2
        assert any(
            np.arctan2(layer.xy[n, 1], layer.xy[n, 0]) == mid
            for n in layer.interface_nodes
        ), "帐篷场的折点必须落在接口节点上"
    res = ref_mm.rotating_boundary(layer, parts=4, pos=pos, a_eq=a_eq)
    A = _exact_field(layer, c=3.7, pos=pos, a_eq=a_eq)
    residual = res.matrix @ A
    scale = float(np.abs(A).max())
    assert np.all(np.abs(residual) <= 1e-13 * scale)


def test_ramp_test_is_not_vacuous(layer):
    """防退化：去掉逐扇区改号后，残差必须明显不为零。

    否则「斜坡场是精确不动点」在任何符号逻辑下都会通过，测试毫无意义。
    """
    pos = 1.0  # 动节点 θ ≈ 45° + 57.3° = 102.3° → 落在奇扇区
    K = ref_mm.rotating_boundary(layer, parts=4, pos=pos, a_eq=False).matrix
    A = np.zeros(layer.n_node)
    for n in layer.interface_nodes:
        psi = np.arctan2(layer.xy[n, 1], layer.xy[n, 0])
        A[n] = 3.7 * (1 - 2 * np.mod(psi, SECTOR) / SECTOR)
    for i in layer.rotor_nodes:
        psi = _row_psi(layer, i, pos)
        A[i] = 3.7 * (1 - 2 * np.mod(psi, SECTOR) / SECTOR)  # 缺 (−1)^k
    assert np.abs(K @ A).max() > 1.0


@pytest.mark.parametrize("pos", POS_SAMPLES)
def test_periodic_constant_field_satisfies_all_rows(layer, pos):
    """周期情形（``a_eq=True``）：常值场满足全部行（行和为零）。

    反周期情形下常值场不自洽（``A(θ+w) = −A(θ)``），行和不为零，所以这条判据只
    在周期分支成立。容差按 ``1e-14 · max|矩阵元| · max|场值|`` 缩放，不取固定值。
    """
    K = ref_mm.rotating_boundary(layer, parts=4, pos=pos, a_eq=True).matrix
    A = np.full(layer.n_node, 2.5)
    residual = K @ A
    scale = float(np.abs(K.data).max()) * float(np.abs(A).max())
    assert np.all(np.abs(residual) <= 1e-14 * scale)


def test_weights_match_hand_computed_angle_ratio(layer):
    """权重 = 手算角度比（独立于实现的搜索过程，故不看符号 —— 符号由上面两条判据管）。"""
    angles = _sorted_interface_angles(layer)
    pos = 0.05
    K = ref_mm.rotating_boundary(layer, parts=4, pos=pos, a_eq=True).matrix
    for i in layer.rotor_nodes:
        phi = np.mod(_row_psi(layer, i, pos), SECTOR)
        up, low = _hand_bracket_angles(angles, phi)
        kp_hand = (phi - low) / (up - low)
        row = K.getrow(i)
        coefs = sorted(float(abs(row[0, c])) for c in row.indices if c != i)
        assert coefs == pytest.approx(sorted([kp_hand, 1 - kp_hand]), rel=1e-12, abs=1e-15), i


def _sorted_interface_angles(layer):
    return sorted(
        float(np.arctan2(layer.xy[n, 1], layer.xy[n, 0])) for n in layer.interface_nodes
    )


def _hand_bracket_angles(angles, phi):
    """手算 φ 的上下括号角度。

    语义与 legacy 的 ``while θ[index] > φ: break`` 一致：``up`` 是第一个**严格大于**
    φ 的角度，``low`` 是最后一个**小于等于** φ 的角度（φ 恰在某个节点角上时
    ``low = φ``，于是 ``kp = 0``）。
    """
    up = next(a for a in angles if a > phi)
    low = max(a for a in angles if a <= phi)
    return up, low


# =============================================================================
# 来源 A：解析导数
# =============================================================================

def test_derivative_matches_hand_computed_slope(layer):
    """``dkp/dpos = 1/(θ_up − θ_low)``（周期分支，故两列系数同绝对值）。"""
    angles = _sorted_interface_angles(layer)
    pos = 0.05
    dK = ref_mm.rotating_boundary_with_derivative(
        layer, parts=4, pos=pos, a_eq=True
    ).d_matrix
    for i in layer.rotor_nodes:
        phi = np.mod(_row_psi(layer, i, pos), SECTOR)
        up, low = _hand_bracket_angles(angles, phi)
        dkp = 1.0 / (up - low)
        row = dK.getrow(i)
        coefs = sorted(float(abs(row[0, c])) for c in row.indices)
        assert coefs == pytest.approx([dkp, dkp], rel=1e-12), i


def test_derivative_diagonal_is_zero(layer):
    """``dK_rb`` 的对角恒为 0（对角元 1 与 pos 无关）。"""
    for a_eq in (False, True):
        dK = ref_mm.rotating_boundary_with_derivative(
            layer, parts=4, pos=0.31, a_eq=a_eq
        ).d_matrix
        assert np.array_equal(dK.diagonal(), np.zeros(layer.n_node))


@pytest.mark.parametrize("a_eq", [False, True])
@pytest.mark.parametrize("pos", [0.05, 0.91, 2.4, 3.7])
def test_derivative_matches_central_difference(layer, pos, a_eq):
    """解析导数与中心差分一致（独立于解析式的外部判据）。

    容差依据：在**括号节点不交换、``other_region`` 不翻转**的区间内，``K(pos)``
    是 ``pos`` 的仿射函数，所以中心差分**没有截断误差**，残差只剩「两个 O(1) 量
    相减」的舍入，被 ``1/(2ε)`` 放大。取 ε = 1e-6（远小于 0.26 rad 的接口角间距，
    故不跨段），放大倍数约 5e5、相对舍入 1e-16 ⇒ 绝对误差量级 5e-11。故
    ``atol = 1e-9`` 宽裕；不取更大的固定容差，以免真正的系数错误溜过。
    """
    eps = 1e-6
    dK = ref_mm.rotating_boundary_with_derivative(
        layer, parts=4, pos=pos, a_eq=a_eq
    ).d_matrix
    K_plus = ref_mm.rotating_boundary(layer, parts=4, pos=pos + eps, a_eq=a_eq).matrix
    K_minus = ref_mm.rotating_boundary(layer, parts=4, pos=pos - eps, a_eq=a_eq).matrix
    fd = (K_plus - K_minus) / (2 * eps)
    assert np.allclose(fd.toarray(), dK.toarray(), rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("pos", [0.4, 1.0, 2.2])
def test_derivative_ratio_is_one_over_phi_minus_theta_low(layer, pos):
    """非对角元的 ``(∂k/∂θ)/k`` 恰是 ``{1/(φ − θ_low), −1/(θ_up − φ)}``。

    这一条**与符号无关地**钉住「导数与 K 用的是同一套 ``s``/``σ``」：若导数漏了
    某个符号因子，比值不再是这两个数（上侧列给 ``∂kp/kp``、下侧列给 ``∂kn/kn``，
    两者符号相反），于是排序后比较会失败。
    """
    res = ref_mm.rotating_boundary_with_derivative(layer, parts=4, pos=pos, a_eq=False)
    K_dense = res.matrix.toarray()
    dK_dense = res.d_matrix.toarray()
    nz_k = set(zip(*np.nonzero(K_dense)))
    nz_d = set(zip(*np.nonzero(dK_dense)))
    assert nz_d and nz_d <= nz_k
    assert all(r == c for (r, c) in (nz_k - nz_d)), "K 只比 dK 多出对角元"
    angles = _sorted_interface_angles(layer)
    for r in layer.rotor_nodes:
        phi = np.mod(_row_psi(layer, r, pos), SECTOR)
        up, low = _hand_bracket_angles(angles, phi)
        expect = sorted([1.0 / (phi - low), -1.0 / (up - phi)])
        cols = np.nonzero(dK_dense[r])[0]
        got = sorted(float(dK_dense[r, c] / K_dense[r, c]) for c in cols)
        assert got == pytest.approx(expect, rel=1e-9), r


# =============================================================================
# 来源 A：几何前提（legacy 会 assert 失败或 IndexError）
# =============================================================================

def _minimal_layer(interface_angle, rotor_angle):
    """只有一个接口节点与一个动节点的最小层。

    用它而不是完整网格，是为了让「越界」这条路径**确定性地**触发：完整网格上
    每次求值要遍历全部动节点，哪个先炸取决于集合迭代顺序。
    """
    r = 0.05
    xy = np.array(
        [
            [r * np.cos(interface_angle), r * np.sin(interface_angle)],
            [r * np.cos(rotor_angle), r * np.sin(rotor_angle)],
        ]
    )
    return ref_mm.MovingMeshLayer(
        xy=xy,
        tri=np.array([[0, 0, 1]]),
        stator_to_rotor={0: 1},
        rotor_to_stator={1: 0},
        interface_nodes=(0,),
        rotor_nodes=(1,),
        pairing=ref_bnd.PeriodicBoundary({}, {}),
    )


def test_interface_must_cover_the_right_endpoint():
    """接口角缺右端点（φ ≥ θ_max）⇒ 没有上侧括号节点（legacy 会 IndexError）。"""
    layer = _minimal_layer(interface_angle=0.3, rotor_angle=0.4)
    with pytest.raises(ValueError, match="右端点"):
        ref_mm.rotating_boundary(layer, parts=4, pos=0.0, a_eq=False)


def test_interface_must_cover_the_left_endpoint():
    """接口角缺左端点（θ_min > φ）⇒ legacy 的 ``assert index_stator > 0`` 失败。"""
    layer = _minimal_layer(interface_angle=0.3, rotor_angle=0.0)
    with pytest.raises(ValueError, match="左端点"):
        ref_mm.rotating_boundary(layer, parts=4, pos=0.0, a_eq=False)


def test_missing_right_endpoint_is_really_an_index_error_in_legacy():
    """同一几何喂给 legacy 片段：确实是 IndexError（证明上面那条报错对应得上）。"""
    node = np.array([[0.0, 0.05 * np.cos(0.3), 0.05 * np.sin(0.3), 0.0],
                     [1.0, 0.05 * np.cos(0.4), 0.05 * np.sin(0.4), 0.0]])
    g = LegacyState(node, [[0, 0, 0, 1, 0]], set(), set(), set())
    g.node_gap_stator = {0: 1}
    g.node_gap_rotor = {1: 0}
    with pytest.raises(IndexError):
        legacy_rotating_boundary(g, 0.0)


def test_brackets_are_mapped_through_slave_remap(layer, legacy):
    """``brackets`` 里的节点号都做过从属→主动替换；θ=90° 的从属节点不出现。"""
    res = ref_mm.rotating_boundary(layer, parts=4, pos=0.0, a_eq=False)
    slave_ids = set(layer.pairing.slave_to_master)
    assert slave_ids & set(layer.rotor_nodes), "新层里确实有从属节点"
    for i, (up, low) in res.brackets.items():
        assert up not in slave_ids and low not in slave_ids
    assert res.brackets == legacy_rotating_boundary(legacy, 0.0)[1]


def test_degenerate_interface_duplicate_write_is_last_wins():
    """接口只有 2 个节点时上下括号映射到**同一**主节点 —— legacy 的 ``lil`` 是
    **后写覆盖**（不是相加），本实现用 ``dict`` 复现同一语义。"""
    node, element_S, e_gap, e_stator, e_rotor = make_interface_mesh(n_ang=2)
    layer = _layer_of(node, element_S, e_gap, e_stator, e_rotor,
                      tuple(interface_set_legacy(node)))
    pos = 0.7
    res = ref_mm.rotating_boundary(layer, parts=4, pos=pos, a_eq=False)
    checked = 0
    for i in layer.rotor_nodes:
        up, low = res.brackets[i]
        if up != low:
            continue
        row = res.matrix.getrow(i)
        assert row.nnz == 2, "对角 + 那一个主节点"
        phi = np.mod(_row_psi(layer, i, pos), SECTOR)
        kn = 1.0 - phi / SECTOR
        s = -1.0 if _is_odd_sector(_row_psi(layer, i, pos)) else 1.0
        # 后写的是下侧分支：−s·σ_low·kn，而 θ=0° 的主节点 σ = +1
        assert row[0, up] == pytest.approx(-s * kn, rel=1e-12, abs=1e-15)
        checked += 1
    assert checked > 0, "该几何下必须出现写冲突，否则这条测试没有测到东西"


# =============================================================================
# 来源 B：与 legacy 逐位一致
# =============================================================================

def test_matches_legacy_insert_layer(layer, legacy):
    """插入层的全部产出与 legacy 逐位一致。"""
    assert np.array_equal(layer.xy, legacy.node[:, 1:3])
    assert np.array_equal(layer.tri, np.array([_e[1:4] for _e in legacy.element_S]))
    assert set(layer.stator_to_rotor.items()) == set(legacy.node_gap_stator.items())
    assert set(layer.rotor_to_stator.items()) == set(legacy.node_gap_rotor.items())
    assert layer.node_stator == legacy.node_stator
    assert layer.node_rotor == legacy.node_rotor


@pytest.mark.parametrize("a_eq", [False, True])
@pytest.mark.parametrize("pos", POS_SAMPLES)
def test_matches_legacy_rotating_boundary(layer, legacy, a_eq, pos):
    legacy.EM_MS_boundary_A_eq = a_eq
    expected_K, expected_brackets = legacy_rotating_boundary(legacy, pos)
    got = ref_mm.rotating_boundary(layer, parts=4, pos=pos, a_eq=a_eq)
    assert np.array_equal(got.matrix.toarray(), expected_K.toarray())
    assert got.brackets == expected_brackets


@pytest.mark.parametrize("a_eq", [False, True])
@pytest.mark.parametrize("pos", POS_SAMPLES)
def test_matches_legacy_rotating_boundary_v2(layer, legacy, a_eq, pos):
    legacy.EM_MS_boundary_A_eq = a_eq
    expected_K, expected_brackets, expected_dK = legacy_rotating_boundary_v2(legacy, pos)
    got = ref_mm.rotating_boundary_with_derivative(
        layer, parts=4, pos=pos, a_eq=a_eq
    )
    assert np.array_equal(got.matrix.toarray(), expected_K.toarray())
    assert np.array_equal(got.d_matrix.toarray(), expected_dK.toarray())
    assert got.brackets == expected_brackets


@pytest.mark.parametrize("a_eq", [False, True])
@pytest.mark.parametrize("pos", [0.0, 0.3, 1.0, 2.0])
def test_sparse_structure_matches_legacy(layer, legacy, a_eq, pos):
    """稀疏**结构**（``indptr``/``indices``/``data``）也逐位一致。

    稠密比较看不出「显式存储的零」：``lil[i, j] = 0.0`` 会把该元丢掉（``pos = 0``
    时 ``kp = 0``，真的会发生），而 ``dict`` → COO 会留下一个存储零。数值上无害，
    但结构会进全局矩阵、进而进 SuperLU 的列排序 ⇒ 必须显式过滤零元。
    """
    legacy.EM_MS_boundary_A_eq = a_eq
    expected_K, _brackets, expected_dK = legacy_rotating_boundary_v2(legacy, pos)
    got = ref_mm.rotating_boundary_with_derivative(
        layer, parts=4, pos=pos, a_eq=a_eq
    )
    for name, expected, actual in (
        ("K", expected_K, got.matrix),
        ("dK", expected_dK, got.d_matrix),
    ):
        assert np.array_equal(actual.indptr, expected.indptr), name
        assert np.array_equal(actual.indices, expected.indices), name
        assert np.array_equal(actual.data, expected.data), name
    # pos = 0 时 kp = 0 的那一列确实没被存下来（否则上面这组断言会失败）
    if pos == 0.0 and not a_eq:
        assert got.matrix.nnz < 3 * len(layer.rotor_nodes)


@pytest.mark.parametrize("a_eq", [False, True])
@pytest.mark.parametrize("pos", [0.0, 0.3, 1.0, 2.0, 2 * SECTOR - 1e-9])
def test_matches_legacy_on_real_scale_interface(dense, a_eq, pos):
    """与真实网格同规模的接口（236 个节点）上逐位一致。"""
    legacy_dense, layer_dense, _raw = dense
    legacy_dense.EM_MS_boundary_A_eq = a_eq
    expected_K, expected_brackets, expected_dK = legacy_rotating_boundary_v2(
        legacy_dense, pos
    )
    got = ref_mm.rotating_boundary_with_derivative(
        layer_dense, parts=4, pos=pos, a_eq=a_eq
    )
    assert np.array_equal(got.matrix.toarray(), expected_K.toarray())
    assert np.array_equal(got.d_matrix.toarray(), expected_dK.toarray())
    assert got.brackets == expected_brackets


def test_sign_branches_are_all_exercised(layer):
    """防退化：``pos`` 采样必须真的覆盖 8 个符号分支，否则 B 级的符号部分是空转。

    独立地（不复用实现）从角度表 + 从属集合算出「原始括号节点是否从属」，这正是
    legacy ``if … in node_slave`` 分支的条件。
    """
    table = sorted(
        ((float(np.arctan2(layer.xy[n, 1], layer.xy[n, 0])), n)
         for n in layer.interface_nodes),
        key=lambda item: item[0],
    )
    angles = [a for (a, _n) in table]
    slave_ids = set(layer.pairing.slave_to_master)
    seen = set()
    for pos in POS_SAMPLES:
        for a_eq in (False, True):
            for i in layer.rotor_nodes:
                psi = _row_psi(layer, i, pos)
                phi = np.mod(psi, SECTOR)
                up, low = _hand_bracket_angles(angles, phi)
                i_up = next(n for (a, n) in table if a == up)
                i_low = next(n for (a, n) in table if a == low)
                seen.add((a_eq, _is_odd_sector(psi) and not a_eq,
                          i_up in slave_ids, i_low in slave_ids))

    assert (False, False, False, False) in seen, "反周期·偶扇区·括号均非从属"
    assert (False, True, False, False) in seen, "反周期·奇扇区（s = −1）"
    assert any(not aeq and not s and (su or sl)
               for (aeq, s, su, sl) in seen), "反周期·偶扇区·括号从属（σ = −1）"
    assert any(not aeq and s and (su or sl)
               for (aeq, s, su, sl) in seen), "反周期·奇扇区·括号从属（s·σ 同时）"
    assert (True, False, False, False) in seen, "周期·括号均非从属"
    assert any(aeq and (su or sl) for (aeq, _s, su, sl) in seen), "周期·括号从属"


# =============================================================================
# 来源 C：ref vs fast
# =============================================================================

@requires_fast
@pytest.mark.parametrize("a_eq", [False, True])
@pytest.mark.parametrize("pos", POS_SAMPLES)
def test_ref_fast_parity(layer, a_eq, pos):
    ref = ref_mm.rotating_boundary_with_derivative(layer, parts=4, pos=pos, a_eq=a_eq)
    fast = fast_mm.rotating_boundary_with_derivative(layer, parts=4, pos=pos, a_eq=a_eq)
    assert np.array_equal(ref.matrix.toarray(), fast.matrix.toarray())
    assert np.array_equal(ref.d_matrix.toarray(), fast.d_matrix.toarray())
    assert ref.brackets == fast.brackets


@requires_fast
def test_ref_fast_parity_insert_layer(dense):
    _legacy, layer, raw = dense
    node, element_S, e_gap, e_stator, e_rotor, interface = raw
    fast_layer = fast_mm.insert_moving_mesh_layer(
        node[:, 1:3],
        tri_of(element_S),
        interface,
        e_gap,
        ref_bnd.pair_master_slave(node[:, 1:3], parts=4),
        element_stator=e_stator,
        element_rotor=e_rotor,
    )
    assert np.array_equal(layer.xy, fast_layer.xy)
    assert np.array_equal(layer.tri, fast_layer.tri)
    assert layer.stator_to_rotor == fast_layer.stator_to_rotor
    assert layer.rotor_to_stator == fast_layer.rotor_to_stator
    assert layer.rotor_nodes == fast_layer.rotor_nodes
    assert layer.node_stator == fast_layer.node_stator
    assert layer.node_rotor == fast_layer.node_rotor
    assert layer.pairing.slave_to_master == fast_layer.pairing.slave_to_master


@requires_fast
@pytest.mark.parametrize("a_eq", [False, True])
@pytest.mark.parametrize("pos", [0.0, 0.3, 1.0, 2 * SECTOR - 1e-9])
def test_ref_fast_parity_sparse_structure(dense, a_eq, pos):
    """ref 与 fast 的**稀疏结构**也逐位一致（含 ``pos = 0`` 的零元丢弃）。"""
    _legacy, layer, _raw = dense
    ref = ref_mm.rotating_boundary_with_derivative(layer, parts=4, pos=pos, a_eq=a_eq)
    fast = fast_mm.rotating_boundary_with_derivative(layer, parts=4, pos=pos, a_eq=a_eq)
    for name, expected, actual in (
        ("K", ref.matrix, fast.matrix),
        ("dK", ref.d_matrix, fast.d_matrix),
    ):
        assert np.array_equal(actual.indptr, expected.indptr), name
        assert np.array_equal(actual.indices, expected.indices), name
        assert np.array_equal(actual.data, expected.data), name


@requires_fast
def test_fast_layer_is_not_slower_than_reference(dense, capsys):
    """性能守卫（宽松）：向量化版不应慢于逐节点循环 —— 这正是它存在的理由。

    阈值只要求「不更慢」，并打印实测倍速。定时的绝对值随机器负载波动，所以
    另取多轮的**最小值**比较，避免把噪声当成回归。
    """
    import timeit

    _legacy, layer, _raw = dense
    n_rotor = len(layer.rotor_nodes)

    def bench(fn):
        return min(
            timeit.repeat(
                lambda: fn(layer, parts=4, pos=0.7, a_eq=False), number=3, repeat=5
            )
        ) / 3

    t_ref = bench(ref_mm.rotating_boundary_with_derivative)
    t_fast = bench(fast_mm.rotating_boundary_with_derivative)
    with capsys.disabled():
        print(
            f"\n[性能守卫] 接口 {n_rotor} 个动节点：逐节点 {t_ref * 1e3:.2f} ms，"
            f"向量化 {t_fast * 1e3:.2f} ms，加速 {t_ref / t_fast:.1f}×"
        )
    assert t_fast < t_ref


# =============================================================================
# P2（6 节点）推广：来源 A 解析性质（2026-10-02，路线 A，Design B）
# =============================================================================
#
# 设计（moving_mesh.py 模块 docstring 第七节 + assembly_p2.py Design B 段）：
# - 接口节点表 = 顶点 + 接口圆边中点（调用方只传半径判据顶点集，中点由插入层
#   从连接表推导补全）；
# - 每个转子动节点（顶点 *和* 中点）走自己的插值行；pos=0 时退化为恒等。
#
# 注意：P2 无 legacy 逐位约束（fast/ 层也未落地），本节的判据全部是来源 A 解析
# 性质，且对顶点与中点同时成立 —— 这是「中点不是端点平均」的 Design B 的测试面。


def make_interface_mesh_p2(n_ang=7, radii=R_RINGS):
    """P2 版环扇网格：6 节点三角形（3 顶点 + 3 边中点，gmsh 列序）。

    顶点编号与 P1 同款（``层 * n_ang + 角``）；边中点编号追加在顶点之后，
    共享边同一编号。区域号同 P1（0=转子铁芯、1=转子气隙、2=定子气隙）。
    """
    angles = np.linspace(0.0, SECTOR, n_ang)
    n_ring = len(radii)
    xy = [[r * np.cos(t), r * np.sin(t)] for r in radii for t in angles]
    tri3, regs = [], []
    for k in range(n_ring - 1):
        for l in range(n_ang - 1):
            a = k * n_ang + l
            b = (k + 1) * n_ang + l
            c = (k + 1) * n_ang + l + 1
            d = k * n_ang + l + 1
            tri3.append((a, b, c)); regs.append(k)
            tri3.append((a, c, d)); regs.append(k)
    n_vertex = len(xy)
    edge_to_mid = {}

    def _mid_of(va, vb):
        key = (min(int(va), int(vb)), max(int(va), int(vb)))
        if key not in edge_to_mid:
            m = n_vertex + len(edge_to_mid)
            edge_to_mid[key] = m
            xa, ya = xy[va]; xb, yb = xy[vb]
            xy.append(((xa + xb) / 2.0, (ya + yb) / 2.0))
        return edge_to_mid[key]

    element_S = []
    for (a, b, c), reg in zip(tri3, regs):
        element_S.append([len(element_S), a, b, c,
                          _mid_of(a, b), _mid_of(b, c), _mid_of(c, a), reg])
    node = np.column_stack(
        [np.arange(len(xy), dtype=float), np.asarray(xy), np.zeros(len(xy))]
    )
    element_rotor_gap = set(_e[0] for _e in element_S if _e[7] == 1)
    element_stator = set(_e[0] for _e in element_S if _e[7] == 2)
    element_rotor = set(_e[0] for _e in element_S if _e[7] in (0, 1))
    return node, element_S, element_rotor_gap, element_stator, element_rotor


def tri6_of(element_S):
    """element_S（8 列）→ (N_elem, 6) 连接表（本内核 P2 的输入约定）。"""
    return np.array([_e[1:7] for _e in element_S], dtype=int)


def _p2_midpoint_edge(tri, mid):
    """在 6 列连接表里找中点号 ``mid`` 的边两端点（共享边只取第一次命中）。"""
    for e in range(tri.shape[0]):
        for k in range(3):
            if int(tri[e, 3 + k]) == mid:
                return int(tri[e, k]), int(tri[e, (k + 1) % 3])
    raise AssertionError(f"中点 {mid} 不在连接表里")


def _p2_midpoint_copies(layer):
    """转子动节点里那些**源节点不是接口圆顶点**的（即边中点复制）。"""
    r_iface = R_RINGS[2]
    return [
        i for i in layer.rotor_nodes
        if abs(np.hypot(layer.xy[layer.rotor_to_stator[i], 0],
                        layer.xy[layer.rotor_to_stator[i], 1]) - r_iface) > 1e-9
    ]


@pytest.fixture(scope="module")
def layer_p2():
    """P2 小网格层：接口 7 顶点 + 6 圆边中点（半径判据只给顶点，中点由插入层推导）。

    与 ``small``（P1）同规模同几何，只是 6 节点。
    """
    node, element_S, e_gap, e_stator, e_rotor = make_interface_mesh_p2(n_ang=7)
    interface = tuple(interface_set_legacy(node))
    pairing = ref_bnd.pair_master_slave(node[:, 1:3], parts=4)
    return ref_mm.insert_moving_mesh_layer(
        node[:, 1:3], tri6_of(element_S), interface, e_gap, pairing,
        element_stator=e_stator, element_rotor=e_rotor,
    )


# --- 插入层：顶点 + 边中点一起复制 -------------------------------------------------


def test_p2_layer_copies_vertices_and_midpoints(layer_p2):
    """接口 = 7 顶点 + 6 圆边中点，全部被复制；坐标逐位相同。"""
    n_ang = 7
    assert len(layer_p2.interface_nodes) == n_ang + (n_ang - 1)
    assert len(layer_p2.rotor_nodes) == len(layer_p2.interface_nodes)
    for i_new, i_old in layer_p2.rotor_to_stator.items():
        assert np.array_equal(layer_p2.xy[i_new], layer_p2.xy[i_old]), i_new
    # 接口集里含 6 个「半径 < 接口圆」的源节点（弦中点，半径判据收不到）
    r_iface = R_RINGS[2]
    mid_srcs = [
        n for n in layer_p2.interface_nodes
        if abs(np.hypot(layer_p2.xy[n, 0], layer_p2.xy[n, 1]) - r_iface) > 1e-9
    ]
    assert len(mid_srcs) == n_ang - 1, "只有圆边中点会低于接口半径"
    assert all(n not in layer_p2.rotor_nodes for n in mid_srcs)


def test_p2_rotor_gap_elements_avoid_interface_nodes(layer_p2):
    """接口节点（顶点 + 中点）在转子气隙单元的 **6 个槽位**里都被换成新节点号。"""
    _, _, e_gap, _, _ = make_interface_mesh_p2(n_ang=7)
    interface = set(layer_p2.interface_nodes)
    assert not (interface & set(layer_p2.rotor_nodes))
    for i_elem in e_gap:
        for n in layer_p2.tri[i_elem]:
            assert int(n) not in interface, (i_elem, int(n))


def test_p2_node_stator_rotor_disjoint(layer_p2):
    """动网格层的意义对 P2 同样成立：``node_stator ∩ node_rotor = ∅``。"""
    assert layer_p2.node_stator is not None and layer_p2.node_rotor is not None
    assert not (layer_p2.node_stator & layer_p2.node_rotor)
    assert set(layer_p2.rotor_nodes) <= layer_p2.node_rotor
    assert set(layer_p2.interface_nodes) <= layer_p2.node_stator


def test_p2_pairing_is_rebound_through_the_layer(layer_p2):
    """重绑后 θ=90° 的动节点（顶点复制）进了从属表。"""
    assert set(layer_p2.rotor_nodes) & set(layer_p2.pairing.slave_to_master)


# --- 插值行：顶点与中点同时满足的解析性质 -------------------------------------------------


def test_p2_pos_zero_identity_for_vertices_and_midpoints(layer_p2):
    """``pos = 0``：每个动节点（顶点 *和* 中点）精确继承它复制自的那个接口节点。

    θ=90° 的动节点（顶点复制）落在奇扇区，由行给出 ``A_i = −A(θ=0°)``（反周期），
    与 P1 同款。中点动节点的源是定子边中点 —— 这说明中点行不是端点平均。
    """
    K = ref_mm.rotating_boundary(layer_p2, parts=4, pos=0.0, a_eq=False).matrix
    rng = np.random.default_rng(0)
    A = np.zeros(layer_p2.n_node)
    for n in layer_p2.interface_nodes:
        A[n] = rng.standard_normal()

    odd = [i for i in layer_p2.rotor_nodes if _is_odd_sector(_row_psi(layer_p2, i, 0.0))]
    assert len(odd) == 1, "只有 θ=90° 那个动节点（顶点复制）是奇扇区"
    A_identity = A.copy()
    for i in layer_p2.rotor_nodes:
        if i not in odd:
            A_identity[i] = A[layer_p2.rotor_to_stator[i]]
    A_identity[odd[0]] = -A[_zero_angle_interface_node(layer_p2)]

    residual = K @ A_identity
    scale = float(np.abs(A_identity).max())
    assert np.all(np.abs(residual) <= 1e-14 * scale)


def test_p2_pos_zero_rows_have_exactly_two_nonzeros(layer_p2):
    """``pos = 0`` 时每行（顶点 + 中点）只有对角 + 一个接口节点，系数 ±1。"""
    K = ref_mm.rotating_boundary(layer_p2, parts=4, pos=0.0, a_eq=False).matrix
    for i in layer_p2.rotor_nodes:
        row = K.getrow(i)
        assert row.nnz == 2, i
        assert row[0, i] == 1.0
        off = [c for c in row.indices if c != i]
        assert len(off) == 1
        assert abs(row[0, off[0]]) == 1.0


@pytest.mark.parametrize("pos", POS_SAMPLES)
@pytest.mark.parametrize("a_eq", [False, True])
def test_p2_exact_field_is_a_fixed_point(layer_p2, pos, a_eq):
    """自洽的分段线性场（斜坡/帐篷）是**全部**插值行的精确不动点（含中点行）。

    角表因含中点而加密，但只要场在相邻表项之间线性（斜坡恒线性；帐篷的折点
    在 45° 顶点上），插值就仍精确。
    """
    if a_eq:
        mid = SECTOR / 2
        assert any(
            np.arctan2(layer_p2.xy[n, 1], layer_p2.xy[n, 0]) == mid
            for n in layer_p2.interface_nodes
        ), "帐篷场的折点必须落在接口节点上"
    res = ref_mm.rotating_boundary(layer_p2, parts=4, pos=pos, a_eq=a_eq)
    A = _exact_field(layer_p2, c=3.7, pos=pos, a_eq=a_eq)
    residual = res.matrix @ A
    scale = float(np.abs(A).max())
    assert np.all(np.abs(residual) <= 1e-13 * scale)


@pytest.mark.parametrize("pos", POS_SAMPLES)
def test_p2_periodic_constant_field_satisfies_all_rows(layer_p2, pos):
    """周期情形：常值场满足全部行（含中点行，行和为零）。"""
    K = ref_mm.rotating_boundary(layer_p2, parts=4, pos=pos, a_eq=True).matrix
    A = np.full(layer_p2.n_node, 2.5)
    residual = K @ A
    scale = float(np.abs(K.data).max()) * float(np.abs(A).max())
    assert np.all(np.abs(residual) <= 1e-14 * scale)


def test_p2_weights_match_hand_computed_angle_ratio(layer_p2):
    """权重 = 手算角度比（角表含中点 ⇒ 顶点行的括号可能是定子中点）。"""
    angles = _sorted_interface_angles(layer_p2)
    pos = 0.05
    K = ref_mm.rotating_boundary(layer_p2, parts=4, pos=pos, a_eq=True).matrix
    for i in layer_p2.rotor_nodes:
        phi = np.mod(_row_psi(layer_p2, i, pos), SECTOR)
        up, low = _hand_bracket_angles(angles, phi)
        kp_hand = (phi - low) / (up - low)
        row = K.getrow(i)
        coefs = sorted(float(abs(row[0, c])) for c in row.indices if c != i)
        assert coefs == pytest.approx(sorted([kp_hand, 1 - kp_hand]), rel=1e-12, abs=1e-15), i


def test_p2_midpoint_row_is_own_bracket_not_endpoint_average(layer_p2):
    """Design B 的防退化：中点行的权重 = 自身 bracket（θ_mid + pos 的角度比），
    **不是**两端点行的平均（否则中点行的两个系数是端点两行的混合，权重对不上）。"""
    angles = _sorted_interface_angles(layer_p2)
    pos = 0.05
    res = ref_mm.rotating_boundary(layer_p2, parts=4, pos=pos, a_eq=True)
    mids = _p2_midpoint_copies(layer_p2)
    assert len(mids) > 0
    for i in mids:
        phi = np.mod(_row_psi(layer_p2, i, pos), SECTOR)
        up, low = _hand_bracket_angles(angles, phi)
        kp = (phi - low) / (up - low)
        row = res.matrix.getrow(i)
        coefs = sorted(float(abs(row[0, c])) for c in row.indices if c != i)
        assert coefs == pytest.approx(sorted([kp, 1 - kp]), rel=1e-12), i
        # 防退化：中点行 ≠ 两端点行的平均（两端点角 ≠ 中点角 ⇒ 权重必不同）
        va, vb = _p2_midpoint_edge(layer_p2.tri, i)
        row_avg = (res.matrix.getrow(va).toarray() + res.matrix.getrow(vb).toarray()) / 2.0
        assert not np.allclose(row.toarray(), row_avg), i


def test_p2_brackets_include_midpoints_and_no_slaves(layer_p2):
    """``brackets`` 含中点动节点；up/low 都做过从属→主动替换（不含从属号）。"""
    res = ref_mm.rotating_boundary(layer_p2, parts=4, pos=0.3, a_eq=False)
    mids = _p2_midpoint_copies(layer_p2)
    assert mids
    for i in mids:
        assert i in res.brackets, i
    slave_ids = set(layer_p2.pairing.slave_to_master)
    for i, (up, low) in res.brackets.items():
        assert up not in slave_ids and low not in slave_ids


def test_p2_driving_rows_cover_all_rotor_nodes(layer_p2):
    """P2 动节点（顶点 + 中点）全被约束行覆盖 —— 否则全局矩阵在动节点行上奇异。

    照搬消费方（``probe_torque_vs_ansys.py``）的 ``row_rep`` 构造。
    """
    rot = ref_mm.rotating_boundary(layer_p2, parts=4, pos=0.3, a_eq=False)
    row_rep = {int(n): np.asarray(rot.matrix[n, :].todense()).ravel()
               for n in layer_p2.rotor_nodes}
    assert set(row_rep) == set(layer_p2.rotor_nodes)
    for n in layer_p2.rotor_nodes:
        assert rot.matrix[n, n] == 1.0, n          # 对角 1 在
        assert rot.matrix[n, :].nnz >= 2, n        # 至少对角 + 一个括号节点


# --- 解析导数（顶点与中点同时满足） -------------------------------------------------


def test_p2_derivative_diagonal_is_zero(layer_p2):
    """``dK_rb`` 的对角恒为 0（含中点行）。"""
    for a_eq in (False, True):
        dK = ref_mm.rotating_boundary_with_derivative(
            layer_p2, parts=4, pos=0.31, a_eq=a_eq
        ).d_matrix
        assert np.array_equal(dK.diagonal(), np.zeros(layer_p2.n_node))


@pytest.mark.parametrize("a_eq", [False, True])
@pytest.mark.parametrize("pos", [0.05, 0.91, 2.4, 3.7])
def test_p2_derivative_matches_central_difference(layer_p2, pos, a_eq):
    """解析导数与中心差分一致（含中点行，角表加密后括号切换点更密仍不跨段）。"""
    eps = 1e-6
    dK = ref_mm.rotating_boundary_with_derivative(
        layer_p2, parts=4, pos=pos, a_eq=a_eq
    ).d_matrix
    K_plus = ref_mm.rotating_boundary(layer_p2, parts=4, pos=pos + eps, a_eq=a_eq).matrix
    K_minus = ref_mm.rotating_boundary(layer_p2, parts=4, pos=pos - eps, a_eq=a_eq).matrix
    fd = (K_plus - K_minus) / (2 * eps)
    assert np.allclose(fd.toarray(), dK.toarray(), rtol=1e-8, atol=1e-9)
