"""Probe-only experiment contracts; no numerical kernels changed."""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from motorfem.config import RunConfig

_spec = importlib.util.spec_from_file_location(
    "probe_optimizers", Path(__file__).resolve().parents[2] / "tools/scripts/probe_optimizers.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_single_flux_after_sidecar():
    cfg = RunConfig()
    probe._apply_motor(cfg, "6s4p")
    phase = cfg.time.phaseA_offset_deg
    probe._single_flux(cfg)
    assert cfg.time.N_point_simulation == 1
    assert cfg.time.sampling_mode == "uniform"
    assert not cfg.time.use_validation_set
    assert cfg.time.rotor_position0_deg == 22.5
    assert cfg.time.phaseA_offset_deg == phase == -29.827
    oc = cfg.objective
    assert oc.objective_role == "con3" and oc.mean_torque_source == "flux"
    assert oc.enable_con2 and oc.enable_con3
    assert not any((oc.enable_con1, oc.enable_con4, oc.enable_stress,
                    oc.enable_uelastic, oc.constraint_schedule, oc.p_exp_continuation))
    assert cfg.topology.max_density == 0.65
    assert oc.objective_role_phase2 is None
    # Opt-in only: an independent legacy config remains untouched.
    legacy = RunConfig.legacy()
    assert legacy.time.N_point_simulation == 11
    assert legacy.objective.enable_con1 and legacy.objective.enable_con4
    assert legacy.objective.enable_uelastic and legacy.objective.enable_stress


def test_flux_copy_checkpoint_pairing(tmp_path):
    call = SimpleNamespace(torque_flux_energy_vec=np.array([3.0]), f=np.array([88., 1.]))
    x = np.array([.4, .6])
    t, d, copied = probe._flux_record(call, x)
    call.torque_flux_energy_vec = None
    x[:] = 0
    assert t == 3 and d == 1
    np.testing.assert_array_equal(copied, [.4, .6])
    designs = [copied, np.array([.5, .5]), np.array([.6, .4])]
    snap = probe._flux_snapshot([3., 5., 4.], [1., 0., -1.], designs)
    assert snap["best_feasible_index"] == 1
    np.testing.assert_array_equal(snap["x_best_feasible"], designs[1])
    args = argparse.Namespace(steps=3, problem="con3", sampling="uniform", adam_sample=None)
    probe._dump(tmp_path / "experiment", {"mma": snap}, 0, args)
    with np.load(tmp_path / "experiment_chk.npz", allow_pickle=False) as data:
        assert data["mma__T_flux_best_feasible"] == 5
        np.testing.assert_array_equal(data["mma__x_last_evaluated"], designs[-1])
        assert "_args_json" in data
