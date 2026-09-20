"""Basic verification tests (spec section 19)."""

import io
import json

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from models.linear_attention import LinearAttentionState
from models.softmax_attention import SoftmaxReference
from experiments.associative_recall import run_associative_recall
from utils.generators import seeded_rng
from utils.plotting import plot_accuracy_vs_bindings


def test_state_update():
    s = LinearAttentionState(8)
    k = np.ones(8)
    v = np.ones(8)
    s.update(k, v)
    assert s.count == 1
    assert s.state_norm > 0


def test_softmax_retrieves_known_value():
    sm = SoftmaxReference()
    rng = seeded_rng(0)
    keys = [rng.standard_normal(8) for _ in range(3)]
    vals = [np.eye(8)[i] for i in range(3)]
    for k, v in zip(keys, vals):
        sm.update(k, v)
    out = sm.query(keys[1])
    assert int(np.argmax(out)) == 1


def test_linear_retrieves_single_value():
    lin = LinearAttentionState(8)
    rng = seeded_rng(1)
    k = rng.standard_normal(8)
    v = np.zeros(8)
    v[3] = 1.0
    lin.update(k, v)
    out = lin.query(k)
    assert int(np.argmax(out)) == 3


def _strip_runtime(rows):
    return [{k: v for k, v in r.items() if k != "runtime_s"} for r in rows]


def test_seed_reproducibility():
    rows1 = run_associative_recall([4, 8], dim=8, trials=2, seed=42)
    rows2 = run_associative_recall([4, 8], dim=8, trials=2, seed=42)
    assert _strip_runtime(rows1) == _strip_runtime(rows2)


def test_csv_export():
    df = pd.DataFrame([{"m": 4, "linear_mean": 0.9}])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    df2 = pd.read_csv(buf)
    assert df2.iloc[0]["m"] == 4


def test_json_export():
    payload = {"config": {"seed": 0}, "rows": [{"m": 4}]}
    s = json.dumps(payload)
    assert json.loads(s)["config"]["seed"] == 0


def test_plots_generate():
    fig = plot_accuracy_vs_bindings([2, 4], [1.0, 0.8], [1.0, 1.0])
    assert fig is not None
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    assert buf.tell() > 0


def test_safety_limits():
    from app import MAX_DIM, MAX_TRIALS, MAX_SEQ_DEFAULT
    assert MAX_DIM <= 128
    assert MAX_TRIALS <= 50
    assert MAX_SEQ_DEFAULT <= 32768


def test_dimensionality_sweep():
    from experiments.associative_recall import run_dimensionality_sweep
    rows = run_dimensionality_sweep([2, 4], dim_values=(8, 16), trials=1, seed=42)
    assert len(rows) == 4
    assert sorted({r["dim"] for r in rows}) == [8, 16]


def test_high_res_save(tmp_path):
    from utils.plotting import plot_simple, save_high_res
    import os
    fig = plot_simple([1, 2], [0.5, 0.25], "x", "y", "t")
    p = save_high_res(fig, tmp_path / "vid.png", dpi=300, dark_mode=True)
    assert os.path.getsize(p) > 0


def test_exp4_softmax_baseline_smoke():
    from experiments.softmax_baseline import run_single_trial, run_softmax_baseline
    r = run_single_trial(32, 8, seed=42)
    assert r["linear_bytes"] < r["softmax_bytes"]  # fixed vs growing storage
    assert set(("linear_recovered", "softmax_recovered",
                "linear_time_s", "softmax_time_s")) <= set(r)
    rows = run_softmax_baseline([32], dim=8, trials=2, seed=42,
                                position_fracs=(0.5,))
    assert len(rows) == 1 and rows[0]["N"] == 32
    assert 0.0 <= rows[0]["softmax_accuracy"] <= 1.0


def test_gated_recall_smoke():
    from experiments.gated_recall import run_gated_trial, run_gamma_sweep
    r = run_gated_trial(64, 8, seed=42, gamma=0.99)
    assert set(("recovered", "state_norm")) <= set(r)
    rows = run_gamma_sweep(gammas=(1.0, 0.99), N=64, dim=8, trials=1, seed=42)
    assert len(rows) == 2 and rows[0]["gamma"] == 1.0


def test_mitigations_smoke():
    from experiments.mitigations import (run_hybrid_trial, run_chunked_trial,
                                         run_selective_trial,
                                         run_mixed_precision)
    assert "recovered" in run_hybrid_trial(64, 8, seed=42, window=8)
    assert "recovered" in run_chunked_trial(64, 8, seed=42, chunk=32)
    assert "recovered" in run_selective_trial(64, 8, seed=42, mode="oracle")
    assert len(run_mixed_precision(n_steps=10)) == 2


def test_dissipation_smoke():
    from experiments.dissipation import (run_dissipation_trial, run_mN_trial,
                                         run_orth_trial)
    r = run_dissipation_trial(64, 8, seed=42)
    assert set(("coarse_recovered", "fine_recovered")) <= set(r)
    assert 0.0 <= run_mN_trial(4, 16, 8, seed=42)["accuracy"] <= 1.0
    assert "linear_accuracy" in run_orth_trial(4, 8, seed=42)


def test_centered_map_zero_mean():
    import numpy as np
    from models.linear_attention import FEATURE_MAPS
    rng = np.random.default_rng(0)
    m = FEATURE_MAPS["centered_elu"](rng.standard_normal(200_000)).mean()
    assert abs(float(m)) < 0.02


def test_variants_smoke():
    from experiments.variants import (run_perchannel_trial,
                                      run_novelty_trial, run_multihead_trial,
                                      run_compress_trial, run_shift_trial,
                                      run_noisy_trial)
    assert "recovered" in run_perchannel_trial(64, 8, seed=42)
    assert 0.0 <= run_novelty_trial(4, 8, seed=42)["accuracy"] <= 1.0
    assert 0.0 <= run_multihead_trial(4, 64, seed=42, n_heads=4)["accuracy"] <= 1.0
    assert "recovered" in run_compress_trial(64, 8, seed=42)
    assert "recovered" in run_shift_trial(64, 8, seed=42)
    r = run_noisy_trial(32, 8, seed=42)
    assert set(("linear_recovered", "softmax_recovered")) <= set(r)


def test_finalbatch_smoke():
    from experiments.finalbatch import (run_gated_msweep, run_gate_msweep,
                                        run_survival, run_cosine_sweep,
                                        run_bf16, run_chunk_summary,
                                        run_stacked, run_random_gate,
                                        run_frozen_calib)
    assert len(run_gated_msweep(m_values=(2, 4), trials=1)[0]) == 6
    assert len(run_gate_msweep(m_values=(2, 4), trials=1)[0]) == 4
    assert len(run_survival(m=4, trials=2)[0]) == 5
    assert len(run_cosine_sweep(N_values=(32,), trials=1)[0]) == 1
    assert len(run_bf16(n_steps=10)[0]) >= 2
    assert len(run_chunk_summary(N=128, trials=1)[0]) == 6
    assert len(run_stacked(N=128, trials=1)[0]) == 8
    assert len(run_random_gate(N_values=(64,), trials=1)[0]) == 3
    assert len(run_frozen_calib(N=128, trials=1)[0]) == 2
