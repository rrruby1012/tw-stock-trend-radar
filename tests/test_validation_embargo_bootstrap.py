"""Phase 2 第 5、6 項新增測試：fit/cal embargo 與日期區塊 bootstrap／四基準分離。

只新增本檔，不改任何既有測試檔。對應任務：
(a) fit/cal 切分間隔 >= embargo 交易日；
(b) bootstrap CI 輸出形狀正確（上下限、有正確的鍵名）；
(c) 四個基準確實分離計算（各有獨立結果，不互相覆蓋）；
(d) 氣候值無測試期洩漏（測試期基準率與訓練期基準率不同的假資料，
    斷言氣候基準用的是訓練期值）；
(e) 固定 0.5 基準每 fold AUC 恆為 0.5。
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.validation import (
    _date_blocks,
    block_bootstrap_ci,
    separate_baselines,
    split_fit_cal_dates,
    walk_forward,
)


def _fake_oos(n_dates=60, n_stocks=2, train_rates=(0.3,), pos_per_10=(7,)):
    """合成樣本外資料：每折列數皆為 10 的倍數，故每折基準率恰為 pos_per_10/10。

    train_rates：每折的訓練窗基準率（氣候值），與測試期基準率刻意不同。
    pos_per_10：每 10 列中的正類數（決定該折測試期基準率）。
    """
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    fold_len = n_dates // len(train_rates)
    rows = []
    idx = 0
    for d_i, d in enumerate(dates):
        fold = min(d_i // fold_len, len(train_rates) - 1)
        pos = pos_per_10[fold % len(pos_per_10)]
        for _ in range(n_stocks):
            y = 1.0 if (idx % 10) < pos else 0.0
            rows.append({
                "stock_id": f"S{idx % n_stocks}",
                "date": d,
                "fold": fold,
                "y_true": y,
                "proba": 0.7 if y == 1.0 else 0.3,
                "proba_mom": 0.6 if idx % 2 == 0 else 0.4,
                "proba_clim": train_rates[fold],
                "proba_always_rule": 1.0,
                "proba_const05": 0.5,
            })
            idx += 1
    return pd.DataFrame(rows)


# ---------------- (a) fit/cal embargo ----------------

@pytest.mark.parametrize("embargo", [20, 5])
def test_split_fit_cal_dates_gap_exact_and_disjoint(embargo):
    # 150 天：80% fit=120 天，空 embargo 天後仍剩 cal 天數（100 天不夠放 20 天 embargo）
    dates = pd.bdate_range("2024-01-02", periods=150)
    fit, cal = split_fit_cal_dates(dates, fit_fraction=0.8, embargo_days=embargo)
    assert len(fit) == 120
    assert len(set(fit) & set(cal)) == 0, "fit 與 cal 不得重疊"
    pos = {d: i for i, d in enumerate(dates)}
    gap = pos[cal[0]] - pos[fit[-1]] - 1
    assert gap == embargo, f"fit/cal 間隔應恰為 {embargo} 個交易日，實為 {gap}"


def test_split_fit_cal_dates_embargo_eats_cal_returns_empty_not_fallback():
    """embargo 吃掉全部 cal 時回傳空 cal，呼叫方必須跳過，不得靜默回退相鄰切分。"""
    dates = pd.bdate_range("2024-01-02", periods=100)
    fit, cal = split_fit_cal_dates(dates, fit_fraction=0.8, embargo_days=30)
    assert len(fit) == 80
    assert len(cal) == 0
    with pytest.raises(ValueError):
        split_fit_cal_dates(dates, embargo_days=-1)


def test_walk_forward_fit_cal_gap_ge_embargo(panel_small, small_cfg):
    """整合：walk_forward 每期間的 fit_end/cal_start 間隔 >= 設定 embargo。"""
    results = walk_forward(panel_small, F.feature_columns(panel_small), small_cfg)
    embargo = small_cfg["model"]["validation"]["embargo_days"]
    target = results["targets"]["up_10"]
    assert len(target["periods"]) >= 2
    panel_dates = list(
        pd.DatetimeIndex(sorted(panel_small.index.get_level_values("date").unique()))
    )
    pos = {d: i for i, d in enumerate(panel_dates)}
    for p in target["periods"]:
        assert p["fit_cal_embargo_days"] == embargo
        gap = pos[pd.Timestamp(p["cal_start"])] - pos[pd.Timestamp(p["fit_end"])] - 1
        assert gap >= embargo, f"fit/cal embargo 不足：{p['fit_end']} ~ {p['cal_start']}"
    oos = target["oos"]
    for col in ["fold", "proba_const05", "proba_clim", "proba_always_rule"]:
        assert col in oos.columns, f"oos 缺少 {col} 欄"
    assert isinstance(target["bootstrap_ci"]["gate_pass"], bool)


def test_train_final_and_predict_smoke_with_embargo(panel_small, small_cfg):
    """update.py 最終訓練沿用同一 split helper：能跑完且機率合法（不驗證 gap
    數值本身——gap 邏輯由共用 helper 的單元測試覆蓋，此處驗證接線無誤）。"""
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "update_mod", root / "scripts" / "update.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    results, max_date = mod.train_final_and_predict(
        panel_small, F.feature_columns(panel_small), small_cfg
    )
    assert "up_10" in results
    proba = results["up_10"]
    assert len(proba) == 3
    assert ((proba >= 0) & (proba <= 1)).all()
    assert max_date == panel_small.index.get_level_values("date").max()


# ---------------- (b) bootstrap CI 形狀 ----------------

def test_bootstrap_ci_shape_and_block_partition():
    oos = _fake_oos()
    out = block_bootstrap_ci(oos, n_boot=200, block_days=20, random_state=42)
    assert out["block_days"] == 20
    assert out["n_blocks"] == 3  # 60 個日期 / 20
    assert out["n_boot_valid"] > 0
    assert isinstance(out["gate_pass"], bool)
    assert isinstance(out["reasons"], list)
    for key in ["auc_minus_const05", "brier_minus_const05", "brier_minus_climatology",
                "brier_minus_always_up_rule", "brier_minus_momentum"]:
        ci = out[key]
        assert set(ci.keys()) == {"point", "lo", "hi"}, f"{key} 鍵名錯誤：{ci.keys()}"
        assert ci["lo"] is not None and ci["hi"] is not None
        assert ci["lo"] <= ci["hi"], f"{key} 下限應 <= 上限"
    # 區塊覆蓋全部日期且互不重疊
    ranges = out["block_date_ranges"]
    assert len(ranges) == 3
    assert ranges[0]["start"] == "2024-01-02"
    for prev, nxt in zip(ranges, ranges[1:]):
        assert pd.Timestamp(nxt["start"]) > pd.Timestamp(prev["end"])
    assert sum(r["n_dates"] for r in ranges) == 60


def test_bootstrap_block_days_below_20_raises():
    with pytest.raises(ValueError):
        block_bootstrap_ci(_fake_oos(), block_days=19)


def test_date_blocks_partition_each_ge_20_and_disjoint():
    dates = pd.bdate_range("2024-01-02", periods=105)
    blocks = _date_blocks(dates, 20)
    assert all(len(b) >= 20 for b in blocks), "每塊都應 >= 20 個交易日（含餘數併入）"
    flat = [d for b in blocks for d in b]
    assert len(flat) == len(dates) and len(set(flat)) == len(dates)
    assert set(flat) == set(dates)


# ---------------- (c) 四基準分離 ----------------

def test_four_baselines_separate_and_independent():
    oos = _fake_oos()
    b = separate_baselines(oos)
    assert b["n"] == 120
    for key in ["const05", "climatology", "always_up_rule", "momentum"]:
        assert key in b, f"缺少基準 {key}"
        assert b[key]["brier"] is not None, f"{key} 應有獨立 Brier"
    briers = {k: b[k]["brier"] for k in
              ["const05", "climatology", "always_up_rule", "momentum"]}
    # 構造值：const 0.25、clim 0.37、rule 0.30、mom 0.24，理應互不相同
    assert len(set(round(v, 6) for v in briers.values())) == 4, f"基準 Brier 互撞：{briers}"
    # 改動單一基準的輸入，只影響該基準，不覆蓋其他基準
    oos2 = oos.copy()
    oos2["proba_mom"] = 0.5
    b2 = separate_baselines(oos2)
    for key in ["const05", "climatology", "always_up_rule"]:
        assert b2[key] == b[key], f"改動動能基準不應影響 {key}"
    assert b2["momentum"]["brier"] != b["momentum"]["brier"]


# ---------------- (d) 氣候值無測試期洩漏 ----------------

def test_climatology_uses_training_rate_not_test_rate():
    # 訓練期基準率 0.3，測試期基準率 0.7（刻意不同）
    oos = _fake_oos(train_rates=(0.3,), pos_per_10=(7,))
    assert abs(oos["y_true"].mean() - 0.7) < 1e-9
    assert (oos["proba_clim"] == 0.3).all()
    clim_brier = separate_baselines(oos)["climatology"]["brier"]
    # 以訓練期值 0.3 計算：0.7*(1-0.3)^2 + 0.3*0.3^2 = 0.37
    assert abs(clim_brier - 0.37) < 1e-9, f"氣候 Brier 應用訓練期值算出 0.37，實為 {clim_brier}"
    # 若誤以測試期值 0.7 回填會得到 0.21，必須明顯不同
    assert abs(clim_brier - 0.21) > 0.05


def test_climatology_missing_refuses_backfill():
    oos = _fake_oos().drop(columns=["proba_clim"])
    b = separate_baselines(oos)
    assert b["climatology"]["brier"] is None
    assert "回填" in (b["climatology"].get("reason") or "")
    assert b["const05"]["brier"] is not None, "缺氣候值時其他基準仍應可算"
    boot = block_bootstrap_ci(oos, n_boot=200, random_state=42)
    assert boot["gate_pass"] is False
    assert "climatology" in boot["missing_baselines"]


# ---------------- (e) 固定 0.5 每折 AUC 恆為 0.5 ----------------

def test_const05_per_fold_auc_always_05():
    # 兩折訓練率不同（0.3 vs 0.6）、測試期基準率也不同（0.7 vs 0.3）：
    # 氣候基準混池 AUC 會偏離 0.5（算出 0.3），但每折 AUC 仍為 0.5——
    # 這正是「不得跨折混池」的原因；固定 0.5 則折內折外皆為 0.5。
    oos = _fake_oos(train_rates=(0.3, 0.6), pos_per_10=(7, 3))
    b = separate_baselines(oos)
    const_fold_auc = b["const05"]["auc_per_fold"]
    assert len(const_fold_auc) == 2
    assert all(v == 0.5 for v in const_fold_auc.values()), const_fold_auc
    assert b["const05"]["auc"] == 0.5
    assert abs(b["const05"]["brier"] - 0.25) < 1e-9
    clim_fold_auc = b["climatology"]["auc_per_fold"]
    assert all(v == 0.5 for v in clim_fold_auc.values()), clim_fold_auc
    assert b["climatology"]["auc"] < 0.5, "混池氣候 AUC 應偏離 0.5（本構造為 0.3）"
