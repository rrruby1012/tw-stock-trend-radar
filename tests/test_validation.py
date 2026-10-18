"""驗證層測試：walk-forward 的時間正確性、切分不重疊、可信度門檻方向。"""
import copy

import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.validation import credibility_check, walk_forward


def _run_wf(panel_small, small_cfg):
    feature_cols = F.feature_columns(panel_small)
    return walk_forward(panel_small, feature_cols, small_cfg)


def test_walk_forward_訓練集早於測試集且有embargo隔離(panel_small, small_cfg):
    """每個期間：訓練集日期全部 < 測試開始 - embargo；測試窗格不重疊；oos 有樣本。"""
    results = _run_wf(panel_small, small_cfg)
    embargo = small_cfg["model"]["validation"]["embargo_days"]
    t10 = results["targets"]["up_10"]
    assert t10["metrics"]["n"] > 0, "樣本外樣本數應 > 0"

    periods = t10["periods"]
    assert len(periods) >= 2
    prev_test_end = None
    for p in periods:
        train_end = pd.Timestamp(p["train_end"])
        test_start = pd.Timestamp(p["test_start"])
        test_end = pd.Timestamp(p["test_end"])
        # 交易日曆以 bdate 計：兩者間隔的交易日數 >= embargo
        gap = len(pd.bdate_range(train_end + pd.Timedelta(days=1), test_start)) - 1
        assert gap >= embargo, f"embargo 不足：{p['train_end']} ~ {p['test_start']}"
        assert test_start > train_end
        if prev_test_end is not None:
            assert test_start > prev_test_end, "測試窗格不得重疊"
        prev_test_end = test_end


def test_walk_forward_oos_預測機率合法(panel_small, small_cfg):
    results = _run_wf(panel_small, small_cfg)
    oos = results["targets"]["up_10"]["oos"]
    for col in ["proba", "proba_always", "proba_mom"]:
        assert ((oos[col] >= 0) & (oos[col] <= 1)).all(), f"{col} 必須在 [0,1]"
    assert set(oos["y_true"].unique()) <= {0.0, 1.0}


def test_walk_forward_特徵欄位誤含標籤會被排除(panel_small, small_cfg):
    """呼叫方若誤把標籤當特徵傳入，walk_forward 必須自動排除（防洩漏）。"""
    feature_cols = F.feature_columns(panel_small) + ["up_10", "fwd_ret_10"]
    results = walk_forward(panel_small, feature_cols, small_cfg)
    assert results["targets"]["up_10"]["metrics"]["n"] > 0


def _fake_results(n, auc, auc_base, brier, brier_always):
    return {"targets": {"up_10": {"metrics": {
        "n": n, "auc": auc, "auc_always": auc_base,
        "brier": brier, "brier_always": brier_always,
    }}}}


def test_credibility_check_隨機水準回傳False並附原因(small_cfg):
    """模型與基準無異時，必須判為不可信並說明原因。"""
    results = _fake_results(n=10000, auc=0.501, auc_base=0.500,
                            brier=0.2500, brier_always=0.2499)
    ok, reasons = credibility_check(results, small_cfg)
    assert ok is False
    assert len(reasons) > 0
    assert any("AUC" in r for r in reasons)


def test_credibility_check_明顯優於基準時回傳True(small_cfg):
    """此測試僅驗證門檻邏輯方向正確（非宣稱作弊特徵可用）。"""
    results = _fake_results(n=10000, auc=0.75, auc_base=0.50,
                            brier=0.18, brier_always=0.25)
    ok, reasons = credibility_check(results, small_cfg)
    assert ok is True
    assert len(reasons) > 0


def test_credibility_check_樣本不足回傳False(small_cfg):
    results = _fake_results(n=50, auc=0.80, auc_base=0.50,
                            brier=0.15, brier_always=0.25)
    ok, reasons = credibility_check(results, small_cfg)
    assert ok is False
    assert any("樣本" in r for r in reasons)


def test_credibility_check_缺up_10結果回傳False(small_cfg):
    ok, reasons = credibility_check({"targets": {}}, small_cfg)
    assert ok is False
