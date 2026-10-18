"""網站產生器測試：JSON 產出完整性、欄位齊全、機率範圍。"""
import json

import numpy as np
import pandas as pd
import pytest

from src.sitegen import generate_site


def _fake_val_results(sids, dates):
    rng = np.random.default_rng(3)
    rows = []
    for sid in sids:
        for d in dates[::10]:
            rows.append({"stock_id": sid, "date": d,
                         "y_true": float(rng.integers(0, 2)),
                         "proba": float(rng.uniform(0.3, 0.7)),
                         "proba_always": 0.5, "proba_mom": 0.5})
    oos = pd.DataFrame(rows)
    metrics = {"n": len(oos), "base_rate": 0.5, "hit_rate": 0.55,
               "brier": 0.24, "auc": 0.56,
               "hit_rate_always": 0.5, "brier_always": 0.25, "auc_always": 0.5,
               "hit_rate_mom": 0.52, "brier_mom": 0.245, "auc_mom": 0.53}
    return {"targets": {t: {"oos": oos, "metrics": dict(metrics), "periods": []}
                        for t in ["up_10", "up_20", "exc_10", "exc_20"]}}


def _run_generate(panel_small, watchlist_small, base_cfg, tmp_path, model_valid=True):
    sids = sorted(panel_small.index.get_level_values("stock_id").unique())
    dates = panel_small.index.get_level_values("date").unique()
    proba_series = {
        "up_10": pd.Series({"1001": 0.62, "1002": 0.45, "1003": 0.58}),
        "up_20": pd.Series({"1001": 0.60, "1002": 0.48, "1003": 0.55}),
        "exc_10": pd.Series({"1001": 0.55, "1002": 0.50, "1003": 0.52}),
        "exc_20": pd.Series({"1001": 0.53, "1002": 0.49, "1003": 0.51}),
    }
    val_results = _fake_val_results(sids, dates)
    backtest_results = {"strategy": {"total_return": 0.05, "annualized": 0.08,
                                     "max_drawdown": -0.10, "n_trades": 40,
                                     "hit_rate": 0.55},
                        "buy_hold_index": {"total_return": 0.03},
                        "always_long": {"total_return": 0.02},
                        "credible": model_valid}
    meta = {"generated_at": "2024-06-01T18:30:00+08:00",
            "data_dates": {"price": "2024-05-31"},
            "sources": ["FinMind"],
            "universe": {"included": 3, "excluded": 0},
            "model_valid": model_valid,
            "risk_disclaimer": "測試用免責聲明"}
    out = tmp_path / "docs"
    generate_site(panel_small, proba_series, val_results, backtest_results,
                  meta, watchlist_small, base_cfg, out)
    return out


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def test_產生四種JSON且皆合法(panel_small, watchlist_small, base_cfg, tmp_path):
    out = _run_generate(panel_small, watchlist_small, base_cfg, tmp_path)
    for rel in ["data/meta.json", "data/predictions.json",
                "data/history_1001.json", "data/validation.json"]:
        p = out / rel
        assert p.exists(), f"{rel} 不存在"
        _load(p)  # JSON 解析失敗會直接報錯


def test_predictions_含必要欄位(panel_small, watchlist_small, base_cfg, tmp_path):
    out = _run_generate(panel_small, watchlist_small, base_cfg, tmp_path)
    preds = _load(out / "data/predictions.json")
    assert len(preds) == 3
    required = ["symbol", "p_up_10", "trend", "interval_10",
                "downside_risk_10", "invalidation"]
    for row in preds:
        for col in required:
            assert col in row, f"缺少欄位 {col}"
        assert row["trend"] in ("偏多", "盤整", "偏空")


def test_所有機率欄位在0到1之間或為None(panel_small, watchlist_small, base_cfg, tmp_path):
    out = _run_generate(panel_small, watchlist_small, base_cfg, tmp_path)
    preds = _load(out / "data/predictions.json")
    for row in preds:
        for col in ["p_up_10", "p_up_20", "p_exc_10", "p_exc_20", "hit_rate_10"]:
            v = row[col]
            assert v is None or 0.0 <= v <= 1.0, f"{row['symbol']}.{col}={v}"


def test_history檔含價格與預測紀錄(panel_small, watchlist_small, base_cfg, tmp_path):
    out = _run_generate(panel_small, watchlist_small, base_cfg, tmp_path)
    hist = _load(out / "data/history_1001.json")
    assert hist["symbol"] == "1001"
    assert len(hist["prices"]) > 0
    assert len(hist["prices"]) <= 260
    # 價格為 [日期, 收盤, 量] 三元組
    assert len(hist["prices"][0]) == 3


def test_每日存檔會寫入並建立索引(panel_small, watchlist_small, base_cfg, tmp_path):
    """generate_site 應在 data/archive/ 下產生日期存檔與 index.json。"""
    out = _run_generate(panel_small, watchlist_small, base_cfg, tmp_path)
    arch = out / "data" / "archive"
    assert (arch / "index.json").exists()
    index = _load(arch / "index.json")
    assert len(index) == 1 and index[0]["date"] == "2024-05-31"
    for name in ["predictions_2024-05-31.json", "meta_2024-05-31.json"]:
        p = arch / name
        assert p.exists(), f"{name} 不存在"
        _load(p)
    # 再跑一次同日期不應重複加入索引
    _run_generate(panel_small, watchlist_small, base_cfg, tmp_path)
    index2 = _load(arch / "index.json")
    assert len(index2) == 1


def test_缺機率時不報錯且趨勢退回盤整(panel_small, watchlist_small, base_cfg, tmp_path):
    """proba_series 缺某檔資料時，網站仍應產生（趨勢以 0.5 計為盤整）。"""
    sids = sorted(panel_small.index.get_level_values("stock_id").unique())
    dates = panel_small.index.get_level_values("date").unique()
    proba_series = {"up_10": pd.Series({"1001": 0.62})}  # 1002/1003 缺值
    val_results = _fake_val_results(sids, dates)
    meta = {"model_valid": True}
    out = tmp_path / "docs2"
    generate_site(panel_small, proba_series, val_results, {"credible": True},
                  meta, watchlist_small, base_cfg, out)
    preds = _load(out / "data/predictions.json")
    by_sid = {r["symbol"]: r for r in preds}
    assert by_sid["1002"]["p_up_10"] is None
    assert by_sid["1002"]["trend"] == "盤整"
