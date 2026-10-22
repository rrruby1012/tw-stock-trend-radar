"""sitegen provenance 測試：三類資料分流欄位、history 標記、存檔索引雜湊。"""
import json

from src.ledger import prediction_sha256
from src.sitegen import generate_site
from tests.test_sitegen import _fake_val_results


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _run(panel_small, watchlist_small, base_cfg, tmp_path):
    sids = sorted(panel_small.index.get_level_values("stock_id").unique())
    dates = panel_small.index.get_level_values("date").unique()
    import pandas as pd
    proba_series = {
        "up_10": pd.Series({"1001": 0.62, "1002": 0.45, "1003": 0.58}),
        "up_20": pd.Series({"1001": 0.60, "1002": 0.48, "1003": 0.55}),
        "exc_10": pd.Series({"1001": 0.55, "1002": 0.50, "1003": 0.52}),
        "exc_20": pd.Series({"1001": 0.53, "1002": 0.49, "1003": 0.51}),
    }
    meta = {"generated_at": "2024-06-01T18:30:00+08:00",
            "data_dates": {"price": "2024-05-31"},
            "sources": ["FinMind"],
            "universe": {"included": 3, "excluded": 0},
            "model_valid": False,
            "risk_disclaimer": "測試用免責聲明"}
    out = tmp_path / "docs"
    generate_site(panel_small, proba_series, _fake_val_results(sids, dates),
                  {"credible": False}, meta, watchlist_small, base_cfg, out,
                  provenance={"model_version": "mv-test",
                              "config_version": "cv-test",
                              "source_commit_sha": "deadbeef",
                              "workflow_run_id": "4242"})
    return out


def test_meta_provenance欄位與雜湊(panel_small, watchlist_small, base_cfg, tmp_path):
    out = _run(panel_small, watchlist_small, base_cfg, tmp_path)
    meta = _load(out / "data/meta.json")
    preds = _load(out / "data/predictions.json")
    prov = meta["provenance"]
    assert meta["data_class"] == "prospective"
    assert prov["data_class"] == "prospective"
    assert prov["model_version"] == "mv-test"
    assert prov["config_version"] == "cv-test"
    assert prov["source_commit_sha"] == "deadbeef"
    assert prov["workflow_run_id"] == "4242"
    # 雜湊必須能由 predictions 內容重算驗證
    assert prov["prediction_sha256"] == prediction_sha256(preds)
    assert prov["ledger_status"] == "pending"


def test_history標記為walkforward_sim(panel_small, watchlist_small, base_cfg, tmp_path):
    out = _run(panel_small, watchlist_small, base_cfg, tmp_path)
    hist = _load(out / "data/history_1001.json")
    assert hist["predictions_class"] == "walkforward_sim"
    assert "非實際發布" in hist["predictions_note"]


def test_存檔與索引帶class與雜湊(panel_small, watchlist_small, base_cfg, tmp_path):
    out = _run(panel_small, watchlist_small, base_cfg, tmp_path)
    arch_meta = _load(out / "data/archive/meta_2024-05-31.json")
    assert arch_meta["data_class"] == "prospective"
    assert arch_meta["provenance"]["prediction_sha256"]
    idx = _load(out / "data/archive/index.json")
    entry = [e for e in idx if e["date"] == "2024-05-31"][0]
    assert entry["data_class"] == "prospective"
    assert entry["prediction_sha256"] == arch_meta["provenance"]["prediction_sha256"]
