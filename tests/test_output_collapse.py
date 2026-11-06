"""模型輸出塌縮（2026-11-05）回歸測試。

涵蓋使用者驗收 D1~D6：
1. 50 檔同機率必定觸發退化。
2. 校準前有差、校準後全同 → 原因為校準飽和（calibration_saturation）。
3. 校準前全同 → 原因為上游塌縮（upstream_collapse）。
4. 退化時網站不得出現偏多或偏空（sitegen 趨勢寫「未提供」，
   前端實驗區顯示「模型輸出失去辨識度」）。
5. 健康輸出維持原行為（未退化時趨勢與機率照常）。
6. 既有 ledger 不改寫；QuotaExceeded re-raise、adjustment failure
   排除、PIT 與 embargo 行為由既有測試覆蓋，本檔只驗證 ledger
   新欄位不破壞 write-once 語意。
另含根因修正測試：三大法人淨額大小寫錯誤、train_final 須依設定
選校準器（舊程式寫死 isotonic，設定改 sigmoid 也無效）。
2026-11-06 加固：Platt（sigmoid）斜率無約束，校準窗分數—標籤關係
為負時斜率為負、排序反轉（28 次 walk-forward 擬合中有 5 次；up_10
第 7 折斜率 −0.36、逐折 AUC 0.4016）。健康檢查原本只記錄 Spearman
而不判定反轉，本節補上方向反轉旗標與原因分類。
"""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import features as F
from src.datasources.finmind import FinMindClient
from src.ledger import LedgerConflict, build_publication_record, publish
from src.output_health import check_prediction_health
from src.sitegen import generate_site

ROOT = Path(__file__).resolve().parent.parent


# ---------- 1. 50 檔同機率必定觸發退化 ----------

def test_五十檔同機率必定觸發退化():
    """公開站 11/05 現象：p_up_10 全為 0.609、p_up_20 全為 0.693。"""
    rng = np.random.default_rng(0)
    pre10 = rng.uniform(0.44, 0.78, 50)  # 校準前仍有 50 種值
    pre20 = rng.uniform(0.48, 0.93, 50)
    health = check_prediction_health({
        "up_10": {"pre": pre10, "post": np.full(50, 0.609),
                  "calibrator": None, "calibrated": True},
        "up_20": {"pre": pre20, "post": np.full(50, 0.693),
                  "calibrator": None, "calibrated": True},
    })
    assert health["degenerate"] is True
    assert health["cause"] == "calibration_saturation"
    assert health["targets"]["up_10"]["flags"]["all_equal"] is True
    assert health["targets"]["up_20"]["flags"]["all_equal"] is True
    assert health["targets"]["up_10"]["post"]["nunique"] == 1


def test_近零變異與相異值過少也會觸發退化():
    """11/03（47×0.609＋3×0.592）與 11/04（48×0.609＋2×0.608）
    只有 2 種值、標準差 < 0.005，必須觸發，不得等到全同才發現。"""
    post_1103 = np.array([0.609] * 47 + [0.592] * 3)
    post_1104 = np.array([0.609] * 48 + [0.608] * 2)
    for post in (post_1103, post_1104):
        health = check_prediction_health({
            "up_10": {"pre": np.linspace(0.44, 0.72, 50), "post": post,
                      "calibrator": None, "calibrated": True},
            "up_20": {"pre": np.linspace(0.48, 0.85, 50),
                      "post": np.linspace(0.60, 0.70, 50),
                      "calibrator": None, "calibrated": True},
        })
        assert health["degenerate"] is True
        assert health["targets"]["up_10"]["flags"]["low_distinct"] is True


# ---------- 2/3. 原因分類：校準飽和 vs 上游塌縮 ----------

def test_校準前有差校準後全同指出校準飽和():
    pre = np.linspace(0.40, 0.80, 50)  # 原始分數明顯有別
    health = check_prediction_health({
        "up_10": {"pre": pre, "post": np.full(50, 0.60),
                  "calibrator": None, "calibrated": True},
        "up_20": {"pre": pre, "post": np.full(50, 0.65),
                  "calibrator": None, "calibrated": True},
    })
    assert health["degenerate"] is True
    assert health["cause"] == "calibration_saturation"


def test_校準前全同指出上游塌縮():
    health = check_prediction_health({
        "up_10": {"pre": np.full(50, 0.55), "post": np.full(50, 0.60),
                  "calibrator": None, "calibrated": True},
        "up_20": {"pre": np.full(50, 0.55), "post": np.full(50, 0.65),
                  "calibrator": None, "calibrated": True},
    })
    assert health["degenerate"] is True
    assert health["cause"] == "upstream_collapse"


def test_校準後失去原始排序會觸發退化():
    """isotonic 階梯平台造成大量平手：50 檔校準前各異、校準後只剩
    3 種值時，成對平手比例 > 50%，Spearman 也會崩解。"""
    rng = np.random.default_rng(1)
    pre = np.sort(rng.uniform(0.40, 0.80, 50))
    # 主平台 40 檔同值：成對平手比例 = (780+10+10)/1225 ≈ 0.65 > 0.5。
    post = np.array([0.57] * 40 + [0.60] * 5 + [0.65] * 5)
    health = check_prediction_health({
        "up_10": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
        "up_20": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
    })
    assert health["targets"]["up_10"]["flags"]["rank_loss"] is True
    assert health["degenerate"] is True


def test_最新分數超出校準範圍會被標示():
    """isotonic out_of_bounds="clip" 會把範圍外分數夾到邊緣平台；
    全部超出時輸出恆為常數，必須能指出來。"""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.array([0.40, 0.50, 0.60]), np.array([0.0, 1.0, 1.0]))
    health = check_prediction_health({
        "up_10": {"pre": np.array([0.70, 0.75, 0.80] * 17)[:50],
                  "post": np.full(50, 1.0), "calibrator": iso, "calibrated": True},
        "up_20": {"pre": np.linspace(0.40, 0.60, 50),
                  "post": np.linspace(0.50, 0.70, 50),
                  "calibrator": iso, "calibrated": True},
    })
    assert health["targets"]["up_10"]["flags"]["out_of_range_all"] is True
    assert health["degenerate"] is True


# ---------- 5. 健康輸出維持原行為 ----------

def test_健康輸出不觸發退化():
    rng = np.random.default_rng(2)
    pre = np.sort(rng.uniform(0.35, 0.75, 50))
    post = np.sort(rng.uniform(0.40, 0.70, 50))
    health = check_prediction_health({
        "up_10": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
        "up_20": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
    })
    assert health["degenerate"] is False
    assert health["cause"] is None
    assert health["reasons"] == []


def test_最新特徵列重複會判定為特徵塌縮():
    """50 檔最新特徵列完全相同時，原因須指向上游特徵而非校準器。"""
    row = pd.DataFrame({"ret_20": [0.01], "rsi_14": [55.0]})
    X = pd.concat([row] * 50, ignore_index=True)
    health = check_prediction_health(
        {"up_10": {"pre": np.full(50, 0.55), "post": np.full(50, 0.60),
                   "calibrator": None, "calibrated": True},
         "up_20": {"pre": np.full(50, 0.55), "post": np.full(50, 0.65),
                   "calibrator": None, "calibrated": True}},
        latest_features=X)
    assert health["degenerate"] is True
    assert health["cause"] == "feature_collapse"


# ---------- 4. 退化時網站不得出現偏多或偏空 ----------

def _sitegen_preds(panel_small, watchlist_small, base_cfg, tmp_path, degenerate):
    """以 panel_small 跑 generate_site，回傳 predictions.json 內容。"""
    sids = sorted(panel_small.index.get_level_values("stock_id").unique())
    dates = panel_small.index.get_level_values("date").unique()
    proba_series = {
        "up_10": pd.Series({"1001": 0.609, "1002": 0.609, "1003": 0.609}),
        "up_20": pd.Series({"1001": 0.693, "1002": 0.693, "1003": 0.693}),
        "exc_10": pd.Series({"1001": 0.55, "1002": 0.55, "1003": 0.55}),
        "exc_20": pd.Series({"1001": 0.53, "1002": 0.53, "1003": 0.53}),
    }
    rows = [{"stock_id": sid, "date": d, "y_true": 1.0, "proba": 0.6,
             "proba_always": 0.5, "proba_mom": 0.5}
            for sid in sids for d in dates[::10]]
    oos = pd.DataFrame(rows)
    metrics = {"n": len(oos), "base_rate": 0.5, "hit_rate": 0.55,
               "brier": 0.24, "auc": 0.56, "hit_rate_always": 0.5,
               "brier_always": 0.25, "auc_always": 0.5,
               "hit_rate_mom": 0.52, "brier_mom": 0.245, "auc_mom": 0.53}
    val_results = {"targets": {t: {"oos": oos, "metrics": dict(metrics), "periods": []}
                               for t in ["up_10", "up_20", "exc_10", "exc_20"]}}
    meta = {"generated_at": "2026-11-05T19:30:00+08:00",
            "data_dates": {"price": "2026-11-05"},
            "sources": ["FinMind"], "universe": {"included": 3, "excluded": 0},
            "model_valid": False, "risk_disclaimer": "測試用免責聲明",
            "model_output_degenerate": degenerate}
    out = tmp_path / ("docs_deg" if degenerate else "docs_ok")
    generate_site(panel_small, proba_series, val_results, {"credible": False},
                  meta, watchlist_small, base_cfg, out)
    return json.loads((out / "data" / "predictions.json").read_text(encoding="utf-8"))


def test_退化時網站趨勢一律未提供(panel_small, watchlist_small, base_cfg, tmp_path):
    preds = _sitegen_preds(panel_small, watchlist_small, base_cfg, tmp_path, True)
    assert len(preds) == 3
    for row in preds:
        assert row["trend"] == "未提供", f"{row['symbol']} 退化時仍有方向：{row['trend']}"
        assert row["trend"] not in ("偏多", "偏空")
    # 機率欄位仍原樣保留供查核（止血是隱藏方向，不是竄改數字）。
    assert all(row["p_up_10"] == 0.609 for row in preds)


def test_健康時網站趨勢照常(panel_small, watchlist_small, base_cfg, tmp_path):
    preds = _sitegen_preds(panel_small, watchlist_small, base_cfg, tmp_path, False)
    assert {row["trend"] for row in preds} == {"偏多"}  # 0.609/0.693 皆高於門檻


def test_前端退化分支顯示失去辨識度且隱藏趨勢():
    """index.html / stock.html 須有退化分支：顯示「模型輸出失去辨識度」，
    且趨勢徽章只在非退化時渲染（以 model_output_degenerate 守門）。"""
    index_html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    stock_html = (ROOT / "docs" / "stock.html").read_text(encoding="utf-8")
    for name, text in [("index.html", index_html), ("stock.html", stock_html)]:
        assert "模型輸出失去辨識度" in text, f"{name} 缺少退化警示文案"
        assert "model_output_degenerate" in text, f"{name} 未讀取退化旗標"
    # 首頁實驗區：趨勢欄只在非退化時加入（退化時表格只剩代號與收盤價）。
    assert "if (!meta.model_output_degenerate)" in index_html
    # 個股頁實驗區：退化時不渲染趨勢徽章。
    assert "window.__modelDegenerate" in stock_html


# ---------- 6. ledger：新欄位不破壞 write-once ----------

def _preds_for_ledger():
    return [{"symbol": "2330", "name": "台積電", "close": 2405.0,
             "p_up_10": 0.609, "p_up_20": 0.693, "trend": "未提供"}]


def test_退化紀錄凍結時攜帶退化旗標且既有紀錄不被改寫(tmp_path):
    ledger_dir = tmp_path / "ledger"
    rec = build_publication_record(
        predictions=_preds_for_ledger(), data_cutoff_date="2026-11-05",
        generated_at="2026-11-05T19:30:00+08:00",
        published_at="2026-11-05T19:33:00+08:00",
        source_commit_sha="abc123", workflow_run_id="1",
        model_version="logreg-sigmoid-v3", config_version="2026-11-05.1",
        model_output_degenerate=True, model_output_cause="calibration_saturation")
    frozen = publish(ledger_dir, rec)
    assert frozen["model_output_degenerate"] is True
    assert frozen["model_output_cause"] == "calibration_saturation"
    stored = json.loads((ledger_dir / "published" / "2026-11-05.json").read_text(encoding="utf-8"))
    assert stored["model_output_degenerate"] is True
    # 同日同內容冪等；同日不同內容仍衝突（write-once 不變）。
    assert publish(ledger_dir, rec)["prediction_sha256"] == frozen["prediction_sha256"]
    tampered = build_publication_record(
        predictions=[{**_preds_for_ledger()[0], "p_up_10": 0.5}],
        data_cutoff_date="2026-11-05",
        generated_at="2026-11-05T19:30:00+08:00",
        published_at="2026-11-05T19:33:00+08:00",
        source_commit_sha="abc123", workflow_run_id="1",
        model_version="logreg-sigmoid-v3", config_version="2026-11-05.1")
    with pytest.raises(LedgerConflict):
        publish(ledger_dir, tampered)


def test_未標退化的舊版呼叫仍可發布():
    """build_publication_record 新參數皆有預設值，舊呼叫不受影響。"""
    rec = build_publication_record(
        predictions=_preds_for_ledger(), data_cutoff_date="2026-11-05",
        generated_at="2026-11-05T19:30:00+08:00",
        published_at="2026-11-05T19:33:00+08:00",
        source_commit_sha="abc123", workflow_run_id="1",
        model_version="logreg-isotonic-v2", config_version="2026-10-29.1")
    assert rec["model_output_degenerate"] is False
    assert rec["model_output_cause"] is None


# ---------- 根因修正：三大法人淨額大小寫 ----------

def test_三大法人淨額小寫欄位可正確計算(monkeypatch):
    """FinMind 回傳 Foreign_Investor_buy（小寫 buy）；舊程式以大寫
    "Buy" 比對，淨額永遠為 0，7 個籌碼特徵死亡。修正後須等於買減賣。"""
    client = FinMindClient(token="", timeout_sec=30)
    fake = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-03", "2026-11-04"]),
        "stock_id": ["2330", "2330"],
        "Foreign_Investor_buy": [100, 200], "Foreign_Investor_sell": [40, 50],
        "Foreign_Dealer_Self_buy": [10, 0], "Foreign_Dealer_Self_sell": [5, 0],
        "Investment_Trust_buy": [30, 0], "Investment_Trust_sell": [10, 20],
        "Dealer_buy": [0, 0], "Dealer_sell": [0, 0],
        "Dealer_self_buy": [7, 0], "Dealer_self_sell": [2, 0],
        "Dealer_Hedging_buy": [3, 0], "Dealer_Hedging_sell": [1, 0],
    })
    monkeypatch.setattr(client, "fetch", lambda *a, **k: fake)
    out = client.institutional_wide("2330", "2026-11-03", "2026-11-04")
    # 外資淨額 = (100-40)；投信 = (30-10)；自營商 = 全部 Dealer 系買賣合計
    assert out["foreign_net"].tolist() == [60, 150]
    assert out["trust_net"].tolist() == [20, -20]
    # 自營商：沿用舊定義（欄名含 Dealer 者皆計入，含 Foreign_Dealer_Self），
    # 本次只修正大小寫比對、不改定義：(10-5)+(0-0)+(7-2)+(3-1)=12；
    # 第二日各欄皆為 0，故為 0。
    assert out["dealer_net"].tolist() == [12, 0]


# ---------- 根因修正：train_final 須依設定選校準器 ----------

def _load_update():
    spec = importlib.util.spec_from_file_location(
        "update_mod_collapse", ROOT / "scripts" / "update.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_train_final依設定選校準器且回傳健康檢查(panel_small, small_cfg):
    """舊程式寫死 CalibratedModel(est)（預設 isotonic），設定改 sigmoid
    也無效。修正後：sigmoid 的校準後機率須保留全部排序（nunique ==
    校準前 nunique），isotonic 則會塌縮；且回傳須含 health dict。"""
    mod = _load_update()
    fc = F.feature_columns(panel_small)
    cfg_sig = copy.deepcopy(small_cfg)
    cfg_sig["model"]["calibration"]["method"] = "sigmoid"
    cfg_iso = copy.deepcopy(small_cfg)
    cfg_iso["model"]["calibration"]["method"] = "isotonic"
    res_sig, _, health_sig = mod.train_final_and_predict(panel_small, fc, cfg_sig)
    res_iso, _, health_iso = mod.train_final_and_predict(panel_small, fc, cfg_iso)
    assert isinstance(health_sig, dict) and "degenerate" in health_sig
    assert isinstance(health_iso, dict) and "degenerate" in health_iso
    # sigmoid 為嚴格遞增映射：校準後相異值數 = 校準前相異值數。
    pre_nunique = health_sig["targets"]["up_10"]["pre"]["nunique"]
    assert health_sig["targets"]["up_10"]["post"]["nunique"] == pre_nunique
    # isotonic 為階梯函數：校準後相異值數 <= 校準前（本合成資料上嚴格小於）。
    assert health_iso["targets"]["up_10"]["post"]["nunique"] <= pre_nunique
    assert res_sig["up_10"].nunique() >= res_iso["up_10"].nunique()


# ---------- 方向反轉（2026-11-06 加固）：負斜率 Platt 校準 ----------

def test_方向反轉必定觸發並分類為方向反轉():
    """raw 遞增、post 遞減：Spearman = −1、有效樣本 50，必須觸發，
    原因為 calibration_direction_reversal（不是校準飽和）。"""
    pre = np.linspace(0.44, 0.78, 50)
    post = np.linspace(0.65, 0.35, 50)  # 嚴格遞減
    health = check_prediction_health({
        "up_10": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
        "up_20": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
    })
    assert health["degenerate"] is True
    assert health["cause"] == "calibration_direction_reversal"
    assert health["targets"]["up_10"]["flags"]["direction_reversal"] is True
    assert health["targets"]["up_10"]["spearman_pre_post"] == pytest.approx(-1.0)
    # 相異值與變異皆健康：不是塌縮，是方向反了。
    assert health["targets"]["up_10"]["post"]["nunique"] == 50
    assert health["targets"]["up_10"]["flags"]["low_distinct"] is False
    assert health["targets"]["up_10"]["flags"]["all_equal"] is False


def test_負斜率Platt校準真實觸發方向反轉():
    """以真實 CalibratedModel 重現稽核發現：校準窗分數—標籤關係為負
    時，Platt 斜率為負，校準後排序反轉，健康檢查必須攔住。
    （walk-forward 實測：up_10 第 7 折斜率 −0.36、逐折 AUC 0.4016。）
    本測試只做偵測，不做任何修飾：不取絕對值、不翻轉符號、不以
    未校準分數當機率，退化時只走既有止血路徑。"""
    from src.models import CalibratedModel, LogisticModel
    rng = np.random.default_rng(0)
    X_fit = rng.normal(0, 1, (200, 1))
    y_fit = (X_fit[:, 0] > 0).astype(int)
    X_cal = rng.normal(0, 1, (300, 1))
    y_cal = (X_cal[:, 0] < 0).astype(int)  # 校準窗標籤與分數反向
    # 用專案實際的 LogisticModel（predict_proba 回傳一維機率），
    # 而非 sklearn 原生 LogisticRegression（回傳二維，CalibratedModel
    # 的校準路徑不支援）。
    model = CalibratedModel(LogisticModel(), method="sigmoid")
    model.fit(X_fit, y_fit, X_cal, y_cal)
    assert model.calibrated_ is True
    assert model.calibrator_.coef_[0][0] < 0  # 負斜率是本測試的前提
    X_latest = np.linspace(-2, 2, 50).reshape(-1, 1)
    # LogisticModel.predict_proba 已回傳一維機率（不需再取 [:, 1]）。
    pre = model.estimator.predict_proba(X_latest)
    post = model.predict_proba(X_latest)
    assert np.all(np.diff(pre) > 0)  # 原始分數遞增
    assert np.all(np.diff(post) < 0)  # 校準後遞減
    health = check_prediction_health({
        "up_10": {"pre": pre, "post": post,
                  "calibrator": model.calibrator_, "calibrated": True},
        "up_20": {"pre": pre, "post": post,
                  "calibrator": model.calibrator_, "calibrated": True},
    })
    assert health["degenerate"] is True
    assert health["cause"] == "calibration_direction_reversal"
    assert health["targets"]["up_10"]["spearman_pre_post"] < 0


def test_正斜率sigmoid維持健康():
    """11/05 最終模型實測：四個目標 Spearman 皆為 1.0。正斜率時
    sigmoid 為嚴格遞增映射，排序完整保留，不得觸發。"""
    pre = np.linspace(0.44, 0.78, 50)
    post = 1 / (1 + np.exp(-(pre - 0.6) * 8))  # 嚴格遞增的 sigmoid 映射
    health = check_prediction_health({
        "up_10": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
        "up_20": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
    })
    assert health["degenerate"] is False
    assert health["cause"] is None
    assert health["targets"]["up_10"]["spearman_pre_post"] == pytest.approx(1.0)
    assert health["targets"]["up_10"]["flags"]["direction_reversal"] is False


def test_isotonic平台仍分類校準飽和且不誤判方向反轉():
    """isotonic 單調非遞減，Spearman 永遠 >= 0；平台塌縮的原因
    必須仍是 calibration_saturation，不得被新規則搶走。"""
    pre = np.linspace(0.40, 0.80, 50)
    post = np.array([0.57] * 40 + [0.60] * 5 + [0.65] * 5)
    health = check_prediction_health({
        "up_10": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
        "up_20": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
    })
    assert health["degenerate"] is True
    assert health["cause"] == "calibration_saturation"
    assert health["targets"]["up_10"]["flags"]["direction_reversal"] is False
    assert health["targets"]["up_10"]["spearman_pre_post"] >= 0


def test_方向反轉樣本不足時不判定():
    """有效配對樣本 < 10 時，等級相關不穩定，不得以方向反轉觸發
    （MIN_DIRECTION_SAMPLES = 10 的守門作用）。"""
    pre = np.linspace(0.44, 0.78, 5)
    post = np.linspace(0.65, 0.35, 5)
    health = check_prediction_health({
        "up_10": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
        "up_20": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
    })
    assert health["targets"]["up_10"]["flags"]["direction_reversal"] is False
    assert health["degenerate"] is False


def test_方向反轉時網站與前端維持隱藏方向(panel_small, watchlist_small, base_cfg, tmp_path):
    """方向反轉的健康檢查結果餵給 sitegen：趨勢一律「未提供」，
    機率原樣保留；前端兩頁的退化分支與其他退化原因相同（只認
    model_output_degenerate 旗標，不認原因字串），故方向同樣隱藏。"""
    pre = np.linspace(0.44, 0.78, 50)
    post = np.linspace(0.65, 0.35, 50)
    health = check_prediction_health({
        "up_10": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
        "up_20": {"pre": pre, "post": post, "calibrator": None, "calibrated": True},
    })
    assert health["cause"] == "calibration_direction_reversal"
    preds = _sitegen_preds(panel_small, watchlist_small, base_cfg, tmp_path,
                           health["degenerate"])
    assert len(preds) == 3
    for row in preds:
        assert row["trend"] == "未提供", f"{row['symbol']} 方向反轉時仍有方向：{row['trend']}"
        assert row["trend"] not in ("偏多", "偏空")
    assert all(row["p_up_10"] == 0.609 for row in preds)
    index_html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    stock_html = (ROOT / "docs" / "stock.html").read_text(encoding="utf-8")
    for name, text in [("index.html", index_html), ("stock.html", stock_html)]:
        assert "模型輸出失去辨識度" in text, f"{name} 缺少退化警示文案"
        assert "model_output_degenerate" in text, f"{name} 未讀取退化旗標"
    assert "if (!meta.model_output_degenerate)" in index_html
    assert "window.__modelDegenerate" in stock_html
