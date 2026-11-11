"""驗證報告輸出完整性測試（2026-11-11；項目 3）。

驗證：
- build_validation_report 在未來新產出的報告中保留四基準（const05 固定 0.5、
  climatology 訓練窗氣候基準率、always_up_rule 恆多規則、momentum 動能基準）、
  bootstrap_ci（CI、gate_pass、不可計算原因）與每目標頂層 gate_pass。
- 四基準彼此獨立、命名不混淆：climatology 的混池 AUC 不得被當成常數規則 AUC。
- JSON 可序列化：numpy 型別（例如 auc_per_fold 的折號鍵）與 NaN/Inf 不得
  讓 json.dumps 失敗；NaN/Inf 轉為 None（不得寫入非標準 JSON）。
- periods 的指標欄位收進巢狀 "metrics"（不再是空物件）；描述欄位留在上層。
- 舊假資料缺 baselines、bootstrap_ci 時寫 None（表示未計算），不得冒充。
- 既有統計門檻不動：本檔只測報告輸出，不測 credibility_check 的門檻
  （門檻測試見 test_validation.py）。
"""
import json

import numpy as np
import pytest

from src.sitegen import _jsonable, _period_out, build_validation_report

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
INDEX = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")


def _fake_baselines():
    """四基準各有獨立數值：climatology 的混池 AUC（0.4824）與 const05 的
    定義值 0.5 刻意不同，測得到底有沒有混淆兩者。"""
    return {
        "n": 1000,
        "const05": {"n": 1000, "base_rate": 0.52, "hit_rate": 0.52,
                    "brier": 0.25, "auc": 0.5,
                    "auc_per_fold": {np.int64(0): 0.5, np.int64(1): 0.5},
                    "source": "fixed 0.5 (rule constant, independent of data)"},
        "climatology": {"n": 1000, "base_rate": 0.52, "hit_rate": 0.52,
                        "brier": 0.2491, "auc": 0.4824,
                        "auc_per_fold": {np.int64(0): 0.5, np.int64(1): 0.5},
                        "source": "training-window base rate per fold (proba_clim)"},
        "always_up_rule": {"n": 1000, "base_rate": 0.52, "hit_rate": 0.52,
                           "brier": 0.48, "auc": 0.5,
                           "auc_per_fold": {np.int64(0): 0.5, np.int64(1): 0.5},
                           "source": "always predict up (probability 1.0)"},
        "momentum": {"n": 1000, "base_rate": 0.52, "hit_rate": 0.53,
                     "brier": 0.248, "auc": 0.506,
                     "auc_per_fold": {np.int64(0): 0.51, np.int64(1): 0.50},
                     "source": "momentum rule baseline (proba_mom)"},
    }


def _fake_bootstrap_ci(gate_pass=False):
    return {
        "block_days": 20, "n_blocks": 5, "block_date_ranges": [
            {"start": "2024-01-02", "end": "2024-01-30", "n_dates": 21}],
        "n_boot": 1000, "n_boot_valid": 950, "alpha": 0.05,
        "gate_pass": gate_pass,
        "reasons": ["CI(AUC-0.5) 下限未 > 0（lo=-0.01）。"] if not gate_pass
                   else ["通過 bootstrap 閘門。"],
        "baselines_evaluated": ["const05", "climatology", "always_up_rule",
                                "momentum"],
        "missing_baselines": [],
        "auc_minus_const05": {"point": 0.01, "lo": -0.01, "hi": 0.03},
        "brier_minus_const05": {"point": 0.001, "lo": -0.002, "hi": 0.004},
        "brier_minus_climatology": {"point": 0.002, "lo": -0.001, "hi": 0.005},
        "brier_minus_always_up_rule": {"point": -0.23, "lo": -0.24, "hi": -0.22},
        "brier_minus_momentum": {"point": 0.003, "lo": -0.001, "hi": 0.007},
    }


def _fake_period():
    return {"fold": 0, "train_start": "2024-01-02", "train_end": "2024-06-28",
            "fit_end": "2024-05-31", "cal_start": "2024-06-03",
            "fit_cal_embargo_days": 20, "train_base_rate": 0.52,
            "test_start": "2024-07-01", "test_end": "2024-08-30",
            "n": 1200, "base_rate": 0.51, "hit_rate": 0.55, "brier": 0.24,
            "auc": 0.56, "hit_rate_always": 0.5, "brier_always": 0.25,
            "auc_always": 0.4824, "hit_rate_mom": 0.52, "brier_mom": 0.245,
            "auc_mom": 0.53}


def _fake_val_results(gate_pass=False, with_extras=True):
    targets = {}
    for t in ["up_10", "up_20"]:
        d = {"metrics": {"n": 1000, "base_rate": 0.52, "hit_rate": 0.55,
                         "brier": 0.24, "auc": 0.56,
                         "hit_rate_always": 0.5, "brier_always": 0.25,
                         "auc_always": 0.4824,
                         "hit_rate_mom": 0.52, "brier_mom": 0.245,
                         "auc_mom": 0.53},
             "periods": [_fake_period()]}
        if with_extras:
            d["baselines"] = _fake_baselines()
            d["bootstrap_ci"] = _fake_bootstrap_ci(gate_pass)
        targets[t] = d
    return {"targets": targets}


def test_報告保留四基準且命名不混淆():
    report = build_validation_report(_fake_val_results(), {"credible": False},
                                     credible=False)
    text = json.dumps(report, ensure_ascii=False)  # 必須可序列化
    assert json.loads(text)["targets"]["up_10"]["baselines"]["const05"]["auc"] == 0.5
    b = report["targets"]["up_10"]["baselines"]
    assert set(b) == {"n", "const05", "climatology", "always_up_rule", "momentum"}
    # 關鍵：climatology 的混池 AUC（0.4824）與 const05 的定義值 0.5 必須不同，
    # 且各自獨立存在——不得把前者說成後者。
    assert b["climatology"]["auc"] == 0.4824
    assert b["const05"]["auc"] == 0.5
    assert b["always_up_rule"]["auc"] == 0.5
    assert b["always_up_rule"]["brier"] == 0.48
    assert b["momentum"]["auc"] == 0.506
    # 來源說明齊全，四者各不相同
    sources = {b[k]["source"] for k in
               ("const05", "climatology", "always_up_rule", "momentum")}
    assert len(sources) == 4


def test_報告保留bootstrap_ci與gate_pass():
    report = build_validation_report(_fake_val_results(gate_pass=False),
                                     {"credible": False}, credible=False)
    t = report["targets"]["up_10"]
    assert t["bootstrap_ci"]["gate_pass"] is False
    assert t["gate_pass"] is False  # 頂層 gate_pass 與 bootstrap_ci 一致
    assert t["bootstrap_ci"]["auc_minus_const05"]["lo"] == -0.01
    assert t["bootstrap_ci"]["reasons"] == ["CI(AUC-0.5) 下限未 > 0（lo=-0.01）。"]
    assert t["bootstrap_ci"]["baselines_evaluated"] == [
        "const05", "climatology", "always_up_rule", "momentum"]
    # 通過的情況也要原樣保留
    report2 = build_validation_report(_fake_val_results(gate_pass=True),
                                      {"credible": True}, credible=True)
    assert report2["targets"]["up_10"]["gate_pass"] is True
    assert report2["credible"] is True


def test_不可計算原因原樣保留():
    boot = _fake_bootstrap_ci(gate_pass=False)
    boot["gate_pass"] = False
    boot["reasons"] = ["有效重抽樣僅 12 次（少於 30 次），不給出 CI（樣本不足，不得硬湊）。"]
    boot["auc_minus_const05"] = {"point": 0.01, "lo": None, "hi": None}
    vr = _fake_val_results()
    vr["targets"]["up_10"]["bootstrap_ci"] = boot
    report = build_validation_report(vr, {}, credible=False)
    assert report["targets"]["up_10"]["bootstrap_ci"]["reasons"] == [
        "有效重抽樣僅 12 次（少於 30 次），不給出 CI（樣本不足，不得硬湊）。"]
    assert report["targets"]["up_10"]["bootstrap_ci"]["auc_minus_const05"]["lo"] is None


def test_numpy鍵與NaN不得破壞JSON序列化():
    vr = _fake_val_results()
    # auc_per_fold 的折號鍵是 numpy int64；brier 塞一個 NaN
    vr["targets"]["up_10"]["baselines"]["momentum"]["brier"] = float("nan")
    vr["targets"]["up_10"]["baselines"]["momentum"]["auc_per_fold"] = {
        np.int64(0): np.float64(0.51), np.int64(1): np.float64(0.5)}
    report = build_validation_report(vr, {}, credible=False)
    text = json.dumps(report, ensure_ascii=False)  # 不得拋 TypeError
    loaded = json.loads(text)
    assert loaded["targets"]["up_10"]["baselines"]["momentum"]["brier"] is None
    # numpy int64 鍵轉為 int（json round-trip 後為字串鍵 "0"/"1" 是 JSON 本身的性質）
    assert set(_jsonable(vr["targets"]["up_10"]["baselines"]["momentum"]
                         ["auc_per_fold"]).keys()) == {0, 1}


def test_period指標收進巢狀metrics不再為空():
    p = _period_out(_fake_period())
    assert p["fold"] == 0
    assert p["test_start"] == "2024-07-01"
    assert p["train_base_rate"] == 0.52  # 描述欄位留在上層
    m = p["metrics"]
    assert m["n"] == 1200 and m["auc"] == 0.56 and m["brier"] == 0.24
    # 註：sitegen 的 _r 一律取 3 位小數（既有顯示精度，本次不改），故 0.4824 → 0.482
    assert m["auc_always"] == 0.482 and m["auc_mom"] == 0.53
    assert "metrics" not in p or p["metrics"]  # 不得為空物件
    # 上游已給巢狀 metrics 時直接沿用
    p2 = _period_out({"fold": 1, "test_start": "2024-09-01",
                      "metrics": {"auc": 0.5, "brier": 0.25}})
    assert p2["metrics"] == {"auc": 0.5, "brier": 0.25}
    assert p2["fold"] == 1


def test_舊假資料缺基準與CI時寫None不冒充():
    report = build_validation_report(_fake_val_results(with_extras=False),
                                     {"credible": False}, credible=False)
    t = report["targets"]["up_10"]
    assert t["baselines"] is None
    assert t["bootstrap_ci"] is None
    assert t["gate_pass"] is None  # 未計算，不得當作通過
    assert t["metrics"]["auc"] == 0.56  # 既有 metrics 不動
    assert t["periods"][0]["metrics"]["auc"] == 0.56
    json.dumps(report, ensure_ascii=False)  # 仍須可序列化


def test_前端基準命名已釐清():
    # 舊的錯誤標籤必須消失：auc_always 是訓練窗氣候基準率，不是恆多規則
    assert "全猜上漲基準" not in INDEX
    assert "訓練窗氣候基準率" in INDEX
    assert "固定 0.5" in INDEX
    assert "恆多規則" in INDEX
    assert "動量基準" in INDEX
    # 新報告的 baselines 與閘門顯示已接線
    assert "baselines" in INDEX
    assert "Bootstrap 閘門" in INDEX
