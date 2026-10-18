"""時間序列 walk-forward 驗證：絕不隨機切割資料，嚴防 look-ahead。

流程（每個預測目標獨立進行）：
1. 依日期排序，每 refit_every_days 個交易日開一個測試窗格。
2. 訓練集 = 測試窗格開始前 embargo_days 個交易日之前的所有資料。
3. 訓練集內再按日期切 80%（模型擬合）/ 20%（較晚日期，機率校準）。
4. 在測試窗格上產生樣本外（out-of-sample）預測，彙總計算指標。

所有填補、標準化、校準的統計量都只從訓練側計算。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .config import get
from .models import (
    TARGETS,
    AlwaysUpBaseline,
    CalibratedModel,
    LogisticModel,
    MomentumRuleBaseline,
)

log = logging.getLogger(__name__)

LABEL_COLS = set(TARGETS) | {"fwd_ret_10", "fwd_ret_20"}


def _safe_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """計算單一機率序列的指標；皆轉為 Python 原生型別以利 JSON 序列化。"""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    n = int(len(y_true))
    if n == 0:
        return {"n": 0, "base_rate": None, "hit_rate": None,
                "brier": None, "auc": None}
    base_rate = float(np.mean(y_true))
    hit_rate = float(np.mean((proba >= 0.5) == (y_true >= 0.5)))
    brier = float(brier_score_loss(y_true, proba))
    auc = None
    if len(np.unique(y_true)) == 2:
        try:
            auc = float(roc_auc_score(y_true, proba))
        except ValueError:
            auc = None
    return {"n": n, "base_rate": base_rate, "hit_rate": hit_rate,
            "brier": brier, "auc": auc}


def _combined_metrics(y_true, p_model, p_always, p_mom) -> dict:
    m = _safe_metrics(y_true, p_model)
    m_always = _safe_metrics(y_true, p_always)
    m_mom = _safe_metrics(y_true, p_mom)
    out = {
        "n": m["n"],
        "base_rate": m["base_rate"],
        "hit_rate": m["hit_rate"],
        "brier": m["brier"],
        "auc": m["auc"],
        "hit_rate_always": m_always["hit_rate"],
        "brier_always": m_always["brier"],
        "auc_always": m_always["auc"],
        "hit_rate_mom": m_mom["hit_rate"],
        "brier_mom": m_mom["brier"],
        "auc_mom": m_mom["auc"],
    }
    return out


def walk_forward(panel: pd.DataFrame, feature_cols: list[str], cfg: dict) -> dict:
    """執行 walk-forward 驗證，回傳每目標的樣本外預測與指標。"""
    vcfg = get(cfg, "model", "validation", default={}) or {}
    ccfg = get(cfg, "model", "calibration", default={}) or {}
    lrcfg = get(cfg, "model", "logistic_regression", default={}) or {}

    refit_every = int(vcfg.get("refit_every_days", 63))
    min_train = int(vcfg.get("min_train_days", 500))
    embargo = int(vcfg.get("embargo_days", 20))
    n_splits_min = int(vcfg.get("n_splits_min", 8))
    cal_method = ccfg.get("method", "isotonic")
    min_cal_samples = int(ccfg.get("min_calibration_samples", 200))

    # 防禦：特徵欄位排除標籤欄（避免呼叫方誤把標籤當特徵傳入造成洩漏）
    feature_cols = [c for c in feature_cols if c not in LABEL_COLS and c in panel.columns]
    if not feature_cols:
        raise ValueError("沒有可用的特徵欄位。")

    dates = pd.DatetimeIndex(sorted(panel.index.get_level_values("date").unique()))
    if len(dates) <= min_train:
        raise ValueError(
            f"資料日期數（{len(dates)}）不足以進行 walk-forward（至少需要 {min_train} 天訓練）。"
        )

    targets = [t for t in TARGETS if t in panel.columns]
    missing = [t for t in TARGETS if t not in panel.columns]
    if missing:
        log.warning("缺少標籤欄位，已略過目標：%s", missing)
    if not targets:
        raise ValueError("panel 中沒有任何可用的預測目標標籤。")

    # 依日期層級快速切片（保留 MultiIndex (stock_id, date)）
    date_level = panel.index.get_level_values("date")

    results: dict = {"targets": {}}
    for target in targets:
        oos_rows: list[dict] = []
        periods: list[dict] = []

        starts = list(range(min_train, len(dates), refit_every))
        if len(starts) < n_splits_min:
            log.warning(
                "walk-forward 切點數（%d）少於建議下限（%d），樣本外評估效力有限。",
                len(starts),
                n_splits_min,
            )

        for i in starts:
            test_dates = dates[i: i + refit_every]
            train_dates = dates[: max(i - embargo, 0)]
            if len(train_dates) < 50 or len(test_dates) == 0:
                continue

            train = panel[date_level.isin(train_dates)]
            test = panel[date_level.isin(test_dates)]
            # 標籤缺值（資料尾端未來報酬未知）直接捨棄該列
            train_t = train.dropna(subset=[target])
            test_t = test.dropna(subset=[target])
            if train_t.empty or test_t.empty:
                continue

            # 訓練集內按日期切 80% fit / 20% cal（cal 為較晚日期）
            train_t_dates = pd.DatetimeIndex(
                sorted(train_t.index.get_level_values("date").unique())
            )
            cut = int(len(train_t_dates) * 0.8)
            fit_dates = train_t_dates[:cut]
            cal_dates = train_t_dates[cut:]
            fit_mask = train_t.index.get_level_values("date").isin(fit_dates)
            cal_mask = train_t.index.get_level_values("date").isin(cal_dates)
            X_fit, y_fit = train_t.loc[fit_mask, feature_cols], train_t.loc[fit_mask, target]
            X_cal, y_cal = train_t.loc[cal_mask, feature_cols], train_t.loc[cal_mask, target]
            X_test = test_t[feature_cols]
            y_test = test_t[target].to_numpy(dtype=float)

            # --- 訓練模型 ---
            lr = LogisticModel(
                C=float(lrcfg.get("C", 1.0)),
                max_iter=int(lrcfg.get("max_iter", 1000)),
                class_weight=lrcfg.get("class_weight", "balanced"),
            )
            model = CalibratedModel(
                lr,
                method=cal_method,
                min_calibration_samples=min_cal_samples,
            )
            model.fit(X_fit, y_fit, X_cal, y_cal)
            proba = np.asarray(model.predict_proba(X_test), dtype=float)

            # --- 基準模型（在完整訓練集上 fit）---
            X_train_full = train_t[feature_cols]
            y_train_full = train_t[target]
            always = AlwaysUpBaseline().fit(y_train_full)
            mom = MomentumRuleBaseline().fit(X_train_full, y_train_full)
            proba_always = always.predict_proba(X_test)
            proba_mom = mom.predict_proba(X_test)

            # --- 收集樣本外預測 ---
            idx = test_t.index
            for k in range(len(test_t)):
                oos_rows.append(
                    {
                        "stock_id": idx[k][0],
                        "date": idx[k][1],
                        "y_true": float(y_test[k]),
                        "proba": float(proba[k]),
                        "proba_always": float(proba_always[k]),
                        "proba_mom": float(proba_mom[k]),
                    }
                )

            # --- 本期指標 ---
            pm = _combined_metrics(y_test, proba, proba_always, proba_mom)
            periods.append(
                {
                    "train_start": str(train_dates[0].date()),
                    "train_end": str(train_dates[-1].date()),
                    "test_start": str(test_dates[0].date()),
                    "test_end": str(test_dates[-1].date()),
                    **pm,
                }
            )
            log.info(
                "目標 %s 期間 %s~%s：n=%d auc=%.3f（基準 %.3f）brier=%.4f（基準 %.4f）",
                target,
                periods[-1]["test_start"],
                periods[-1]["test_end"],
                pm["n"],
                pm["auc"] if pm["auc"] is not None else float("nan"),
                pm["auc_always"] if pm["auc_always"] is not None else float("nan"),
                pm["brier"] if pm["brier"] is not None else float("nan"),
                pm["brier_always"] if pm["brier_always"] is not None else float("nan"),
            )

        oos = pd.DataFrame(oos_rows)
        if not oos.empty:
            oos["date"] = pd.to_datetime(oos["date"])
            metrics = _combined_metrics(
                oos["y_true"].to_numpy(),
                oos["proba"].to_numpy(),
                oos["proba_always"].to_numpy(),
                oos["proba_mom"].to_numpy(),
            )
        else:
            metrics = _combined_metrics(np.array([]), np.array([]), np.array([]), np.array([]))

        results["targets"][target] = {
            "oos": oos,
            "metrics": metrics,
            "periods": periods,
        }
        log.info(
            "目標 %s walk-forward 完成：樣本外 n=%d，AUC=%.3f，Brier=%.4f",
            target,
            metrics["n"],
            metrics["auc"] if metrics["auc"] is not None else float("nan"),
            metrics["brier"] if metrics["brier"] is not None else float("nan"),
        )
    return results


def credibility_check(results: dict, cfg: dict) -> tuple[bool, list[str]]:
    """判斷模型是否具備可信的預測能力（以 up_10 為主）。

    條件（門檻皆來自 config credibility 區段）：
    - 樣本外樣本數 >= min_oos_samples
    - AUC 優於 always_up 基準 >= min_auc_improvement
    - Brier score 優於 always_up 基準 >= min_brier_improvement

    回傳 (是否可信, 原因字串列表)。任一條件不滿足都會寫清楚原因。
    """
    cfcfg = get(cfg, "credibility", default={}) or {}
    min_oos = int(cfcfg.get("min_oos_samples", 500))
    min_auc_imp = float(cfcfg.get("min_auc_improvement", 0.02))
    min_brier_imp = float(cfcfg.get("min_brier_improvement", 0.005))

    reasons: list[str] = []
    target = results.get("targets", {}).get("up_10")
    if target is None:
        return False, ["缺少 up_10 目標的驗證結果，無法評估可信度。"]
    m = target.get("metrics", {}) or {}

    n = int(m.get("n") or 0)
    if n < min_oos:
        reasons.append(f"樣本外樣本數僅 {n}，未達門檻 {min_oos}。")

    auc, auc_base = m.get("auc"), m.get("auc_always")
    if auc is None or auc_base is None:
        reasons.append("樣本外僅有單一類別（或類別不足），無法計算 ROC-AUC。")
    elif (auc - auc_base) < min_auc_imp:
        reasons.append(
            f"AUC 僅優於基準 {auc - auc_base:.4f}（模型 {auc:.4f} vs 基準 {auc_base:.4f}），"
            f"未達 {min_auc_imp} 門檻。"
        )

    brier, brier_base = m.get("brier"), m.get("brier_always")
    if brier is None or brier_base is None:
        reasons.append("無法計算 Brier score。")
    elif (brier_base - brier) < min_brier_imp:
        reasons.append(
            f"Brier score 僅優於基準 {brier_base - brier:.5f}"
            f"（模型 {brier:.5f} vs 基準 {brier_base:.5f}），"
            f"未達 {min_brier_imp} 門檻。"
        )

    credible = not reasons
    if credible:
        reasons.append(
            f"通過可信度檢驗：樣本外 n={n}，AUC {auc:.4f}（優於基準 {auc - auc_base:.4f}），"
            f"Brier {brier:.5f}（優於基準 {brier_base - brier:.5f}）。"
        )
    else:
        log.warning("模型未通過可信度檢驗：%s", "；".join(reasons))
    return credible, reasons
