"""模型輸出健康檢查：集中偵測「整批預測機率失去辨識度」。

背景（2026-11-05 診斷）：公開站 p_up_10 自 10/29 的 6 種值逐日收斂，
11/05 全數 50 檔皆為 0.609、p_up_20 全為 0.693、趨勢全為「偏多」。
重現證實：校準前機率仍有 50 種值（raw decision_function 標準差約
0.12~0.54），isotonic 校準後只剩 2~4 種值——校準器的階梯平台把
不同原始分數壓成同一輸出。健康檢查的目的，是讓這種退化在「發布前」
就被抓住，而不是等使用者在公開站上發現。

本模組只「偵測與定性」，絕不「修飾輸出」：
- 觸發時呼叫方必須把 model_valid / credible 強制為 false，並在網站
  實驗區顯示「模型輸出失去辨識度」，不得顯示偏多、偏空、排名或推薦。
- 禁止用亂數、人工展開、重新縮放或 raw score 排名來製造表面差異；
  退化的正確處理是換校準器（見 C 階段）或修上游特徵，不是把數字攤開。

偵測規則（每一個目標各自判定，任一主要目標退化即整批退化）：
1. ALL_EQUAL：校準後機率全數相同（nunique == 1）。
2. NEAR_ZERO_VARIANCE：校準後機率標準差 < 0.005（50 檔時約等於
   全部擠在 ±0.01 內；11/03 的 47×0.609＋3×0.592 標準差約 0.004，
   11/04 的 48×0.609＋2×0.608 標準差約 0.0003，皆會觸發）。
3. LOW_DISTINCT：校準後相異值過少（nunique <= max(2, n // 10)；
   50 檔即 <= 5 種值。連續型邏輯迴歸的校準後機率若只剩個位數種值，
   只可能來自校準平台，不可能是模型本身的輸出）。
4. RANK_LOSS：校準後失去原始排序——校準前相異、校準後成對平手
   （tie）的比例超過 50%。isotonic 的階梯平台會把大量不同原始分數
   壓成同一值，排序資訊直接消失。
5. OUT_OF_RANGE_ALL：最新原始分數全部落在 isotonic 校準器
   X_thresholds_ 範圍之外（out_of_bounds="clip" 會把它們全夾到
   同一邊緣平台，輸出恆為常數）。部分超出只記為警示，不單獨觸發。
6. DIRECTION_REVERSAL：校準後排序方向反轉——有效配對樣本
   ≥ MIN_DIRECTION_SAMPLES（10）時，Spearman（校準前，校準後）< 0。
   Platt（sigmoid）校準的斜率無約束：校準窗的分數—標籤關係為負時
   斜率為負，排序直接反轉（2026-11-06 稽核：28 次 walk-forward 擬合
   中有 5 次斜率為負，例如 up_10 第 7 折斜率 −0.36、逐折 AUC 0.4016；
   up_20 第 2/3/12/13 折亦然）。isotonic 單調非遞減，永遠不會反轉；
   但凡 Spearman < 0，輸出不是「無辨識度」而是「方向錯誤」，必須攔住。
   判定只做安全止血：不以未校準分數當機率、不取絕對值、不翻轉符號、
   不以亂數或人工展開修飾；呼叫方照既有退化路徑處理即可。

原因判定（依序，取最上游者）：
- feature_collapse：最新特徵列跨股票重複或大半欄位為常數。
- upstream_collapse：校準前機率本身已全同或近零變異。
- calibration_direction_reversal：校準後排序方向反轉（Spearman < 0）。
- calibration_saturation：校準前有辨識力、校準後塌縮。
- calibration_sample_insufficient：校準未完成（樣本不足或單一類別）
  且輸出同時退化。
- out_of_calibration_range：最新分數超出校準範圍且輸出退化。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------- 門檻（集中定義，測試直接引用） ----------
# 近零變異：校準後機率標準差低於此值即視為失去辨識度。
NEAR_ZERO_STD = 0.005
# 排序損失：校準後成對平手比例高於此值即視為失去原始排序。
RANK_LOSS_TIE_FRACTION = 0.5
# 相異值過少：50 檔時為 5 種值；一般寫法為 max(2, n // 10)。
LOW_DISTINCT_FRACTION_DENOMINATOR = 10
# 方向反轉：Spearman 判定所需的最小有效配對樣本數；少於此數時
# 等級相關不穩定，不判定反轉（避免小樣本誤判）。正常發布為 50 檔。
MIN_DIRECTION_SAMPLES = 10


def _describe(values) -> dict:
    """單一機率序列的描述統計（皆為 Python 原生型別，可直接寫 JSON）。"""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0, "min": None, "max": None, "mean": None,
                "std": None, "nunique": 0}
    return {"n": int(len(a)), "min": float(np.min(a)), "max": float(np.max(a)),
            "mean": float(np.mean(a)), "std": float(np.std(a)),
            "nunique": int(len(np.unique(np.round(a, 12))))}


def _spearman(a, b) -> float | None:
    """Spearman 等級相關；任一序列無變異時回傳 None（等級無定義）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or len(np.unique(a)) <= 1 or len(np.unique(b)) <= 1:
        return None
    try:
        return float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    except Exception:  # noqa: BLE001
        return None


def _tie_fraction(values) -> float:
    """成對平手比例：任取兩列，校準後值相等的比例。

    連續型機率不應大量平手；isotonic 階梯平台會讓此比例飆高
    （11/05 公開站：全部 50 檔同值，平手比例 = 1.0）。
    """
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 2:
        return 0.0
    _, counts = np.unique(np.round(a, 12), return_counts=True)
    tied_pairs = float(np.sum(counts * (counts - 1) / 2))
    total_pairs = float(n * (n - 1) / 2)
    return tied_pairs / total_pairs if total_pairs else 0.0


def _isotonic_range(calibrator) -> dict | None:
    """讀 isotonic 校準器的 X 閾值範圍；非 isotonic 時回傳 None。"""
    if calibrator is None:
        return None
    x_thr = getattr(calibrator, "X_thresholds_", None)
    y_thr = getattr(calibrator, "y_thresholds_", None)
    if x_thr is None or y_thr is None:
        return None
    x_thr = np.asarray(x_thr, dtype=float)
    y_thr = np.asarray(y_thr, dtype=float)
    return {"x_min": float(np.min(x_thr)), "x_max": float(np.max(x_thr)),
            "y_min": float(np.min(y_thr)), "y_max": float(np.max(y_thr)),
            "n_thresholds": int(len(x_thr)),
            "y_nunique": int(len(np.unique(y_thr)))}


def assess_target(target: str, pre, post, calibrator=None,
                  calibrated: bool | None = None) -> dict:
    """判定單一目標的輸出是否退化，並回傳統計與原因線索。

    pre：校準前機率（底層 logistic 輸出）；post：校準後機率。
    calibrator：已擬合的校準器（isotonic 時檢查範圍；sigmoid 時無範圍）。
    calibrated：校準是否完成（False 表示校準集不足而退回原始機率）。
    """
    pre_s = _describe(pre)
    post_s = _describe(post)
    reasons: list[str] = []
    flags: dict[str, bool] = {}
    n = post_s["n"]

    # 規則 1：校準後全數相同。
    flags["all_equal"] = bool(n >= 2 and post_s["nunique"] == 1)
    if flags["all_equal"]:
        reasons.append(f"{target} 校準後機率全數相同（{n} 檔皆為 {post_s['min']}）。")
    # 規則 2：近零變異。
    flags["near_zero_variance"] = bool(
        n >= 10 and post_s["std"] is not None and post_s["std"] < NEAR_ZERO_STD)
    if flags["near_zero_variance"]:
        reasons.append(f"{target} 校準後機率近零變異（標準差 {post_s['std']:.6f} < {NEAR_ZERO_STD}）。")
    # 規則 3：相異值過少。
    low_distinct_limit = max(2, n // LOW_DISTINCT_FRACTION_DENOMINATOR) if n >= 20 else 2
    flags["low_distinct"] = bool(n >= 20 and post_s["nunique"] <= low_distinct_limit)
    if flags["low_distinct"]:
        reasons.append(f"{target} 校準後相異值過少（{post_s['nunique']} 種值 <= {low_distinct_limit}）。")
    # 規則 4：校準後失去原始排序（成對平手比例過高）。
    tie = _tie_fraction(post)
    flags["rank_loss"] = bool(pre_s["nunique"] > 1 and tie > RANK_LOSS_TIE_FRACTION)
    if flags["rank_loss"]:
        reasons.append(f"{target} 校準後失去原始排序（成對平手比例 {tie:.2f} > {RANK_LOSS_TIE_FRACTION}）。")
    # 規則 5：最新分數超出校準範圍（僅 isotonic 有範圍概念）。
    iso = _isotonic_range(calibrator)
    out_frac = None
    if iso is not None and pre_s["n"] > 0:
        pre_arr = np.asarray(pre, dtype=float)
        pre_arr = pre_arr[np.isfinite(pre_arr)]
        out_frac = float(np.mean((pre_arr < iso["x_min"]) | (pre_arr > iso["x_max"]))) if len(pre_arr) else 0.0
    flags["out_of_range_all"] = bool(out_frac is not None and out_frac >= 1.0)
    if flags["out_of_range_all"]:
        reasons.append(f"{target} 最新原始分數全部落在校準範圍之外（isotonic 範圍 "
                       f"[{iso['x_min']:.4f}, {iso['x_max']:.4f}]），會被夾到同一邊緣平台。")
    spearman = _spearman(pre, post)
    # 規則 6：校準後排序方向反轉（僅 sigmoid 可能；isotonic 單調非遞減，
    # 永遠不會反轉）。有效配對樣本須足夠，否則不判定。
    a = np.asarray(pre, dtype=float)
    b = np.asarray(post, dtype=float)
    m = min(len(a), len(b))
    valid_pairs = int(np.sum(np.isfinite(a[:m]) & np.isfinite(b[:m]))) if m else 0
    flags["direction_reversal"] = bool(
        spearman is not None and spearman < 0
        and valid_pairs >= MIN_DIRECTION_SAMPLES)
    if flags["direction_reversal"]:
        reasons.append(
            f"{target} 校準後排序方向反轉（Spearman {spearman:.3f} < 0，"
            f"有效樣本 {valid_pairs} 檔；原始分數愈高、校準後機率反而愈低，"
            f"疑似負斜率 Platt 校準）。")
    degenerate = bool(any(flags.values()))
    return {"target": target, "degenerate": degenerate, "reasons": reasons,
            "flags": flags, "pre": pre_s, "post": post_s,
            "tie_fraction": float(tie), "spearman_pre_post": spearman,
            "isotonic": iso, "out_of_range_fraction": out_frac,
            "calibrated": calibrated}


def assess_latest_features(X_latest: pd.DataFrame | None) -> dict:
    """檢查最新特徵列是否跨股票塌縮（重複列、常數欄）。

    回傳 degenerate=True 表示特徵層本身已失去辨識度：最新 50 列幾乎
    相同，或過半欄位在 50 檔間完全是常數。正常情況下只有少數欄位
    （例如全為 0 的籌碼欄）會是常數，絕不應過半。
    """
    if X_latest is None or len(X_latest) == 0:
        return {"degenerate": False, "reasons": ["無最新特徵列可檢查。"],
                "n_rows": 0, "duplicate_rows": 0, "constant_cols": [],
                "constant_fraction": None}
    n_rows = int(len(X_latest))
    dup_rows = int(X_latest.duplicated().sum())
    const_cols = [c for c in X_latest.columns if X_latest[c].nunique(dropna=False) <= 1]
    const_frac = len(const_cols) / max(len(X_latest.columns), 1)
    reasons: list[str] = []
    degenerate = False
    if dup_rows >= max(1, n_rows // 2):
        degenerate = True
        reasons.append(f"最新特徵列大量重複（{dup_rows}/{n_rows} 列與他檔完全相同）。")
    if const_frac > 0.5:
        degenerate = True
        reasons.append(f"最新特徵欄位過半為常數（{len(const_cols)}/{len(X_latest.columns)} 欄）。")
    return {"degenerate": degenerate, "reasons": reasons, "n_rows": n_rows,
            "duplicate_rows": dup_rows, "constant_cols": const_cols,
            "constant_fraction": float(const_frac)}


def check_prediction_health(targets: dict, latest_features=None) -> dict:
    """集中式輸出健康檢查（供 scripts/update.py 在發布前呼叫）。

    targets：{目標名: {"pre": 校準前機率, "post": 校準後機率,
              "calibrator": 校準器或 None, "calibrated": 是否完成校準}}。
    只檢查 up_10 / up_20 兩個主要目標（exc_* 為輔助目標，不單獨觸發；
    但其統計仍會列入 targets 明細供查核）。
    回傳 {"degenerate": bool, "cause": 原因分類或 None,
           "reasons": [...], "targets": {...}, "features": {...}}。
    cause 依序取最上游者：feature_collapse > upstream_collapse >
    calibration_direction_reversal > calibration_saturation >
    calibration_sample_insufficient > out_of_calibration_range。
    方向反轉排在校準飽和之前：輸出「方向錯誤」比「沒有方向」更嚴重，
    不得以飽和之名掩蓋反轉。
    """
    primary = [t for t in ("up_10", "up_20") if t in targets]
    checked = primary if primary else sorted(targets.keys())
    per_target: dict[str, dict] = {}
    for t in sorted(targets.keys()):
        info = targets[t] or {}
        per_target[t] = assess_target(
            t, info.get("pre", []), info.get("post", []),
            calibrator=info.get("calibrator"),
            calibrated=info.get("calibrated"))
    features = assess_latest_features(latest_features)
    reasons: list[str] = []
    for t in checked:
        reasons.extend(per_target[t]["reasons"])
    reasons.extend(features["reasons"] if features["degenerate"] else [])
    degenerate = bool(any(per_target[t]["degenerate"] for t in checked)
                      or features["degenerate"])

    # 原因分類：取最上游者（特徵 > 原始分數 > 方向反轉 > 校準飽和 >
    # 校準樣本 > 超出範圍）。方向反轉優先於飽和：見上方 docstring。
    cause = None
    if degenerate:
        pre_collapsed = any(
            per_target[t]["pre"]["nunique"] <= 1
            or (per_target[t]["pre"]["std"] is not None
                and per_target[t]["pre"]["std"] < 1e-12)
            for t in checked)
        post_collapsed_with_pre_spread = any(
            per_target[t]["degenerate"] and per_target[t]["pre"]["nunique"] > 1
            for t in checked)
        not_calibrated = any(
            per_target[t].get("calibrated") is False for t in checked)
        all_out = any(
            per_target[t]["flags"].get("out_of_range_all") for t in checked)
        reversed_any = any(
            per_target[t]["flags"].get("direction_reversal") for t in checked)
        if features["degenerate"] and pre_collapsed:
            cause = "feature_collapse"
        elif pre_collapsed:
            cause = "upstream_collapse"
        elif reversed_any:
            cause = "calibration_direction_reversal"
        elif post_collapsed_with_pre_spread:
            cause = "calibration_saturation"
        elif not_calibrated:
            cause = "calibration_sample_insufficient"
        elif all_out:
            cause = "out_of_calibration_range"
        else:
            cause = "calibration_saturation"
    return {"degenerate": bool(degenerate), "cause": cause,
            "reasons": reasons, "targets": per_target,
            "features": features}


def health_summary(health: dict) -> str:
    """給 log / meta 用的單行摘要（不含機率細節，只含判定與原因）。"""
    if not health.get("degenerate"):
        return "模型輸出健康：未偵測到退化。"
    return ("模型輸出退化（原因：" + str(health.get("cause")) + "）："
            + "；".join(health.get("reasons", [])))
