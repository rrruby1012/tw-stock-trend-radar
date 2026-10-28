"""時間序列 walk-forward 驗證：絕不隨機切割資料，嚴防 look-ahead。

流程（每個預測目標獨立進行）：
1. 依日期排序，每 refit_every_days 個交易日開一個測試窗格。
2. 訓練集 = 測試窗格開始前 embargo_days 個交易日之前的所有資料。
3. 訓練集內再按日期切 80%（模型擬合）/ 20%（較晚日期，機率校準），
   且 fit 與 cal 之間再隔 embargo_days 個交易日（見 split_fit_cal_dates；
   train/test 的 embargo_days 數值本身不動，只是同一慣例值也用於 fit/cal）。
4. 在測試窗格上產生樣本外（out-of-sample）預測，彙總計算指標。
5. 四個基準分開計算（見 separate_baselines），並以日期區塊 bootstrap
   計算 CI 與閘門（見 block_bootstrap_ci）；既有 credibility_check 的
   門檻與行為不變，bootstrap 閘門結果另外回報，不覆蓋既有門檻。

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

# fit/cal embargo 的預設值（交易日數）。與 train/test embargo 取同一慣例值：
# 正式設定 embargo_days 預設即為 20 個交易日（見 config.yaml validation 區段），
# 本函數預設值只是讓設定缺失時仍與該慣例一致；walk_forward 與最終訓練實際
# 使用的是設定中的 embargo_days，絕不改動其數值。
FIT_CAL_EMBARGO_DEFAULT_DAYS = 20


def split_fit_cal_dates(dates, fit_fraction: float = 0.8,
                        embargo_days: int = FIT_CAL_EMBARGO_DEFAULT_DAYS):
    """把訓練窗的日期切成 fit / cal，兩段之間隔 embargo_days 個交易日。

    - fit 取前 fit_fraction 的日期；cal 從 fit 結束後再空 embargo_days
      個交易日才開始；被空出的日期兩邊都不用（不挪作 fit、cal 或測試）。
    - embargo 以「傳入日期序列中的個數」計，傳入的即是交易日序列，
      不以自然日換算（與 train/test embargo 的交易日計法一致）。
    - 若 embargo 後已無日期，cal 回傳空序列——呼叫方必須跳過該折，
      絕對不得回退為無 embargo 的相鄰切分（否則隔離保證會靜默消失）。

    回傳 (fit_dates, cal_dates)，皆為去重排序後的 DatetimeIndex。
    """
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    embargo_days = int(embargo_days)
    if embargo_days < 0:
        raise ValueError(f"embargo_days 不得為負：{embargo_days}")
    if not 0.0 < float(fit_fraction) < 1.0:
        raise ValueError(f"fit_fraction 必須在 (0, 1) 之間：{fit_fraction}")
    cut = int(len(dates) * float(fit_fraction))
    fit_dates = dates[:cut]
    cal_dates = dates[min(cut + embargo_days, len(dates)):]
    return fit_dates, cal_dates


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

        for fold_i, i in enumerate(starts):
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

            # 訓練集內按日期切 80% fit / 20% cal（cal 為較晚日期），且 fit 與
            # cal 之間隔 embargo_days 個交易日（與 train/test embargo 取同
            # 一慣例值，預設 20 日；train/test 的 embargo 數值本身不動）。
            # 審計原問題：舊切分 fit/cal 日期相鄰（cal 緊接 fit），標籤窗格
            # 重疊的樣本會同時出現在 fit 與 cal 兩側，造成校準洩漏。
            train_t_dates = pd.DatetimeIndex(
                sorted(train_t.index.get_level_values("date").unique())
            )
            fit_dates, cal_dates = split_fit_cal_dates(
                train_t_dates, fit_fraction=0.8, embargo_days=embargo
            )
            if len(fit_dates) == 0 or len(cal_dates) == 0:
                log.warning(
                    "目標 %s 期間 %s~%s：fit/cal embargo 後校準集為空，跳過該期間"
                    "（不回退為無 embargo 切分）。",
                    target, test_dates[0].date(), test_dates[-1].date(),
                )
                continue
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
            # train_base_rate = 本折訓練窗的基準率（氣候值），即 climatology
            # 基準的機率；舊欄位 proba_always 恰好也是此值（兩者同源），新欄位
            # proba_clim 只是把它明確命名，四基準分離見 separate_baselines。
            train_base_rate = float(np.mean(np.asarray(y_train_full, dtype=float)))
            idx = test_t.index
            for k in range(len(test_t)):
                oos_rows.append(
                    {
                        "stock_id": idx[k][0],
                        "date": idx[k][1],
                        "fold": fold_i,
                        "y_true": float(y_test[k]),
                        "proba": float(proba[k]),
                        "proba_always": float(proba_always[k]),
                        "proba_mom": float(proba_mom[k]),
                        "proba_const05": 0.5,
                        "proba_clim": train_base_rate,
                        "proba_always_rule": 1.0,
                    }
                )

            # --- 本期指標 ---
            pm = _combined_metrics(y_test, proba, proba_always, proba_mom)
            periods.append(
                {
                    "fold": fold_i,
                    "train_start": str(train_dates[0].date()),
                    "train_end": str(train_dates[-1].date()),
                    "fit_end": str(fit_dates[-1].date()),
                    "cal_start": str(cal_dates[0].date()),
                    "fit_cal_embargo_days": embargo,
                    "train_base_rate": train_base_rate,
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

        # 四基準分離計算與日期區塊 bootstrap CI（新增，不動既有 metrics 與
        # credibility_check 門檻；bootstrap 閘門結果另外回報）。
        baselines = separate_baselines(oos)
        bootstrap_ci = block_bootstrap_ci(oos)

        results["targets"][target] = {
            "oos": oos,
            "metrics": metrics,
            "periods": periods,
            "baselines": baselines,
            "bootstrap_ci": bootstrap_ci,
        }
        log.info(
            "目標 %s walk-forward 完成：樣本外 n=%d，AUC=%.3f，Brier=%.4f",
            target,
            metrics["n"],
            metrics["auc"] if metrics["auc"] is not None else float("nan"),
            metrics["brier"] if metrics["brier"] is not None else float("nan"),
        )
    return results


def _auc_for_series(y_true, proba):
    """單一序列的 AUC；常數機率（無排序資訊）按定義為 0.5，不呼叫 roc_auc_score。

    單一類別（或空序列）時 AUC 無定義，回傳 None——不得硬湊 0.5 掩蓋。
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(proba, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    if len(np.unique(p)) == 1:
        return 0.5
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return None


def _per_fold_auc(df: pd.DataFrame, proba) -> dict:
    """每折各自算 AUC（常數機率的折按定義為 0.5），絕不跨折混池。

    混池的陷阱：氣候值在每折內是常數（折內 AUC 恆 0.5），但跨折常數不同，
    混池算出的 AUC 不再是 0.5——把混池值當成氣候基準的 AUC 會高估基準。
    固定 0.5 基準因所有折常數相同，混池碰巧也是 0.5，但本函數仍一律按折
    計算，避免以後換基準時把混池習慣帶進來。
    """
    proba = np.asarray(proba, dtype=float)
    y = df["y_true"].to_numpy(dtype=float)
    out: dict = {}
    if "fold" in df.columns:
        folds = df["fold"].to_numpy()
        for f in sorted(set(folds.tolist())):
            mask = folds == f
            out[f] = _auc_for_series(y[mask], proba[mask])
    else:
        out["all"] = _auc_for_series(y, proba)
    return out


def _date_blocks(dates, block_days: int) -> list:
    """把已排序的唯一日期切成連續區塊；餘數併入最後一塊。

    保證：每塊日期互不重疊、聯集覆蓋全部日期；除非總日期數 < block_days，
    否則每塊都 ≥ block_days 個日期（餘數併入後最後一塊介於 block_days
    與 2*block_days-1 之間）。
    """
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    block_days = int(block_days)
    if block_days < 1:
        raise ValueError(f"block_days 必須 >= 1：{block_days}")
    blocks = [dates[i: i + block_days] for i in range(0, len(dates), block_days)]
    if len(blocks) > 1 and len(blocks[-1]) < block_days:
        blocks[-2] = blocks[-2].append(blocks[-1])
        blocks.pop()
    return blocks


def _empty_baseline(n: int, reason: str) -> dict:
    return {"n": int(n), "base_rate": None, "hit_rate": None, "brier": None,
            "auc": None, "auc_per_fold": {}, "reason": reason}


def _baseline_entry(y, proba, df: pd.DataFrame) -> dict:
    """單一基準的 pooled 指標 + 每折 AUC；常數機率的 AUC 以定義值 0.5 計。"""
    m = _safe_metrics(y, proba)
    m["auc"] = _auc_for_series(y, proba)
    m["auc_per_fold"] = _per_fold_auc(df, proba)
    return m


def separate_baselines(oos: pd.DataFrame | None) -> dict:
    """四個基準分開計算，各有獨立結果，互不覆蓋。

    (a) const05：固定機率 0.5（規則常數，與任何資料無關）；每折 AUC 按定義
        恆為 0.5，Brier 恆為 0.25。
    (b) climatology：僅用訓練窗的氣候值——呼叫方必須以 proba_clim 欄提供
        每折訓練窗基準率（walk_forward 已逐折寫入）。若缺此欄，本函數回傳
        None 並明說原因，絕對不以測試期 y_true 現算基準率回填（那是洩漏）。
    (c) always_up_rule：恆多分類規則，每列皆預測上漲（機率 1.0）；常數故
        每折 AUC 按定義為 0.5，hit_rate 等於測試期基準率。
    (d) momentum：動能基準，以 proba_mom 欄提供；缺欄時回傳 None 並明說。

    注意與舊欄位的關係：舊的 proba_always / metrics auc_always 等是「訓練窗
    基準率」本身（即 climatology 的機率），並非恆多規則；舊鍵與舊門檻一律
    不動，本函數只是把過去被「always」一名混用的兩個概念拆開明說。
    """
    if oos is None or len(oos) == 0:
        return {"n": 0,
                "const05": _empty_baseline(0, "沒有樣本外預測。"),
                "climatology": _empty_baseline(0, "沒有樣本外預測。"),
                "always_up_rule": _empty_baseline(0, "沒有樣本外預測。"),
                "momentum": _empty_baseline(0, "沒有樣本外預測。")}
    df = oos.copy()
    df["date"] = pd.to_datetime(df["date"])
    y = df["y_true"].to_numpy(dtype=float)
    n = int(len(df))
    out: dict = {"n": n}

    p_const = np.full(n, 0.5)
    entry = _baseline_entry(y, p_const, df)
    entry["source"] = "fixed 0.5 (rule constant, independent of data)"
    out["const05"] = entry

    if "proba_clim" in df.columns:
        entry = _baseline_entry(y, df["proba_clim"].to_numpy(dtype=float), df)
        entry["source"] = "training-window base rate per fold (proba_clim)"
        out["climatology"] = entry
    else:
        entry = _empty_baseline(n, "缺少 proba_clim（訓練窗基準率），"
                                "拒絕以測試期基準率回填，氣候基準無法計算。")
        out["climatology"] = entry

    if "proba_always_rule" in df.columns:
        p_rule = df["proba_always_rule"].to_numpy(dtype=float)
    else:
        p_rule = np.full(n, 1.0)
    entry = _baseline_entry(y, p_rule, df)
    entry["source"] = "always predict up (probability 1.0)"
    out["always_up_rule"] = entry

    if "proba_mom" in df.columns:
        entry = _baseline_entry(y, df["proba_mom"].to_numpy(dtype=float), df)
        entry["source"] = "momentum rule baseline (proba_mom)"
        out["momentum"] = entry
    else:
        entry = _empty_baseline(n, "缺少 proba_mom，動能基準無法計算。")
        out["momentum"] = entry
    return out


def block_bootstrap_ci(oos: pd.DataFrame | None, n_boot: int = 1000,
                       block_days: int = 20, alpha: float = 0.05,
                       random_state: int = 42) -> dict:
    """日期區塊 bootstrap 的 CI 與閘門（區塊以日期為單位抽樣）。

    - 區塊：把樣本外日期切成連續區塊（見 _date_blocks），每塊 ≥ block_days
      個交易日；同一日期的所有股票列永遠在同一區塊內，抽樣以整塊為單位，
      不拆散同日列（同日股票報酬高度相關，逐列抽樣會低估不確定性）。
    - 重抽：每次抽 n_blocks 塊（放回），在重抽樣本上算模型與各基準的指標
      差值；僅保留 y 同時含兩類的重抽（否則 AUC 無定義），有效重抽少於
      30 次時不給 CI（樣本不足，不得硬湊）。
    - CI：差值的百分位數區間（alpha/2，1-alpha/2）。
    - 閘門 gate_pass（判斷點，見回報）：CI(AUC - 0.5) 下限 > 0，且模型
      Brier 對「每一個可算基準」的差值 CI 上限 < 0；任一基準缺失（例如
      無訓練窗氣候值）時閘門不通過，不靜默放行。
    - 既有 credibility_check 的門檻與行為不變；本閘門結果另外回報，
      是否併入 credibility_check 屬後續決策，本次不動。
    """
    block_days = int(block_days)
    if block_days < 20:
        raise ValueError(f"區塊至少 20 個交易日：block_days={block_days}")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError(f"n_boot 必須 >= 1：{n_boot}")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha 必須在 (0, 1) 之間：{alpha}")

    ci_keys = ["auc_minus_const05", "brier_minus_const05", "brier_minus_climatology",
               "brier_minus_always_up_rule", "brier_minus_momentum"]
    base = {"block_days": block_days, "n_blocks": 0, "block_date_ranges": [],
            "n_boot": n_boot, "n_boot_valid": 0, "alpha": float(alpha),
            "gate_pass": False, "reasons": [], "baselines_evaluated": [],
            "missing_baselines": []}
    for k in ci_keys:
        base[k] = {"point": None, "lo": None, "hi": None}
    if oos is None or len(oos) == 0:
        base["reasons"] = ["沒有樣本外預測，無法計算 bootstrap CI。"]
        return base

    df = oos.reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    if len(dates) < block_days:
        base["reasons"] = [f"樣本外日期數（{len(dates)}）少於區塊下限"
                           f"（{block_days} 個交易日），無法計算 bootstrap CI。"]
        return base
    blocks = _date_blocks(dates, block_days)
    base["n_blocks"] = len(blocks)
    base["block_date_ranges"] = [
        {"start": str(b[0].date()), "end": str(b[-1].date()), "n_dates": int(len(b))}
        for b in blocks
    ]
    date_vals = df["date"].to_numpy()
    block_idx = [np.where(np.isin(date_vals, b.to_numpy()))[0] for b in blocks]

    y = df["y_true"].to_numpy(dtype=float)
    p_model = df["proba"].to_numpy(dtype=float)
    series = {
        "const05": np.full(len(df), 0.5),
        "climatology": (df["proba_clim"].to_numpy(dtype=float)
                        if "proba_clim" in df.columns else None),
        "always_up_rule": (df["proba_always_rule"].to_numpy(dtype=float)
                           if "proba_always_rule" in df.columns
                           else np.full(len(df), 1.0)),
        "momentum": (df["proba_mom"].to_numpy(dtype=float)
                     if "proba_mom" in df.columns else None),
    }
    evaluated = [k for k, v in series.items() if v is not None]
    missing = [k for k, v in series.items() if v is None]
    base["baselines_evaluated"] = evaluated
    base["missing_baselines"] = missing

    def _brier(a, b):
        return float(brier_score_loss(a, b))

    if len(np.unique(y)) < 2:
        base["reasons"] = ["樣本外僅有單一類別，無法計算 AUC 差值 CI。"]
        return base
    model_auc = _auc_for_series(y, p_model)
    model_brier = _brier(y, p_model)
    point = {"auc_minus_const05": (model_auc - 0.5) if model_auc is not None else None}
    for k in evaluated:
        point[f"brier_minus_{k}"] = model_brier - _brier(y, series[k])
    for k in ci_keys:
        base[k]["point"] = point.get(k)

    rng = np.random.default_rng(random_state)
    diffs: dict[str, list[float]] = {k: [] for k in ci_keys}
    n_valid = 0
    for _ in range(n_boot):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([block_idx[c] for c in chosen])
        y_r = y[idx]
        if len(np.unique(y_r)) < 2:
            continue
        auc_r = _auc_for_series(y_r, p_model[idx])
        if auc_r is None:
            continue
        diffs["auc_minus_const05"].append(auc_r - 0.5)
        brier_r = _brier(y_r, p_model[idx])
        for k in evaluated:
            diffs[f"brier_minus_{k}"].append(brier_r - _brier(y_r, series[k][idx]))
        n_valid += 1
    base["n_boot_valid"] = n_valid
    if n_valid < 30:
        base["reasons"] = [f"有效重抽樣僅 {n_valid} 次（少於 30 次），"
                           "不給出 CI（樣本不足，不得硬湊）。"]
        return base
    lo_q, hi_q = float(alpha) / 2.0, 1.0 - float(alpha) / 2.0
    for k in ci_keys:
        vals = diffs[k]
        if not vals:
            continue
        base[k]["lo"] = float(np.quantile(vals, lo_q))
        base[k]["hi"] = float(np.quantile(vals, hi_q))

    reasons: list[str] = []
    auc_ci = base["auc_minus_const05"]
    auc_ok = auc_ci["lo"] is not None and auc_ci["lo"] > 0
    if not auc_ok:
        reasons.append(f"CI(AUC-0.5) 下限未 > 0（lo={auc_ci['lo']}）。")
    brier_ok = True
    for k in evaluated:
        ci = base[f"brier_minus_{k}"]
        if ci["hi"] is None or not ci["hi"] < 0:
            brier_ok = False
            reasons.append(f"CI(模型 Brier-{k} Brier) 上限未 < 0（hi={ci['hi']}）。")
    for k in missing:
        reasons.append(f"基準 {k} 無法計算（缺對應機率欄），閘門不通過。")
    base["gate_pass"] = bool(auc_ok and brier_ok and not missing)
    base["reasons"] = reasons or ["通過 bootstrap 閘門：CI(AUC-0.5) 下限 > 0，"
                                  "且對每個可算基準的 Brier 差值 CI 上限 < 0。"]
    return base


def credibility_check(results: dict, cfg: dict) -> tuple[bool, list[str], str]:
    """判斷模型是否具備可信的預測能力（以 up_10 為主）。

    條件（門檻皆來自 config credibility 區段）：
    - 樣本外樣本數 >= min_oos_samples
    - AUC 優於 always_up 基準 >= min_auc_improvement
    - Brier score 優於 always_up 基準 >= min_brier_improvement

    回傳 (是否可信, 原因字串列表, 狀態)。
    狀態為 "ok"（通過）、"failed"（驗證執行成功但未優於基準）、
    "insufficient"（驗證未執行或資料不足，無法評估）。
    任一條件不滿足都會寫清楚原因。
    """
    cfcfg = get(cfg, "credibility", default={}) or {}
    min_oos = int(cfcfg.get("min_oos_samples", 500))
    min_auc_imp = float(cfcfg.get("min_auc_improvement", 0.02))
    min_brier_imp = float(cfcfg.get("min_brier_improvement", 0.005))

    reasons: list[str] = []
    insufficient = False
    target = results.get("targets", {}).get("up_10")
    if target is None:
        return False, ["缺少 up_10 目標的驗證結果，無法評估可信度。"], "insufficient"
    m = target.get("metrics", {}) or {}

    n = int(m.get("n") or 0)
    if n < min_oos:
        reasons.append(f"樣本外樣本數僅 {n}，未達門檻 {min_oos}。")
        insufficient = True

    auc, auc_base = m.get("auc"), m.get("auc_always")
    if auc is None or auc_base is None:
        reasons.append("樣本外僅有單一類別（或類別不足），無法計算 ROC-AUC。")
        insufficient = True
    elif (auc - auc_base) < min_auc_imp:
        reasons.append(
            f"AUC 僅優於基準 {auc - auc_base:.4f}（模型 {auc:.4f} vs 基準 {auc_base:.4f}），"
            f"未達 {min_auc_imp} 門檻。"
        )

    brier, brier_base = m.get("brier"), m.get("brier_always")
    if brier is None or brier_base is None:
        reasons.append("無法計算 Brier score。")
        insufficient = True
    elif (brier_base - brier) < min_brier_imp:
        reasons.append(
            f"Brier score 僅優於基準 {brier_base - brier:.5f}"
            f"（模型 {brier:.5f} vs 基準 {brier_base:.5f}），"
            f"未達 {min_brier_imp} 門檻。"
        )

    credible = not reasons
    if credible:
        status = "ok"
        reasons.append(
            f"通過可信度檢驗：樣本外 n={n}，AUC {auc:.4f}（優於基準 {auc - auc_base:.4f}），"
            f"Brier {brier:.5f}（優於基準 {brier_base - brier:.5f}）。"
        )
    else:
        status = "insufficient" if insufficient else "failed"
        log.warning("模型未通過可信度檢驗：%s", "；".join(reasons))
    return credible, reasons, status
