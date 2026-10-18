"""模型層：基準模型、可解釋的機器學習模型、機率校準、價格區間與趨勢判定。

設計原則：
- 所有模型只輸出「機率估計值」，不輸出保證獲利之類的斷言。
- 基準模型（全部猜上漲、簡單動能規則）用來檢驗 ML 模型是否真的學到東西。
- ML 模型選用邏輯迴歸（可解釋：係數即特徵方向與強度），並做機率校準。
- 缺值填補的統計量只從訓練集計算，避免測試資料洩漏。
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

TARGETS = ["up_10", "up_20", "exc_10", "exc_20"]
"""四個預測目標：10/20 日上漲、10/20 日跑贏大盤。"""


def _to_frame(X: pd.DataFrame | np.ndarray, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=columns)


class AlwaysUpBaseline:
    """基準模型：全部猜上漲，機率 = 訓練集的上漲比例。"""

    def __init__(self):
        self.base_rate_: float = 0.5

    def fit(self, y) -> "AlwaysUpBaseline":
        y = np.asarray(y, dtype=float)
        y = y[~np.isnan(y)]
        self.base_rate_ = float(np.mean(y)) if len(y) else 0.5
        # 夾在 (0, 1) 避免退化機率
        self.base_rate_ = min(max(self.base_rate_, 1e-6), 1 - 1e-6)
        return self

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        return np.full(n, self.base_rate_, dtype=float)


class MomentumRuleBaseline:
    """基準模型：簡單動能規則。

    以訓練集估計 P(up | ret_20 > 0) 與 P(up | ret_20 <= 0)，
    預測時依每列的 ret_20 正負回傳對應機率。
    """

    def __init__(self, momentum_col: str = "ret_20"):
        self.momentum_col = momentum_col
        self.p_up_given_pos_: float = 0.5
        self.p_up_given_neg_: float = 0.5
        self.base_rate_: float = 0.5

    def fit(self, X, y) -> "MomentumRuleBaseline":
        # 純位置運算，避免 MultiIndex 下布林遮罩對齊問題
        X = _to_frame(X).reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        valid = ~np.isnan(y)
        yv = y[valid]
        self.base_rate_ = float(np.mean(yv)) if len(yv) else 0.5
        if self.momentum_col in X.columns and len(yv):
            mom = pd.to_numeric(X[self.momentum_col], errors="coerce").to_numpy()[valid]
            pos = (mom > 0) & ~np.isnan(mom)
            neg = (mom <= 0) & ~np.isnan(mom)
            if pos.sum() > 0:
                self.p_up_given_pos_ = float(np.mean(yv[pos]))
            if neg.sum() > 0:
                self.p_up_given_neg_ = float(np.mean(yv[neg]))
        for attr in ("p_up_given_pos_", "p_up_given_neg_", "base_rate_"):
            v = getattr(self, attr)
            setattr(self, attr, min(max(v, 1e-6), 1 - 1e-6))
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = _to_frame(X)
        n = len(X)
        out = np.full(n, self.base_rate_, dtype=float)
        if self.momentum_col in X.columns:
            mom = pd.to_numeric(X[self.momentum_col], errors="coerce").to_numpy()
            pos = (mom > 0) & ~np.isnan(mom)
            neg = (mom <= 0) & ~np.isnan(mom)
            out[pos] = self.p_up_given_pos_
            out[neg] = self.p_up_given_neg_
        return out


class LogisticModel:
    """可解釋 ML 模型：標準化 + 邏輯迴歸。

    缺值以訓練集各欄位中位數填補（填補值只從訓練集計算，不看測試集）。
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 class_weight: str = "balanced", random_state: int = 42):
        self.pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        class_weight=class_weight,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.medians_: pd.Series | None = None
        self.feature_names_: list[str] | None = None

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = _to_frame(X, self.feature_names_).copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.medians_ is None:
            med = X.median(numeric_only=True)
            med = med.fillna(0.0)  # 全缺值的欄位以 0 填補
            self.medians_ = med
        # 對齊欄位順序（預測時欄位可能與訓練時不同）
        X = X.reindex(columns=self.medians_.index)
        return X.fillna(self.medians_)

    def fit(self, X, y) -> "LogisticModel":
        X = _to_frame(X)
        self.feature_names_ = list(X.columns)
        y = np.asarray(y, dtype=float)
        mask = ~np.isnan(y)
        X_imp = self._impute(X.iloc[mask])
        self.pipe.fit(X_imp, y[mask])
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_imp = self._impute(_to_frame(X, self.feature_names_))
        return self.pipe.predict_proba(X_imp)[:, 1]

    @property
    def coefficients_(self) -> pd.Series | None:
        """回傳特徵係數（可解釋性用）；尚未訓練時回傳 None。"""
        if self.feature_names_ is None:
            return None
        coef = self.pipe.named_steps["clf"].coef_[0]
        return pd.Series(coef, index=self.feature_names_).sort_values()


class CalibratedModel:
    """包裝任一機率模型，加上等調迴歸（isotonic）機率校準。

    fit(X_fit, y_fit, X_cal, y_cal)：在 fit 集訓練底層模型，
    在 cal 集（較晚的日期）上訓練校準映射。校準集樣本不足或
    僅有單一類別時，退回未校準機率並記錄警告。
    """

    def __init__(self, estimator, method: str = "isotonic",
                 min_calibration_samples: int = 200):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"不支援的校準方法：{method}")
        self.estimator = estimator
        self.method = method
        self.min_calibration_samples = min_calibration_samples
        self.calibrator_: IsotonicRegression | None = None
        self.calibrated_: bool = False

    def fit(self, X_fit, y_fit, X_cal, y_cal) -> "CalibratedModel":
        self.estimator.fit(X_fit, y_fit)
        y_cal = np.asarray(y_cal, dtype=float)
        cal_mask = ~np.isnan(y_cal)
        n_cal = int(cal_mask.sum())
        n_classes = len(np.unique(y_cal[cal_mask])) if n_cal else 0
        if n_cal < self.min_calibration_samples or n_classes < 2:
            log.warning(
                "校準集樣本不足（n=%d，類別數=%d），略過校準直接使用原始機率。",
                n_cal,
                n_classes,
            )
            self.calibrated_ = False
            return self
        proba_cal = np.asarray(self.estimator.predict_proba(X_cal), dtype=float)
        proba_cal = np.clip(proba_cal, 1e-6, 1 - 1e-6)
        if self.method == "isotonic":
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(proba_cal[cal_mask], y_cal[cal_mask])
        else:  # sigmoid（Platt scaling）：以邏輯迴歸擬合
            from sklearn.linear_model import LogisticRegression as _LR

            lr = _LR()
            lr.fit(proba_cal[cal_mask].reshape(-1, 1), y_cal[cal_mask])
            self.calibrator_ = lr
        self.calibrated_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.asarray(self.estimator.predict_proba(X), dtype=float)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        if not self.calibrated_ or self.calibrator_ is None:
            return p
        if isinstance(self.calibrator_, IsotonicRegression):
            return np.asarray(self.calibrator_.predict(p), dtype=float)
        return self.calibrator_.predict_proba(p.reshape(-1, 1))[:, 1]


# ---------- 價格區間與趨勢 ----------

def price_interval(close: float, annual_vol: float, horizon_days: int) -> dict | None:
    """常態近似估計未來 h 個交易日的 80% 價格區間與下行風險。

    - daily_vol = annual_vol / sqrt(252)
    - h 期波動 = daily_vol * sqrt(h)
    - 80% 區間用 z=1.28；下行風險用 5% VaR（z=1.645），以報酬率表示。
    - 波動度缺值或不合法時回傳 None（呼叫方應標示「無法估計」而非硬湊數字）。
    """
    try:
        close_f = float(close)
        vol = float(annual_vol)
        h = int(horizon_days)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(close_f) or close_f <= 0:
        return None
    if not math.isfinite(vol) or vol <= 0 or h <= 0:
        return None
    h_vol = vol / math.sqrt(252) * math.sqrt(h)
    return {
        "lo": close_f * (1 - 1.28 * h_vol),
        "hi": close_f * (1 + 1.28 * h_vol),
        "downside_var5": -1.645 * h_vol,  # 報酬率表示的 5% VaR
    }


def classify_trend(p_up_10: float, p_up_20: float,
                   bullish_thr: float, bearish_thr: float) -> str:
    """依 10 日/20 日上漲機率的平均判定趨勢：偏多、盤整、偏空。"""
    score = (float(p_up_10) + float(p_up_20)) / 2.0
    if score >= bullish_thr:
        return "偏多"
    if score <= bearish_thr:
        return "偏空"
    return "盤整"
