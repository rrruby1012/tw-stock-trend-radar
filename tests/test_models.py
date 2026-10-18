"""模型層測試：基準模型、校準、價格區間、趨勢判定。"""
import numpy as np
import pandas as pd
import pytest

from src.models import (
    AlwaysUpBaseline,
    CalibratedModel,
    LogisticModel,
    MomentumRuleBaseline,
    classify_trend,
    price_interval,
)


def test_AlwaysUpBaseline_機率等於訓練集上漲比例():
    y = np.array([1] * 70 + [0] * 30, dtype=float)
    model = AlwaysUpBaseline().fit(y)
    proba = model.predict_proba(pd.DataFrame({"a": range(5)}))
    assert np.allclose(proba, 0.7)


def test_AlwaysUpBaseline_空標籤不報錯():
    model = AlwaysUpBaseline().fit(np.array([]))
    assert model.predict_proba([1, 2, 3]).tolist() == [0.5, 0.5, 0.5]


def test_MomentumRuleBaseline_依ret_20正負給不同機率():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"ret_20": [0.05] * 50 + [-0.05] * 50,
                      "noise": rng.normal(size=100)})
    y = np.array([1] * 50 + [0] * 50, dtype=float)  # 動能為正則上漲
    model = MomentumRuleBaseline().fit(X, y)
    proba = model.predict_proba(X)
    assert (proba[:50] > 0.9).all(), "ret_20>0 時機率應接近 1"
    assert (proba[50:] < 0.1).all(), "ret_20<=0 時機率應接近 0"


def test_MomentumRuleBaseline_缺ret_20欄位時退回基準比例():
    X = pd.DataFrame({"other": [1.0, 2.0]})
    model = MomentumRuleBaseline().fit(X, np.array([1.0, 1.0, 0.0, 0.0]))
    proba = model.predict_proba(X)
    assert np.allclose(proba, 0.5)


def test_LogisticModel_缺值填補只用訓練集統計量():
    X_train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 1.0, 1.0, 1.0]})
    y_train = np.array([0, 0, 1, 1], dtype=float)
    model = LogisticModel().fit(X_train, y_train)
    # 訓練集 a 欄中位數 = 2.0；預測時 NaN 應以 2.0 填補（不看測試集）
    assert model.medians_["a"] == pytest.approx(2.0)
    proba = model.predict_proba(pd.DataFrame({"a": [np.nan], "b": [1.0]}))
    assert 0.0 <= proba[0] <= 1.0


def test_CalibratedModel_樣本不足時退回未校準():
    X = pd.DataFrame({"a": np.arange(20, dtype=float)})
    y = np.array([0, 1] * 10, dtype=float)
    cal = CalibratedModel(LogisticModel(), min_calibration_samples=1000)
    cal.fit(X, y, X.iloc[:5], y[:5])
    assert cal.calibrated_ is False  # 樣本不足 → 不校準也不報錯
    assert len(cal.predict_proba(X)) == 20


def test_price_interval_正常情況lo小於收盤小於hi():
    iv = price_interval(100.0, 0.25, 10)
    assert iv is not None
    assert iv["lo"] < 100.0 < iv["hi"]
    assert iv["downside_var5"] < 0  # 下行風險以負報酬表示


def test_price_interval_波動度為0或NaN回傳None():
    assert price_interval(100.0, 0.0, 10) is None
    assert price_interval(100.0, float("nan"), 10) is None
    assert price_interval(100.0, -0.1, 10) is None
    assert price_interval(None, 0.25, 10) is None


def test_price_interval_期間越長區間越寬():
    iv10 = price_interval(100.0, 0.25, 10)
    iv20 = price_interval(100.0, 0.25, 20)
    assert (iv20["hi"] - iv20["lo"]) > (iv10["hi"] - iv10["lo"])


def test_classify_trend_邊界():
    assert classify_trend(0.7, 0.7, 0.58, 0.42) == "偏多"
    assert classify_trend(0.3, 0.3, 0.58, 0.42) == "偏空"
    assert classify_trend(0.5, 0.5, 0.58, 0.42) == "盤整"
    # 恰在門檻上：>= 偏多、<= 偏空
    assert classify_trend(0.58, 0.58, 0.58, 0.42) == "偏多"
    assert classify_trend(0.42, 0.42, 0.58, 0.42) == "偏空"
