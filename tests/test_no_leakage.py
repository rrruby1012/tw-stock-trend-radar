"""禁止答案欄位進入模型特徵（目標洩漏防護）。

背景：2026-10-19 發現 feature_columns() 誤將標籤欄
(up_10/up_20/exc_10/exc_20/fwd_ret_10/fwd_ret_20) 當成特徵，
導致最終預測 50 檔機率全為 1.0。此測試確保此類迴歸不再發生。
"""
import re
import pandas as pd

from src import features as F


# 答案欄位的命名模式：任何看起來像「未來答案」的欄位都不該進特徵
ANSWER_PATTERNS = [
    re.compile(r"^(up|exc|fwd_ret)_\d+$"),   # up_10, exc_20, fwd_ret_10 …
    re.compile(r"^future_return"),            # future_return*
    re.compile(r"^target$"),                  # target
    re.compile(r"^label$"),                   # label
    re.compile(r"^actual_"),                   # actual_up_10, actual_ret_10 …
]


def _assert_no_answer_cols(cols, where):
    bad = [c for c in cols
           if any(p.match(c) for p in ANSWER_PATTERNS)]
    assert not bad, f"{where} 混入答案欄位：{bad}"


def test_feature_columns_排除所有答案欄位():
    panel = pd.DataFrame({
        "ret_10": [0.01], "close": [100.0], "volume": [1000.0],
        "close_raw": [100.0],
        # 標籤欄
        "up_10": [1.0], "up_20": [0.0], "exc_10": [1.0], "exc_20": [0.0],
        "fwd_ret_10": [0.05], "fwd_ret_20": [-0.02],
        # 其他可能的答案欄位命名
        "actual_up_10": [1.0], "actual_up_20": [0.0],
        "future_return_10": [0.05], "target": [1.0], "label": [0.0],
    })
    cols = F.feature_columns(panel)
    _assert_no_answer_cols(cols, "feature_columns()")
    # 正常特徵必須保留
    assert "ret_10" in cols
    # 顯示用欄位也不進模型
    assert "close_raw" not in cols
    assert "close" not in cols and "volume" not in cols


def test_is_label_column_涵蓋所有標籤():
    for c in ["up_10", "up_20", "exc_10", "exc_20",
              "fwd_ret_10", "fwd_ret_20", "up_100"]:
        assert F._is_label_column(c), f"{c} 應被判為標籤欄"
    for c in ["ret_10", "ma_ratio_60", "close", "volume",
              "revenue_yoy", "update", "exciting"]:
        assert not F._is_label_column(c), f"{c} 不應被判為標籤欄"


def test_預測機率不應全為極端值():
    """迴歸測試：若特徵含標籤，最終預測會全擠向 0/1。
    此測試用合成資料驗證：標籤欄被排除後，模型輸出應有變異。"""
    import numpy as np
    from src.models import LogisticModel

    rng = np.random.RandomState(42)
    n = 2000
    # 真實特徵：與標籤弱相關
    x1 = rng.randn(n)
    y = (x1 + rng.randn(n) * 2 > 0).astype(float)
    # 標籤欄（洩漏源）：若誤用，模型會完全依賴它
    leak = y.copy()

    X_clean = pd.DataFrame({"x1": x1})
    X_leaky = pd.DataFrame({"x1": x1, "up_10": leak})

    m_clean = LogisticModel().fit(X_clean, y)
    m_leaky = LogisticModel().fit(X_leaky, y)

    # 預測時標籤未知 → 以中位數填補（模擬真實預測情境）
    X_pred_clean = pd.DataFrame({"x1": rng.randn(50)})
    X_pred_leaky = pd.DataFrame({"x1": rng.randn(50),
                                 "up_10": [np.median(leak)] * 50})

    p_clean = m_clean.predict_proba(X_pred_clean)
    p_leaky = m_leaky.predict_proba(X_pred_leaky)
    # 乾淨模型的預測應有變異；洩漏模型的預測會擠向極端
    assert np.std(p_clean) > 0.01, "乾淨特徵的預測應有變異"
    # 洩漏模型的係數會異常大（學到答案）
    coef_leaky = dict(zip(X_leaky.columns, m_leaky.coefficients_))
    assert abs(coef_leaky["up_10"]) > 1.0, \
        "洩漏測試前置條件：標籤欄係數應異常大"
