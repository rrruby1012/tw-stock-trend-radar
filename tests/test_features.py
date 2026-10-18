"""特徵工程測試：無 look-ahead、PIT 對齊、技術指標合理性。"""
import numpy as np
import pandas as pd
import pytest

from src import features as F
from .conftest import make_prices, make_revenue


def test_無_look_ahead_未來資料不影響歷史特徵值(base_cfg):
    """同一股票：用 500 天算 vs 用 600 天（含未來 100 天）算，
    前 500 天的特徵值必須完全一致。"""
    px = make_prices("1001", 600, seed=1)
    f_short = F.build_features_for_stock(px.iloc[:500], None, None, None, None, None, base_cfg)
    f_long = F.build_features_for_stock(px.iloc[:600], None, None, None, None, None, base_cfg)
    pd.testing.assert_frame_equal(f_short, f_long.loc[f_short.index])


def test_無_look_ahead_含籌碼與營收時亦成立(base_cfg):
    """加入三大法人（T-1 落後）與月營收後，仍無 look-ahead。"""
    from .conftest import make_inst, make_margin
    px = make_prices("1001", 600, seed=2)
    inst = make_inst("1001", px["date"], seed=20)
    margin = make_margin("1001", px["date"], seed=21)
    revenue = make_revenue("1001", seed=22)
    kwargs = dict(inst=inst, margin=margin, revenue=revenue, financials=None,
                  index_close=None, cfg=base_cfg)
    f_short = F.build_features_for_stock(px.iloc[:500], **kwargs)
    # 注意：inst/margin 也要截斷，模擬「當時可取得」的資料邊界
    kwargs2 = dict(kwargs, inst=inst.iloc[:500], margin=margin.iloc[:500])
    f_long = F.build_features_for_stock(px.iloc[:600], **kwargs2)
    pd.testing.assert_frame_equal(f_short, f_long.loc[f_short.index])


def test_月營收_PIT_月底加10日之前為NaN(base_cfg):
    """13 個月營收中只有最後一個月有 revenue_yoy；
    在「月底+10日」之前該值必須為 NaN，之後才出現。"""
    # 月份：2023-01 ~ 2024-01（13 個月）；最後一個月可用日 = 2024-02-10
    revenue = pd.DataFrame({
        "date": pd.date_range("2023-01-31", periods=13, freq="M"),
        "stock_id": "1001",
        "revenue": 1e8 * (1.01 ** np.arange(13)),
        "revenue_year": list(range(2023, 2024)) * 0 + [2023] * 12 + [2024],
        "revenue_month": list(range(1, 13)) + [1],
    })
    px = make_prices("1001", 90, seed=3, start="2024-01-02")
    feats = F.build_features_for_stock(px, None, None, revenue, None, None, base_cfg)
    before = feats.loc["2024-02-09", "revenue_yoy"]
    after = feats.loc["2024-02-12", "revenue_yoy"]
    assert pd.isna(before), f"可用日前應為 NaN，實際為 {before}"
    assert not pd.isna(after), "可用日後應有值"
    # 年增率 = 1.01^12 - 1（第 13 個月 vs 第 1 個月）
    assert after == pytest.approx(1.01 ** 12 - 1, rel=1e-6)


def test_RSI_介於0到100之間(base_cfg):
    px = make_prices("1001", 300, seed=4)
    feats = F.build_features_for_stock(px, None, None, None, None, None, base_cfg)
    rsi = feats["rsi_14"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_RSI_常數價格序列為50(base_cfg):
    """無漲跌時 RSI 應為中性 50，而非 NaN 或極端值。"""
    px = make_prices("1001", 100, seed=5)
    px["close"] = 100.0
    px["open"] = 100.0
    feats = F.build_features_for_stock(px, None, None, None, None, None, base_cfg)
    vals = feats["rsi_14"].dropna().to_numpy()
    assert np.allclose(vals, 50.0), f"常數序列 RSI 應為 50，實際 {vals[:5]}"


def test_MACD_常數價格序列趨近於0():
    close = pd.Series([100.0] * 100)
    line, sig, hist = F.macd(close)
    assert (line.dropna().abs() < 1e-6).all()
    assert (hist.dropna().abs() < 1e-6).all()


def test_特徵欄位不包含標籤欄():
    """feature_columns() 必須排除 close/volume 以外的標籤欄（防呆）。"""
    panel = pd.DataFrame({"ret_10": [1.0], "close": [2.0], "volume": [3.0],
                          "up_10": [1.0], "fwd_ret_10": [0.1]})
    cols = F.feature_columns(panel)
    assert "close" not in cols and "volume" not in cols
    assert "ret_10" in cols
