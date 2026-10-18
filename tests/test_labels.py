"""標籤測試：未來報酬方向標籤的正確性（標籤用未來資料是正常的，僅限訓練用）。"""
import numpy as np
import pandas as pd
import pytest

from src import labels as L


def _panel_from_close(close_values, stock_id="1001", start="2024-01-02"):
    dates = pd.bdate_range(start, periods=len(close_values))
    close = pd.Series(close_values, index=dates, dtype=float)
    panel = pd.DataFrame({"close": close, "volume": 1000.0})
    panel.index.name = "date"  # 日期層級必須命名為 date（與正式 panel 一致）
    panel["stock_id"] = stock_id
    panel = panel.set_index("stock_id", append=True).swaplevel(0, 1).sort_index()
    return panel


def test_up_10_標籤等於10日後是否上漲():
    """手動 30 天序列：第 1 天收 100、第 11 天收 90 → up_10 應為 0；
    第 2 天收 101、第 12 天收 111 → up_10 應為 1。"""
    close = [100.0 + i for i in range(30)]
    close[10] = 90.0  # 第 11 天暴跌
    panel = _panel_from_close(close)
    labeled = L.add_labels(panel, None, [10])
    day1 = labeled.loc[("1001", labeled.index.get_level_values("date")[0])]
    day2 = labeled.loc[("1001", labeled.index.get_level_values("date")[1])]
    assert day1["up_10"] == 0.0
    assert day1["fwd_ret_10"] == pytest.approx(90.0 / 100.0 - 1)
    assert day2["up_10"] == 1.0


def test_尾端標籤為NaN不應硬填():
    """最後 10 天沒有未來報酬，up_10 必須為 NaN（不得用當日報酬冒充）。"""
    close = [100.0 + i for i in range(30)]
    panel = _panel_from_close(close)
    labeled = L.add_labels(panel, None, [10])
    last = labeled.loc[("1001", labeled.index.get_level_values("date")[-1])]
    assert pd.isna(last["up_10"])
    assert pd.isna(last["fwd_ret_10"])


def test_exc_10_股票漲指數跌時為1():
    """股票未來 10 日上漲、加權指數同期下跌 → exc_10 = 1。"""
    close = [100.0 + i for i in range(30)]  # 單調上漲
    panel = _panel_from_close(close)
    dates = panel.index.get_level_values("date").unique()
    index_close = pd.Series([1000.0 - i for i in range(30)], index=dates)  # 單調下跌
    labeled = L.add_labels(panel, index_close, [10])
    day1 = labeled.loc[("1001", dates[0])]
    assert day1["exc_10"] == 1.0


def test_exc_10_股票跌指數漲時為0():
    close = [100.0 - i for i in range(30)]  # 單調下跌
    panel = _panel_from_close(close)
    dates = panel.index.get_level_values("date").unique()
    index_close = pd.Series([1000.0 + i for i in range(30)], index=dates)  # 單調上漲
    labeled = L.add_labels(panel, index_close, [10])
    day1 = labeled.loc[("1001", dates[0])]
    assert day1["exc_10"] == 0.0
