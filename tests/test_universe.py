"""Universe 測試：排除規則（資料不足、流動性過低）。"""
import numpy as np
import pandas as pd

from src.universe import apply_exclusion_rules, load_watchlist
from .conftest import make_prices


def _watchlist():
    return pd.DataFrame({
        "symbol": ["1001", "1002", "1003"],
        "name": ["正常股", "資料不足股", "低流動股"],
    })


def test_資料不足的股票被排除並附原因():
    prices = {
        "1001": make_prices("1001", 600, seed=1),
        "1002": make_prices("1002", 100, seed=2),  # 僅 100 天 < 500 天門檻
        "1003": make_prices("1003", 600, seed=3),
    }
    included, excluded = apply_exclusion_rules(_watchlist(), prices)
    assert "1001" in included["symbol"].values
    assert "1003" in included["symbol"].values
    assert "1002" in excluded["symbol"].values
    reason = excluded.loc[excluded["symbol"] == "1002", "排除原因"].iloc[0]
    assert "資料不足" in reason


def test_流動性過低的股票被排除():
    px = make_prices("1003", 600, seed=3)
    px["volume"] = 100.0  # 每日僅 100 股
    px["amount"] = px["close"] * px["volume"]  # 遠低於 500 萬門檻
    prices = {
        "1001": make_prices("1001", 600, seed=1),
        "1002": make_prices("1002", 600, seed=2),
        "1003": px,
    }
    included, excluded = apply_exclusion_rules(_watchlist(), prices)
    assert "1003" in excluded["symbol"].values
    reason = excluded.loc[excluded["symbol"] == "1003", "排除原因"].iloc[0]
    assert "流動性" in reason


def test_長期無量視為停止交易嫌疑():
    px = make_prices("1002", 600, seed=2)
    px.loc[px.index[300:], "volume"] = 0.0  # 後半段完全無量
    prices = {"1001": make_prices("1001", 600, seed=1),
              "1002": px, "1003": make_prices("1003", 600, seed=3)}
    included, excluded = apply_exclusion_rules(_watchlist(), prices)
    assert "1002" in excluded["symbol"].values


def test_load_watchlist_代號補零與去重(tmp_path):
    p = tmp_path / "watchlist.csv"
    p.write_text("symbol,name\n2330,台積電\n2330,台積電\n5,測試\n", encoding="utf-8")
    df = load_watchlist(p)
    assert df["symbol"].tolist() == ["2330", "0005"]
