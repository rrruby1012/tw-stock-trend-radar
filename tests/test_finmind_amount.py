"""還原權息時 amount 不得被調整（Phase 2 第 1 項）。

回歸測試：stock_price_adj 的向後還原只應調整 open/high/low/close，
amount（成交金額）與 volume 必須維持原始值；close_raw 仍為官方原始收盤價。
全部 mock，不打真實 API。
"""
from unittest.mock import patch

import pandas as pd
import pytest

from src.datasources.finmind import FinMindClient


def _client():
    return FinMindClient(polite_delay_sec=0, retry_backoff_sec=0)


def _price_df():
    # 除權息日為 2024-06-28；前兩日應被調整，當日及之後不調整。
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-06-26", "2024-06-27", "2024-06-28", "2024-06-29"]),
        "stock_id": ["2330"] * 4,
        "open": [100.0, 101.0, 90.0, 91.0],
        "high": [102.0, 103.0, 92.0, 93.0],
        "low": [99.0, 100.0, 89.0, 90.0],
        "close": [101.0, 102.0, 91.0, 92.0],
        "volume": [10000.0, 10000.0, 10000.0, 10000.0],
        "amount": [1_010_000.0, 1_020_000.0, 910_000.0, 920_000.0],
    })


def _div_df():
    # before_price 100 -> reference_price 90，調整因子 0.9
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-06-28"]),
        "stock_id": ["2330"],
        "before_price": [100.0],
        "reference_price": [90.0],
    })


def _run_adj(price_df, div_df):
    client = _client()
    with patch.object(client, "stock_price", return_value=price_df), \
            patch.object(client, "dividend_result", return_value=div_df):
        return client.stock_price_adj("2330", "2024-06-26", "2024-06-29")


def test_還原調整後amount完全不變而OHLC正確調整():
    price = _price_df()
    adj = _run_adj(price, _div_df())

    # amount：調整前後完全相同（逐列相等，非近似）
    pd.testing.assert_series_equal(
        adj["amount"].reset_index(drop=True),
        price["amount"].reset_index(drop=True),
        check_names=False,
    )
    # volume 同樣不應被調整（原程式本就未調整，此處鎖定不回歸）
    pd.testing.assert_series_equal(
        adj["volume"].reset_index(drop=True),
        price["volume"].reset_index(drop=True),
        check_names=False,
    )
    # 除權息日前兩日 OHLC 乘上 0.9
    assert adj["close"].tolist() == pytest.approx(
        [101.0 * 0.9, 102.0 * 0.9, 91.0, 92.0])
    assert adj["open"].tolist() == pytest.approx(
        [100.0 * 0.9, 101.0 * 0.9, 90.0, 91.0])
    assert adj["high"].tolist() == pytest.approx(
        [102.0 * 0.9, 103.0 * 0.9, 92.0, 93.0])
    assert adj["low"].tolist() == pytest.approx(
        [99.0 * 0.9, 100.0 * 0.9, 89.0, 90.0])
    # close_raw 維持官方原始收盤價，不動
    assert adj["close_raw"].tolist() == pytest.approx([101.0, 102.0, 91.0, 92.0])
    assert adj["adjusted"].all()


def test_多個除權息事件累乘時amount仍完全不變():
    price = _price_df()
    div = pd.DataFrame({
        "date": pd.to_datetime(["2024-06-28", "2024-06-29"]),
        "stock_id": ["2330", "2330"],
        "before_price": [100.0, 90.0],
        # 因子：0.9（06-28）與 80/90 ≈ 0.8889（06-29）；06-26/27 應乘 0.8
        "reference_price": [90.0, 80.0],
    })
    adj = _run_adj(price, div)

    pd.testing.assert_series_equal(
        adj["amount"].reset_index(drop=True),
        price["amount"].reset_index(drop=True),
        check_names=False,
    )
    # 06-26/27：0.9 * (80/90) = 0.8；06-28：僅 80/90；06-29：不調整
    assert adj["close"].tolist() == pytest.approx(
        [101.0 * 0.8, 102.0 * 0.8, 91.0 * (80.0 / 90.0), 92.0])
    assert adj["close_raw"].tolist() == pytest.approx([101.0, 102.0, 91.0, 92.0])


def test_無除權息事件時原樣回傳且amount不變():
    price = _price_df()
    adj = _run_adj(price, pd.DataFrame(columns=["date", "stock_id"]))
    # 無事件時 stock_price_adj 直接回傳未調整的 price（無 close_raw/adjusted 欄位亦屬原行為）
    pd.testing.assert_series_equal(
        adj["amount"].reset_index(drop=True),
        price["amount"].reset_index(drop=True),
        check_names=False,
    )
    assert adj["close"].tolist() == pytest.approx([101.0, 102.0, 91.0, 92.0])
