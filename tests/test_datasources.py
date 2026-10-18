"""資料來源測試：錯誤處理、重試、欄位合併（全部 mock，不打真實 API）。"""
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from src.datasources.finmind import FinMindClient, FinMindError, QuotaExceeded


def _client(**kw):
    kw.setdefault("polite_delay_sec", 0)
    kw.setdefault("retry_backoff_sec", 0)
    return FinMindClient(**kw)


def test_HTTP402時拋出QuotaExceeded():
    client = _client(max_retries=0)
    mock_resp = Mock(status_code=402)
    with patch.object(client.session, "get", return_value=mock_resp):
        with pytest.raises(QuotaExceeded):
            client.fetch("TaiwanStockPriceAdj", "2330", "2024-01-01", "2024-01-02")


def test_超時會重試指定次數後拋出FinMindError():
    client = _client(max_retries=2)
    with patch.object(client.session, "get",
                      side_effect=requests.Timeout("timeout")) as m:
        with pytest.raises(FinMindError):
            client.fetch("TaiwanStockPriceAdj", "2330", "2024-01-01", "2024-01-02")
        assert m.call_count == 3, "應為 1 次初試 + 2 次重試"


def test_HTTP500會重試後拋出FinMindError():
    client = _client(max_retries=1)
    mock_resp = Mock(status_code=500)
    mock_resp.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    with patch.object(client.session, "get", return_value=mock_resp) as m:
        with pytest.raises(FinMindError):
            client.fetch("TaiwanStockPriceAdj", "2330", "2024-01-01", "2024-01-02")
        assert m.call_count == 2


def test_成功時回傳DataFrame且date為日期型別():
    client = _client()
    mock_resp = Mock(status_code=200)
    mock_resp.json.return_value = {
        "status": 200,
        "data": [{"date": "2024-01-02", "stock_id": "2330", "close": "100.5"}],
    }
    with patch.object(client.session, "get", return_value=mock_resp):
        df = client.fetch("TaiwanStockPriceAdj", "2330", "2024-01-01", "2024-01-02")
    assert len(df) == 1
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def _wide_rows(style):
    base = {"date": "2024-01-02", "stock_id": "2330",
            "Foreign_Investor_Buy": 1000, "Foreign_Investor_Sell": 400,
            "Investment_Trust_Buy": 200, "Investment_Trust_Sell": 100}
    if style == "old":
        base.update({"Dealer_Buy": 100, "Dealer_Sell": 40})
    else:  # 2014-12-01 後：自營商拆分為自行 + 避險
        base.update({"Dealer_self_Buy": 100, "Dealer_self_Sell": 40,
                     "Dealer_Hedging_Buy": 10, "Dealer_Hedging_Sell": 5})
    return [base]


def test_institutional_wide_舊制Dealer欄位合併正確():
    client = _client()
    with patch.object(client, "_request", return_value=_wide_rows("old")):
        df = client.institutional_wide("2330", "2024-01-01", "2024-01-02")
    assert df["foreign_net"].iloc[0] == 600
    assert df["trust_net"].iloc[0] == 100
    assert df["dealer_net"].iloc[0] == 60


def test_institutional_wide_新制Dealer自行加避險合併正確():
    client = _client()
    with patch.object(client, "_request", return_value=_wide_rows("new")):
        df = client.institutional_wide("2330", "2024-01-01", "2024-01-02")
    # (100-40) + (10-5) = 65：自行與避險都要加總，否則序列會斷裂
    assert df["dealer_net"].iloc[0] == 65


def test_空資料回傳空表不報錯():
    client = _client()
    with patch.object(client, "_request", return_value=[]):
        df = client.institutional_wide("2330", "2024-01-01", "2024-01-02")
    assert df.empty
