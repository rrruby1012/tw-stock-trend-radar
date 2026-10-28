"""調整資料不可用時不得靜默改用未還原價（Phase 2 第 2 項）。

- 除權息表抓取失敗 → stock_price_adj 拋出 AdjustmentDataUnavailable，不再回傳未還原價。
- fetch_all 捕捉後將該檔列入 excluded_adjustment（附原因），且該檔不在 price_adj 內；
  其餘股票照常處理；metadata 不再為缺席者背書。
- QuotaExceeded 仍直接 re-raise（使用者定奪的 main 行為）。
全部 mock，不打真實 API。
"""
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from src.datasources.finmind import (
    AdjustmentDataUnavailable,
    FinMindClient,
    QuotaExceeded,
)
from src.fetch import fetch_all


def _adj_frame(sid):
    dates = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame({
        "date": dates,
        "stock_id": [sid] * 5,
        "open": [100.0] * 5,
        "high": [102.0] * 5,
        "low": [99.0] * 5,
        "close": [101.0] * 5,
        "volume": [10000.0] * 5,
        "amount": [1_010_000.0] * 5,
        "adjusted": [True] * 5,
    })


class _FakeClient:
    """依 sid 決定行為：ok / adj_fail（調整資料不可用）/ quota（配額用盡）。"""

    authenticated = False

    def __init__(self, behavior):
        self.behavior = behavior

    def stock_price_adj(self, sid, start, end):
        b = self.behavior.get(sid, "ok")
        if b == "adj_fail":
            raise AdjustmentDataUnavailable(f"{sid} 除權息表抓取失敗，無法還原權息：boom")
        if b == "quota":
            raise QuotaExceeded("402 Quota Exceeded")
        return _adj_frame(sid)

    def institutional_wide(self, sid, start, end):
        return pd.DataFrame()

    def margin(self, sid, start, end):
        return pd.DataFrame()

    def month_revenue(self, sid, start, end):
        return pd.DataFrame()

    def financial_statements(self, sid, start, end):
        return pd.DataFrame()

    def taiex_total_return_index(self, start, end):
        return pd.DataFrame()


def _cfg(tmp_path):
    return {"data": {"cache_dir": str(tmp_path)}}


def _run_fetch(tmp_path, behavior, symbols, incremental=False):
    with patch("src.fetch.FinMindClient", return_value=_FakeClient(behavior)):
        return fetch_all(
            _cfg(tmp_path), symbols,
            date(2024, 1, 2), date(2024, 1, 10),
            cache_dir=str(tmp_path), incremental=incremental,
        )


def test_調整失敗時該檔被排除且不混入原始價(tmp_path):
    result = _run_fetch(tmp_path, {"1001": "ok", "1002": "adj_fail"}, ["1001", "1002"])

    # 失敗檔不在 price_adj 內：沒有任何原始價能混進還原序列
    assert "1002" not in result["datasets"]["price_adj"]
    assert "1001" in result["datasets"]["price_adj"]
    # 排除紀錄附原因
    assert "1002" in result["excluded_adjustment"]
    assert "調整資料不可用" in result["excluded_adjustment"]["1002"]
    assert any("1002" in w and "調整資料不可用" in w for w in result["warnings"])
    # adjusted_price=True 只描述已納入者（皆為還原價），不再為缺席者背書
    assert result["adjusted_price"] is True
    assert result["datasets"]["price_adj"]["1001"]["adjusted"].all()
    assert result["data_dates"]["price_adj"] is not None


def test_調整失敗時不使用舊快取續命(tmp_path):
    # 先給 1002 種一筆舊的還原快取；調整失敗時仍須排除，不得拿舊快取續命
    cache_dir = tmp_path / "price_adj"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _adj_frame("1002").to_parquet(cache_dir / "1002.parquet", index=False)

    result = _run_fetch(
        tmp_path, {"1001": "ok", "1002": "adj_fail"}, ["1001", "1002"],
        incremental=True,
    )
    assert "1002" not in result["datasets"]["price_adj"]
    assert "1002" in result["excluded_adjustment"]


def test_調整成功路徑不變(tmp_path):
    result = _run_fetch(tmp_path, {"1001": "ok", "1002": "ok"}, ["1001", "1002"])

    assert set(result["datasets"]["price_adj"]) == {"1001", "1002"}
    assert result["excluded_adjustment"] == {}
    assert not any("調整資料不可用" in w for w in result["warnings"])


def test_除權息表抓取失敗時client拋出可辨識例外():
    client = FinMindClient(polite_delay_sec=0, retry_backoff_sec=0)
    price = _adj_frame("2330").drop(columns=["adjusted"])
    with patch.object(client, "stock_price", return_value=price), \
            patch.object(client, "dividend_result", side_effect=RuntimeError("boom")):
        with pytest.raises(AdjustmentDataUnavailable):
            client.stock_price_adj("2330", "2024-01-02", "2024-01-10")


def test_除權息表為空時不拋出且原樣回傳():
    # 除權息表成功回傳但無事件＝無需調整，與「抓取失敗」是兩回事，不得混為一談
    client = FinMindClient(polite_delay_sec=0, retry_backoff_sec=0)
    price = _adj_frame("2330").drop(columns=["adjusted"])
    with patch.object(client, "stock_price", return_value=price), \
            patch.object(client, "dividend_result",
                          return_value=pd.DataFrame(columns=["date", "stock_id"])):
        adj = client.stock_price_adj("2330", "2024-01-02", "2024-01-10")
    pd.testing.assert_frame_equal(adj.reset_index(drop=True), price.reset_index(drop=True))


def test_配額用盡仍直接re_raise(tmp_path):
    with patch("src.fetch.FinMindClient",
               return_value=_FakeClient({"1001": "quota", "1002": "ok"})):
        with pytest.raises(QuotaExceeded):
            fetch_all(
                _cfg(tmp_path), ["1001", "1002"],
                date(2024, 1, 2), date(2024, 1, 10),
                cache_dir=str(tmp_path), incremental=False,
            )
