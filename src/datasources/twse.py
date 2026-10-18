"""TWSE OpenAPI 備援客戶端。

用途：FinMind 暫時失敗（維護、配額、逾時）時，抓取當日最新日報表，
並保留前一次正確結果。注意：
- TWSE 日成交資訊為「未還原」價格，除權息日會跳空，僅作備援顯示用，
  不得直接用於報酬計算（或需以除權息表排除事件日）。
- 多數端點僅提供最新交易日，不適合做歷史建庫。
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://openapi.twse.com.tw/v1"


class TWSEClient:
    def __init__(
        self,
        timeout_sec: int = 30,
        max_retries: int = 3,
        retry_backoff_sec: float = 5.0,
        polite_delay_sec: float = 2.0,
    ):
        self.timeout = timeout_sec
        self.max_retries = max_retries
        self.backoff = retry_backoff_sec
        self.delay = polite_delay_sec
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "tw-stock-trend-radar/0.1 (research use)"

    def _get(self, path: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if attempt:
                    time.sleep(self.delay)
                resp = self.session.get(f"{BASE}{path}", timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("TWSE 請求失敗 %s（嘗試 %d）：%s", path, attempt + 1, e)
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2**attempt))
        raise RuntimeError(f"TWSE 請求多次重試仍失敗 {path}：{last_err}")

    @staticmethod
    def _to_frame(payload: dict) -> pd.DataFrame:
        fields = payload.get("fields", [])
        data = payload.get("data", [])
        df = pd.DataFrame(data, columns=fields)
        return df

    @staticmethod
    def _num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(
            s.astype(str).str.replace(",", "", regex=False).str.strip().replace(
                {"--": None, "": None, "X": None}
            ),
            errors="coerce",
        )

    def stock_day_all(self) -> pd.DataFrame:
        """全體上市公司每日收盤行情（最新交易日，未還原）。"""
        df = self._to_frame(self._get("/exchangeReport/STOCK_DAY_ALL"))
        if df.empty:
            return df
        rename = {
            "證券代號": "stock_id",
            "證券名稱": "name",
            "開盤價": "open",
            "最高價": "high",
            "最低價": "low",
            "收盤價": "close",
            "成交股數": "volume",
            "成交金額": "amount",
        }
        df = df.rename(columns=rename)
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df.columns:
                df[c] = self._num(df[c])
        # volume 單位為股
        return df

    def institutional_all(self) -> pd.DataFrame:
        """三大法人買賣超日報（最新交易日，逐檔）。端點 ID 請以 swagger 核對。"""
        df = self._to_frame(self._get("/exchangeReport/TWT48U"))
        return df

    def margin_all(self) -> pd.DataFrame:
        """融資融券彙總（最新交易日，逐檔）。端點 ID 請以 swagger 核對。"""
        df = self._to_frame(self._get("/exchangeReport/MI_MARGN"))
        return df
