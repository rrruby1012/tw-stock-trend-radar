"""FinMind 資料來源客戶端（主力資料來源）。

- 還原權息日線、三大法人、融資融券、月營收、財報、股利、加權報酬指數、交易日曆
- token 只從環境變數 FINMIND_TOKEN 讀取；無 token 時降級為匿名（300 次/小時）
- 配額用盡時 FinMind 回 HTTP 402（非 429），此處特別處理
- 遵守服務條款：合理節流、指數退避；僅做盤後研究用途
"""
from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd
import requests

log = logging.getLogger(__name__)

API_BASE = "https://api.finmindtrade.com/api/v4/data"


class FinMindError(RuntimeError):
    pass


class QuotaExceeded(FinMindError):
    """FinMind 配額用盡（HTTP 402）。"""


class FinMindClient:
    def __init__(
        self,
        token: str = "",
        timeout_sec: int = 30,
        max_retries: int = 3,
        retry_backoff_sec: float = 5.0,
        polite_delay_sec: float = 1.0,
    ):
        self.token = token or ""
        self.timeout = timeout_sec
        self.max_retries = max_retries
        self.backoff = retry_backoff_sec
        self.delay = polite_delay_sec
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"
        self.session.headers["User-Agent"] = "tw-stock-trend-radar/0.1 (research use)"
        self._n_calls = 0

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    def _request(self, params: dict) -> list:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                if self._n_calls and self.delay:
                    time.sleep(self.delay)
                resp = self.session.get(API_BASE, params=params, timeout=self.timeout)
                self._n_calls += 1
                if resp.status_code == 402:
                    raise QuotaExceeded(
                        "FinMind API 配額用盡（HTTP 402）。請稍後重試或提高方案。"
                    )
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("status") not in (200, "200"):
                    raise FinMindError(f"FinMind API 錯誤：{payload.get('msg')}")
                return payload.get("data", [])
            except QuotaExceeded:
                raise
            except Exception as e:  # noqa: BLE001 - 重試各類暫時性錯誤
                last_err = e
                log.warning("FinMind 請求失敗（嘗試 %d）：%s", attempt + 1, e)
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2**attempt))
        raise FinMindError(f"FinMind 請求多次重試仍失敗：{last_err}")

    def fetch(
        self,
        dataset: str,
        data_id: str | None = None,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
    ) -> pd.DataFrame:
        """通用查詢，回傳 DataFrame（空資料回傳空表，不報錯）。"""
        params: dict = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start_date:
            params["start_date"] = str(start_date)
        if end_date:
            params["end_date"] = str(end_date)
        rows = self._request(params)
        df = pd.DataFrame(rows)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df

    # ---------- 各資料集便利方法 ----------

    def stock_price(
        self, stock_id: str, start_date: str | date, end_date: str | date
    ) -> pd.DataFrame:
        """未還原日線（TaiwanStockPrice）。注意：除權息日會跳空，不可直接算報酬率。"""
        df = self.fetch("TaiwanStockPrice", stock_id, start_date, end_date)
        if df.empty:
            return df
        rename = {
            "max": "high",
            "min": "low",
            "Trading_Volume": "volume",
            "Trading_money": "amount",
            "Trading_turnover": "turnover",
        }
        df = df.rename(columns=rename)
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["stock_id"] = df["stock_id"].astype(str)
        # 無成交價之日開高低收存為 0，以 close > 0 過濾
        df = df[df["close"] > 0].copy()
        return df.sort_values("date").reset_index(drop=True)

    def dividend_result(
        self, stock_id: str, start_date: str | date, end_date: str | date
    ) -> pd.DataFrame:
        """除權息結果表（含除權息參考價，用於自行還原權息）。"""
        df = self.fetch("TaiwanStockDividendResult", stock_id, start_date, end_date)
        if df.empty:
            return df
        df["stock_id"] = df["stock_id"].astype(str)
        return df.sort_values("date").reset_index(drop=True)

    def stock_price_adj(
        self, stock_id: str, start_date: str | date, end_date: str | date
    ) -> pd.DataFrame:
        """還原權息日線（主力價格來源）。

        FinMind 的 TaiwanStockPriceAdj 需付費等級，免費方案無法使用；
        此處改以免費的 TaiwanStockPrice（未還原）＋ TaiwanStockDividendResult
        （除權息參考價）自行做「向後還原」：
        除權息日 D 的調整因子 r = reference_price / before_price，
        D 之前所有交易日的 OHLC 乘上 r（多事件累乘）。
        """
        price = self.stock_price(stock_id, start_date, end_date)
        if price.empty:
            return price
        try:
            div = self.dividend_result(stock_id, start_date, end_date)
        except Exception as e:  # noqa: BLE001
            log.warning("%s 除權息表抓取失敗，改用未還原價：%s", stock_id, e)
            return price

        factors = []
        for _, r in div.iterrows():
            try:
                before = float(r["before_price"])
                ref = float(r["reference_price"])
            except (KeyError, TypeError, ValueError):
                continue
            if before > 0 and ref > 0 and ref != before:
                factors.append((pd.to_datetime(r["date"]), ref / before))
        if not factors:
            return price

        f = pd.Series(
            [ratio for _, ratio in factors],
            index=pd.DatetimeIndex([d for d, _ in factors]),
        ).sort_index()
        # 由後往前累乘：cum[i] = f[i] * f[i+1] * ...
        cum = f.iloc[::-1].cumprod().iloc[::-1]
        price_dates = pd.to_datetime(price["date"])
        # 每個價格日 t：取第一個「除權息日 > t」的累乘值；之後無事件則為 1
        pos = cum.index.searchsorted(price_dates, side="right")
        ratios = pd.Series(1.0, index=price.index)
        mask = pos < len(cum)
        ratios.loc[mask] = cum.values[pos[mask]]

        adj = price.copy()
        for col in ["open", "high", "low", "close", "amount"]:
            if col in adj.columns:
                adj[col] = adj[col] * ratios.values
        adj["adjusted"] = True
        n_events = len(factors)
        log.debug("%s 還原權息：%d 個除權息事件", stock_id, n_events)
        return adj

    def institutional_wide(
        self, stock_id: str, start_date: str | date, end_date: str | date
    ) -> pd.DataFrame:
        """三大法人買賣超（寬表，一檔一日一列）。"""
        df = self.fetch(
            "TaiwanStockInstitutionalInvestorsBuySellWide",
            stock_id,
            start_date,
            end_date,
        )
        if df.empty:
            return df
        df["stock_id"] = df["stock_id"].astype(str)

        def net(prefix: str) -> pd.Series:
            buy_candidates = [c for c in df.columns if c.startswith(prefix) and "Buy" in c]
            sell_candidates = [c for c in df.columns if c.startswith(prefix) and "Sell" in c]
            buy = df[buy_candidates].apply(pd.to_numeric, errors="coerce").sum(axis=1) if buy_candidates else 0
            sell = df[sell_candidates].apply(pd.to_numeric, errors="coerce").sum(axis=1) if sell_candidates else 0
            return buy - sell

        # 外資 / 投信 / 自營商（含 2014-12-01 前後欄位切分：Dealer、Dealer_self、Dealer_Hedging）
        df["foreign_net"] = net("Foreign_Investor")
        df["trust_net"] = net("Investment_Trust")
        dealer_cols = [c for c in df.columns if "Dealer" in c and ("Buy" in c or "Sell" in c)]
        if dealer_cols:
            buy = df[[c for c in dealer_cols if "Buy" in c]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            sell = df[[c for c in dealer_cols if "Sell" in c]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            df["dealer_net"] = buy - sell
        else:
            df["dealer_net"] = 0.0
        return df.sort_values("date").reset_index(drop=True)

    def margin(
        self, stock_id: str, start_date: str | date, end_date: str | date
    ) -> pd.DataFrame:
        """融資融券餘額與變化。"""
        df = self.fetch(
            "TaiwanStockMarginPurchaseShortSale", stock_id, start_date, end_date
        )
        if df.empty:
            return df
        df["stock_id"] = df["stock_id"].astype(str)
        for c in ["MarginPurchaseTodayBalance", "ShortSaleTodayBalance"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    def month_revenue(
        self, stock_id: str, start_date: str | date, end_date: str | date
    ) -> pd.DataFrame:
        """月營收。注意：create_time 非公告時間，PIT 對齊由 features.py 以保守規則處理。"""
        df = self.fetch("TaiwanStockMonthRevenue", stock_id, start_date, end_date)
        if df.empty:
            return df
        df["stock_id"] = df["stock_id"].astype(str)
        df["revenue"] = pd.to_numeric(df.get("revenue"), errors="coerce")
        df["revenue_year"] = pd.to_numeric(df.get("revenue_year"), errors="coerce")
        df["revenue_month"] = pd.to_numeric(df.get("revenue_month"), errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    def financial_statements(
        self, stock_id: str, start_date: str | date, end_date: str | date
    ) -> pd.DataFrame:
        """綜合損益表（long format：date/stock_id/type/value）。"""
        df = self.fetch(
            "TaiwanStockFinancialStatements", stock_id, start_date, end_date
        )
        if df.empty:
            return df
        df["stock_id"] = df["stock_id"].astype(str)
        df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
        return df.sort_values("date").reset_index(drop=True)

    def taiex_total_return_index(
        self, start_date: str | date, end_date: str | date
    ) -> pd.DataFrame:
        """加權報酬指數（data_id=TAIEX，含息，跑贏大盤比較基準）。"""
        df = self.fetch("TaiwanStockTotalReturnIndex", "TAIEX", start_date, end_date)
        if df.empty:
            return df
        # 防禦性欄位處理：找出指數值欄位
        value_col = next(
            (c for c in ["index", "TAIEX", "value", "close"] if c in df.columns), None
        )
        if value_col is None:
            # 取第一個數值欄位（排除 date/data_id）
            num_cols = [
                c
                for c in df.columns
                if c not in ("date", "data_id") and pd.api.types.is_numeric_dtype(df[c])
            ]
            value_col = num_cols[0] if num_cols else None
        if value_col is None:
            raise FinMindError(f"無法辨識報酬指數欄位：{list(df.columns)}")
        out = df[["date"]].copy()
        out["index_close"] = pd.to_numeric(df[value_col], errors="coerce")
        return out.sort_values("date").reset_index(drop=True)

    def trading_dates(self, start_date: str | date, end_date: str | date) -> pd.DataFrame:
        """台股交易日曆。"""
        return self.fetch("TaiwanStockTradingDate", None, start_date, end_date)

    def stock_info(self) -> pd.DataFrame:
        """台股總覽（代號、名稱、產業、市場別；取每檔最新 date 列）。"""
        df = self.fetch("TaiwanStockInfo")
        if df.empty:
            return df
        df["stock_id"] = df["stock_id"].astype(str)
        df = df.sort_values("date").drop_duplicates("stock_id", keep="last")
        return df.reset_index(drop=True)
