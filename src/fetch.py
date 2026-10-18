"""資料抓取編排：FinMind 主力、TWSE OpenAPI 備援，parquet 快取增量更新。"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .datasources import FinMindClient, QuotaExceeded, TWSEClient

log = logging.getLogger(__name__)

DATASETS = [
    "price_adj",
    "institutional",
    "margin",
    "revenue",
    "financials",
]

# 每個邏輯資料集對應的抓取函式
def _fetch_dataset(client: FinMindClient, dataset: str, sid: str, start, end) -> pd.DataFrame:
    if dataset == "price_adj":
        return client.stock_price_adj(sid, start, end)
    if dataset == "institutional":
        return client.institutional_wide(sid, start, end)
    if dataset == "margin":
        return client.margin(sid, start, end)
    if dataset == "revenue":
        return client.month_revenue(sid, start, end)
    if dataset == "financials":
        return client.financial_statements(sid, start, end)
    raise ValueError(f"未知資料集：{dataset}")


def _cache_path(cache_dir: Path, dataset: str, sid: str) -> Path:
    return cache_dir / dataset / f"{sid}.parquet"


def load_cached(cache_dir: Path, dataset: str, sid: str) -> pd.DataFrame:
    p = _cache_path(cache_dir, dataset, sid)
    if p.exists():
        try:
            return pd.read_parquet(p)
        except Exception as e:  # noqa: BLE001
            log.warning("快取讀取失敗 %s：%s，將重新抓取", p, e)
    return pd.DataFrame()


def save_cached(cache_dir: Path, dataset: str, sid: str, df: pd.DataFrame) -> None:
    p = _cache_path(cache_dir, dataset, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def fetch_all(
    cfg: dict,
    symbols: list[str],
    start_date: date,
    end_date: date,
    cache_dir: str | Path | None = None,
    incremental: bool = True,
) -> dict:
    """抓取全部資料，回傳 dict（含各資料集 dict[stock_id, DataFrame]、加權指數、詮釋資料）。

    - 增量更新：有快取時只抓新區間（往前重疊 5 天避免斷裂）。
    - 失敗策略：單一股票/資料集失敗只記錄警告，不中斷整體；全部失敗才報錯。
    """
    dcfg = cfg["data"]
    cache = Path(cache_dir or dcfg.get("cache_dir", "data/cache"))
    client = FinMindClient(
        token=dcfg.get("finmind_token", ""),
        timeout_sec=dcfg.get("request_timeout_sec", 30),
        max_retries=dcfg.get("max_retries", 3),
        retry_backoff_sec=dcfg.get("retry_backoff_sec", 5.0),
    )
    log.info("FinMind 連線：%s", "已登入 token" if client.authenticated else "匿名（300次/小時）")

    data: dict[str, dict[str, pd.DataFrame]] = {ds: {} for ds in DATASETS}
    data_dates: dict[str, str | None] = {}
    warnings: list[str] = []
    failed = 0

    for ds in DATASETS:
        max_date = None
        for sid in symbols:
            cached = load_cached(cache, ds, sid) if incremental else pd.DataFrame()
            if not cached.empty and "date" in cached.columns:
                cached["date"] = pd.to_datetime(cached["date"])
                fetch_start = (cached["date"].max() - timedelta(days=5)).date()
            else:
                cached = pd.DataFrame()
                fetch_start = start_date
            if fetch_start > end_date:
                df = cached
            else:
                try:
                    fresh = _fetch_dataset(client, ds, sid, fetch_start, end_date)
                except QuotaExceeded:
                    raise
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    warnings.append(f"{sid}/{ds} 抓取失敗：{e}")
                    log.warning("%s/%s 抓取失敗：%s", sid, ds, e)
                    fresh = pd.DataFrame()
                if fresh.empty:
                    df = cached
                elif cached.empty:
                    df = fresh
                else:
                    df = pd.concat([cached, fresh], ignore_index=True)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
            if not df.empty:
                save_cached(cache, ds, sid, df)
                data[ds][sid] = df
                dmax = pd.to_datetime(df["date"]).max()
                if max_date is None or dmax > max_date:
                    max_date = dmax
        data_dates[ds] = max_date.strftime("%Y-%m-%d") if max_date is not None else None
        log.info("%s：%d 檔有資料，最新 %s", ds, len(data[ds]), data_dates[ds])

    # 加權報酬指數（比較基準）
    try:
        index_df = client.taiex_total_return_index(start_date, end_date)
        index_close = (
            index_df.set_index("date")["index_close"].sort_index()
            if not index_df.empty
            else pd.Series(dtype=float)
        )
        data_dates["index"] = (
            index_close.index.max().strftime("%Y-%m-%d") if not index_close.empty else None
        )
    except Exception as e:  # noqa: BLE001
        warnings.append(f"加權報酬指數抓取失敗：{e}")
        index_close = pd.Series(dtype=float)
        data_dates["index"] = None

    if not data["price_adj"]:
        raise RuntimeError("所有股票的還原股價皆抓取失敗，無法繼續。" + "；".join(warnings[:3]))

    return {
        "datasets": data,
        "index_close": index_close,
        "data_dates": data_dates,
        "warnings": warnings,
        "source": "FinMind",
        "adjusted_price": True,
    }


def fetch_twse_fallback(symbols: list[str]) -> dict:
    """TWSE OpenAPI 備援：僅抓當日最新日報表（未還原價，僅供顯示/比對）。"""
    client = TWSEClient()
    out: dict = {"source": "TWSE OpenAPI（備援）", "adjusted_price": False}
    try:
        out["stock_day_all"] = client.stock_day_all()
    except Exception as e:  # noqa: BLE001
        log.warning("TWSE 備援抓取失敗：%s", e)
        out["stock_day_all"] = pd.DataFrame()
    return out
