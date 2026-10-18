"""分析宇宙（universe）管理：載入 watchlist、套用排除規則。"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_watchlist(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"symbol": str})
    df["symbol"] = df["symbol"].str.strip().str.zfill(4)
    df = df.dropna(subset=["symbol"]).drop_duplicates("symbol")
    return df.reset_index(drop=True)


def apply_exclusion_rules(
    watchlist: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    min_history_days: int = 500,
    max_missing_ratio: float = 0.05,
    min_avg_turnover: float = 5_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """套用排除規則，回傳 (納入名單, 排除名單含原因)。

    排除：資料不足、長期停止交易（缺資料比例過高）、流動性過低。
    """
    included, excluded = [], []
    for _, row in watchlist.iterrows():
        sid = row["symbol"]
        df = prices.get(sid)
        reason = None
        n_days = 0 if df is None or df.empty else len(df)
        if n_days < min_history_days:
            reason = f"資料不足（{n_days} 天 < {min_history_days} 天）"
        else:
            # 缺資料比例：以日期連續性估算（交易日曆由上游提供時可更精確）
            expected = n_days  # 保守：以實際取得天數為基準，另檢視成交量為 0 的天數
            zero_vol = int((df["volume"] <= 0).sum())
            missing_ratio = zero_vol / max(n_days, 1)
            avg_turnover = float(df.tail(60)["amount"].mean())
            if missing_ratio > max_missing_ratio:
                reason = f"長期停止交易嫌疑（無量天數比例 {missing_ratio:.1%}）"
            elif avg_turnover < min_avg_turnover:
                reason = f"流動性過低（近60日平均成交金額 {avg_turnover:,.0f} 元）"
        if reason:
            excluded.append({**row.to_dict(), "排除原因": reason, "資料天數": n_days})
            log.info("排除 %s：%s", sid, reason)
        else:
            included.append(row.to_dict())
    return (
        pd.DataFrame(included).reset_index(drop=True),
        pd.DataFrame(excluded).reset_index(drop=True),
    )
