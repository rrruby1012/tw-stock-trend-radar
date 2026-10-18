"""標籤建立：未來報酬方向標籤（僅用於訓練/驗證，絕不用作特徵）。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_labels(
    panel: pd.DataFrame,
    index_close: pd.Series | None,
    horizons: list[int] = (10, 20),
) -> pd.DataFrame:
    """為 panel 加上各 horizon 的標籤。

    - up_{h}：h 個交易日後報酬率 > 0
    - exc_{h}：h 個交易日後報酬率 > 加權報酬指數同期報酬率
    - fwd_ret_{h}：h 個交易日後報酬率（用於價格區間與回測）
    """
    df = panel.copy()
    close = df["close"]

    def fwd_ret(h: int) -> pd.Series:
        return close.groupby(level="stock_id").shift(-h) / close - 1

    for h in horizons:
        fr = fwd_ret(h)
        df[f"fwd_ret_{h}"] = fr
        # 注意：(NaN > 0) 在 pandas 會得 False，若直接 astype(float)
        # 會把「未知」誤標為 0（下跌）。必須把未來報酬未知的列保持為 NaN，
        # 下游 walk-forward 會以 dropna 捨棄這些列。
        up = pd.Series(np.where(fr > 0, 1.0, 0.0), index=fr.index)
        up[fr.isna()] = np.nan
        df[f"up_{h}"] = up
        if index_close is not None and not index_close.empty:
            idx = index_close.sort_index()
            idx_fwd = idx.shift(-h) / idx - 1
            # 對齊到每個 (stock_id, date)
            dates = df.index.get_level_values("date")
            idx_vals = idx_fwd.reindex(dates).to_numpy()
            fr_vals = fr.to_numpy()
            exc = pd.Series(np.where(fr_vals > idx_vals, 1.0, 0.0), index=fr.index)
            exc[pd.isna(fr_vals) | pd.isna(idx_vals)] = np.nan
            df[f"exc_{h}"] = exc
    return df
