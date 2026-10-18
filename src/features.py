"""特徵工程：所有特徵皆為嚴格 point-in-time。

鐵律：特徵日期 t 的特徵值，只能使用在 t 當時已經公開的資料。
- 價格/量能：t 收盤後已公開（17:30 前），可用 t 當日資料。
- 三大法人（20:00 公布）、融資融券（21:00 公布）：排程為 18:30，
  因此特徵日期 t 只能用到 t-1 的籌碼資料（以 chip_lag_days 控制）。
- 月營收：採保守規則「月份結束後第 10 日」視為可取得。
- 財報：採保守規則「季底後 75 日」視為可取得。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------- 技術指標 ----------

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, min_periods=slow, adjust=False).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


# ---------- PIT 對齊輔助 ----------

def _pit_monthly_revenue(revenue: pd.DataFrame) -> pd.DataFrame:
    """回傳以 availability_date 為鍵的月營收表（含 revenue_yoy）。"""
    if revenue.empty:
        return pd.DataFrame(columns=["availability_date", "revenue_yoy"])
    df = revenue.dropna(subset=["revenue_year", "revenue_month", "revenue"]).copy()
    df["ym"] = (
        df["revenue_year"].astype(int).astype(str)
        + df["revenue_month"].astype(int).astype(str).str.zfill(2)
    )
    df = df.sort_values(["ym", "date"]).drop_duplicates("ym", keep="last")
    # 保守可用日：月份結束後第 10 日
    df["month_end"] = pd.to_datetime(
        df["revenue_year"].astype(int).astype(str)
        + "-"
        + df["revenue_month"].astype(int).astype(str).str.zfill(2)
        + "-01"
    ) + pd.offsets.MonthEnd(0)
    df["availability_date"] = df["month_end"] + pd.Timedelta(days=10)
    df = df.sort_values("availability_date")
    df["revenue_yoy"] = df["revenue"] / df["revenue"].shift(12) - 1
    return df[["availability_date", "revenue_yoy"]].dropna()


def _pit_financials(fin: pd.DataFrame) -> pd.DataFrame:
    """回傳以 availability_date 為鍵的財報特徵（季底後 75 日保守規則）。"""
    if fin.empty:
        return pd.DataFrame(columns=["availability_date", "ni_yoy", "gross_margin"])
    df = fin.copy()
    # date 為季底；同一季同一 type 取最後一筆
    df = df.sort_values(["date", "type"]).drop_duplicates(["date", "type"], keep="last")
    pivot = df.pivot_table(index="date", columns="type", values="value", aggfunc="last")
    pivot = pivot.sort_index()
    out = pd.DataFrame(index=pivot.index)
    out["availability_date"] = pd.to_datetime(pivot.index) + pd.Timedelta(days=75)
    # 稅後淨利年增率（單季比單季；EPS 不直接跨季相加，見陷阱說明）
    ni_col = next((c for c in ["NetIncome", "net_income", "本期淨利"] if c in pivot.columns), None)
    rev_col = next((c for c in ["Revenue", "revenue", "營業收入"] if c in pivot.columns), None)
    gp_col = next((c for c in ["GrossProfit", "gross_profit", "營業毛利"] if c in pivot.columns), None)
    if ni_col:
        ni = pd.to_numeric(pivot[ni_col], errors="coerce")
        out["ni_yoy"] = ni / ni.shift(4) - 1
    if gp_col and rev_col:
        gp = pd.to_numeric(pivot[gp_col], errors="coerce")
        rv = pd.to_numeric(pivot[rev_col], errors="coerce").replace(0, np.nan)
        out["gross_margin"] = gp / rv
    keep = ["availability_date"] + [
        c for c in ["ni_yoy", "gross_margin"] if c in out.columns
    ]
    return out.reset_index(drop=True)[keep]


# ---------- 主函式 ----------

def build_features_for_stock(
    prices: pd.DataFrame,
    inst: pd.DataFrame | None,
    margin: pd.DataFrame | None,
    revenue: pd.DataFrame | None,
    financials: pd.DataFrame | None,
    index_close: pd.Series | None,
    cfg: dict,
) -> pd.DataFrame:
    """為單一股票建立特徵表（index 為 date）。"""
    fcfg = cfg["features"]
    df = prices.sort_values("date").copy()
    df = df.set_index("date")
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    feats = pd.DataFrame(index=df.index)

    # 報酬率
    for n in fcfg["return_windows"]:
        feats[f"ret_{n}"] = close / close.shift(n) - 1
    # 均線乖離與斜率
    for n in fcfg["ma_windows"]:
        ma = close.rolling(n, min_periods=n).mean()
        feats[f"ma_ratio_{n}"] = close / ma - 1
    ma10 = close.rolling(10, min_periods=10).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    feats["ma_slope_10_20"] = ma10 / ma20 - 1
    feats["ma_slope_20_60"] = ma20 / ma60 - 1
    # 量能
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    feats["vol_ratio_20"] = volume / vol_ma20 - 1
    amount = df["amount"].astype(float) if "amount" in df.columns else volume * close
    amt_ma60 = amount.rolling(60, min_periods=60).mean()
    feats["amount_ratio_60"] = amount / amt_ma60 - 1
    # 波動度、回撤、缺口
    logret = np.log(close / close.shift(1))
    for n in fcfg["volatility_windows"]:
        feats[f"volatility_{n}"] = logret.rolling(n, min_periods=n).std() * np.sqrt(252)
    roll_max_60 = close.rolling(60, min_periods=60).max()
    feats["maxdd_60"] = close / roll_max_60 - 1
    feats["gap"] = df["open"].astype(float) / close.shift(1) - 1
    # 技術指標
    feats["rsi_14"] = rsi(close, fcfg["rsi_window"])
    m_line, m_sig, m_hist = macd(close, fcfg["macd_fast"], fcfg["macd_slow"], fcfg["macd_signal"])
    feats["macd_line"] = m_line / close
    feats["macd_hist"] = m_hist / close

    # 大盤相對強弱（報酬指數，含息）
    if index_close is not None and not index_close.empty:
        idx = index_close.sort_index()
        idx_ret20 = idx / idx.shift(20) - 1
        idx_ret60 = idx / idx.shift(60) - 1
        idx_aligned = idx_ret20.reindex(feats.index)
        feats["rel_strength_20"] = feats["ret_20"] - idx_aligned
        feats["rel_strength_60"] = feats["ret_60"] - idx_ret60.reindex(feats.index)

    # 籌碼：三大法人（T-1 落後，反映 18:30 排程時當日資料尚未公布）
    chip_lag = int(cfg["data"].get("chip_lag_days", 1))
    if inst is not None and not inst.empty:
        ins = inst.sort_values("date").set_index("date")
        for col, name in [("foreign_net", "foreign"), ("trust_net", "trust"), ("dealer_net", "dealer")]:
            if col in ins.columns:
                net = pd.to_numeric(ins[col], errors="coerce").fillna(0)
                vol_sum20 = volume.reindex(net.index).fillna(0).rolling(20, min_periods=5).sum()
                net5 = net.rolling(5, min_periods=3).sum().shift(chip_lag)
                net20 = net.rolling(20, min_periods=10).sum().shift(chip_lag)
                denom = vol_sum20.shift(chip_lag).replace(0, np.nan)
                feats[f"inst_{name}_5"] = (net5 / denom).reindex(feats.index)
                feats[f"inst_{name}_20"] = (net20 / denom).reindex(feats.index)
        # 連買/連賣天數（合計）
        if all(c in ins.columns for c in ["foreign_net", "trust_net", "dealer_net"]):
            total = ins[["foreign_net", "trust_net", "dealer_net"]].apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0).sum(axis=1)
            sign = np.sign(total).replace(0, np.nan).ffill().fillna(0)
            streak = sign * (sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1)
            feats["inst_streak"] = streak.shift(chip_lag).reindex(feats.index).clip(-10, 10)

    # 融資融券（同樣 T-1）
    if margin is not None and not margin.empty:
        mg = margin.sort_values("date").set_index("date")
        for bal_col, name in [("MarginPurchaseTodayBalance", "margin"), ("ShortSaleTodayBalance", "short")]:
            if bal_col in mg.columns:
                bal = pd.to_numeric(mg[bal_col], errors="coerce")
                feats[f"{name}_chg_5"] = (bal / bal.shift(5) - 1).shift(chip_lag).reindex(feats.index)
                feats[f"{name}_chg_20"] = (bal / bal.shift(20) - 1).shift(chip_lag).reindex(feats.index)

    # 月營收年增率（PIT：月份結束後第 10 日）
    if revenue is not None and cfg["features"].get("use_monthly_revenue", True):
        rev_pit = _pit_monthly_revenue(revenue)
        if not rev_pit.empty:
            merged = pd.merge_asof(
                pd.DataFrame({"date": feats.index}).sort_values("date"),
                rev_pit.sort_values("availability_date"),
                left_on="date",
                right_on="availability_date",
                direction="backward",
            ).set_index("date")
            feats["revenue_yoy"] = merged["revenue_yoy"]

    # 財報（PIT：季底後 75 日）
    if financials is not None and cfg["features"].get("use_financial_statements", True):
        fin_pit = _pit_financials(financials)
        if not fin_pit.empty:
            merged = pd.merge_asof(
                pd.DataFrame({"date": feats.index}).sort_values("date"),
                fin_pit.sort_values("availability_date"),
                left_on="date",
                right_on="availability_date",
                direction="backward",
            ).set_index("date")
            for c in ["ni_yoy", "gross_margin"]:
                if c in merged.columns:
                    feats[c] = merged[c]

    feats["close"] = close
    feats["volume"] = volume
    return feats


def build_panel(
    prices: dict[str, pd.DataFrame],
    inst: dict[str, pd.DataFrame],
    margin: dict[str, pd.DataFrame],
    revenue: dict[str, pd.DataFrame],
    financials: dict[str, pd.DataFrame],
    index_close: pd.Series | None,
    cfg: dict,
) -> pd.DataFrame:
    """建立全市場特徵 panel，MultiIndex (stock_id, date)。"""
    frames = []
    for sid, px in prices.items():
        if px is None or px.empty:
            continue
        f = build_features_for_stock(
            px,
            inst.get(sid),
            margin.get(sid),
            revenue.get(sid),
            financials.get(sid),
            index_close,
            cfg,
        )
        f["stock_id"] = sid
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames)
    panel = panel.set_index("stock_id", append=True).swaplevel(0, 1).sort_index()
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    exclude = {"close", "volume"}
    return [c for c in panel.columns if c not in exclude]
