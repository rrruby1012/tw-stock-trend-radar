"""虛擬交易回測（僅供研究與教育用途，不構成任何交易建議）。

以 up_10 目標的樣本外預測做虛擬交易：
- 每 10 個交易日再平衡一次；每次選取當日預測機率前 30% 的股票
  （至少 3 檔、至多 10 檔），持有 10 個交易日，等權配置。
- 以 panel 中的 fwd_ret_10（未來 10 日報酬率）計算毛報酬，
  再扣除台股實際交易成本：手續費、證券交易稅、滑價。
- 比較基準：(1) 持有加權報酬指數（buy_hold_index，不含交易成本）；
  (2) 等權持有全部宇宙股票、相同再平衡頻率（always_long，含相同成本）。

注意：回測使用的是「樣本外預測」，且選股邏輯只用到當時已知的
預測機率，沒有使用未來資料。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import get

log = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def _trade_net_return(
    allocation: float,
    fwd_ret: float,
    commission_rate: float,
    commission_discount: float,
    commission_min: float,
    tax_rate: float,
    slippage_bps: float,
) -> float:
    """單筆交易的淨報酬率（含買賣雙邊成本）。

    - 買進：手續費 = max(配置金額 × 費率 × 折扣, 最低 20 元)；滑價 = 配置金額 × bps
    - 賣出：手續費同上（以賣出金額計）；證交稅 = 賣出金額 × 0.3%；滑價同上
    """
    slip_rate = slippage_bps / 10_000.0
    buy_comm = max(allocation * commission_rate * commission_discount, commission_min)
    buy_slip = allocation * slip_rate
    proceeds = allocation * (1.0 + fwd_ret)
    sell_comm = max(proceeds * commission_rate * commission_discount, commission_min)
    sell_slip = proceeds * slip_rate
    sell_tax = proceeds * tax_rate
    cost_basis = allocation + buy_comm + buy_slip
    end_value = proceeds - sell_comm - sell_slip - sell_tax
    return end_value / cost_basis - 1.0


def _equity_stats(equity: list[list], n_trades: int, n_profitable: int,
                  span_trading_days: int) -> dict:
    """由權益曲線計算總報酬、年化、最大回撤、勝率。"""
    values = np.array([v for _, v in equity], dtype=float)
    total_return = float(values[-1] / values[0] - 1.0) if len(values) else 0.0
    if span_trading_days > 0 and values[-1] > 0:
        annualized = float((values[-1] / values[0]) ** (TRADING_DAYS_PER_YEAR / span_trading_days) - 1.0)
    else:
        annualized = 0.0
    peak = np.maximum.accumulate(values)
    drawdown = np.where(peak > 0, values / peak - 1.0, 0.0)
    max_dd = float(np.min(drawdown)) if len(drawdown) else 0.0
    return {
        "equity": equity,
        "total_return": total_return,
        "annualized": annualized,
        "max_drawdown": max_dd,  # 負值表示回撤幅度
        "n_trades": int(n_trades),
        "hit_rate": float(n_profitable / n_trades) if n_trades else None,
    }


def run_backtest(
    oos: dict,
    panel: pd.DataFrame,
    index_close: pd.Series | None,
    cfg: dict,
    credible: bool | None = None,
) -> dict:
    """執行虛擬交易回測。

    參數：
        oos: walk_forward() 的回傳 dict（取 targets["up_10"]["oos"]）。
        panel: 含 fwd_ret_10 的特徵 panel。
        index_close: 加權報酬指數序列（buy_hold 基準用）。
        cfg: 中央設定。
        credible: 模型是否通過可信度檢驗（由呼叫方傳入；None 則嘗試從 oos 讀取）。
    """
    ccfg = get(cfg, "costs", default={}) or {}
    commission_rate = float(ccfg.get("commission_rate", 0.001425))
    commission_discount = float(ccfg.get("commission_discount", 0.6))
    commission_min = float(ccfg.get("commission_min_ntd", 20))
    tax_rate = float(ccfg.get("securities_tax_rate", 0.003))
    slippage_bps = float(ccfg.get("slippage_bps", 10))
    notional = 1_000_000.0  # 虛擬組合規模（新台幣元）

    if credible is None:
        credible = bool(oos.get("credible", False))

    target_oos = oos.get("targets", {}).get("up_10", {})
    oos_df: pd.DataFrame = target_oos.get("oos")
    empty_result = {
        "strategy": None, "buy_hold_index": None, "always_long": None,
        "credible": credible,
        "warning": "樣本外預測為空，無法回測。",
    }
    if oos_df is None or oos_df.empty or "fwd_ret_10" not in panel.columns:
        log.warning("回測略過：樣本外預測為空或 panel 缺少 fwd_ret_10。")
        return empty_result

    oos_df = oos_df.copy()
    oos_df["date"] = pd.to_datetime(oos_df["date"])
    reb_dates = pd.DatetimeIndex(sorted(oos_df["date"].unique()))[::10]
    if len(reb_dates) < 2:
        log.warning("再平衡日期不足（%d），無法回測。", len(reb_dates))
        return {**empty_result, "warning": "再平衡期數不足。"}

    fwd = panel["fwd_ret_10"]
    cost_kwargs = dict(
        commission_rate=commission_rate,
        commission_discount=commission_discount,
        commission_min=commission_min,
        tax_rate=tax_rate,
        slippage_bps=slippage_bps,
    )

    def _trades_for(date, proba_cutoff: float | None) -> list[float]:
        """回傳某再平衡日、依選擇規則選出的各檔淨報酬率。"""
        day = oos_df[oos_df["date"] == date].dropna(subset=["proba"])
        if day.empty:
            return []
        if proba_cutoff is None:
            # always_long：持有全部
            selected = day
        else:
            day = day.sort_values("proba", ascending=False)
            k = min(max(int(len(day) * 0.30), 3), 10)
            selected = day.head(k)
        rets = []
        alloc = notional / max(len(selected), 1)
        for _, row in selected.iterrows():
            try:
                fr = float(fwd.loc[(row["stock_id"], row["date"])])
            except KeyError:
                continue
            if not np.isfinite(fr):
                continue
            rets.append(_trade_net_return(alloc, fr, **cost_kwargs))
        return rets

    # --- 策略：預測機率前 30% ---
    strat_equity = [[str(reb_dates[0].date()), 1.0]]
    strat_trades: list[float] = []
    # --- always_long：等權持有全部 ---
    long_equity = [[str(reb_dates[0].date()), 1.0]]
    long_trades: list[float] = []

    for d in reb_dates[1:]:
        for store, eq, cutoff in (
            (strat_trades, strat_equity, 0.0),
            (long_trades, long_equity, None),
        ):
            rets = _trades_for(d, cutoff)
            period_ret = float(np.mean(rets)) if rets else 0.0
            store.extend(rets)
            eq.append([str(d.date()), eq[-1][1] * (1.0 + period_ret)])

    # --- buy_hold_index：持有加權報酬指數（不含交易成本，僅為比較基準）---
    bh_equity = None
    if index_close is not None and not index_close.empty:
        idx = index_close.sort_index()
        idx_at = idx.reindex(reb_dates, method="ffill")
        if idx_at.notna().all() and (idx_at.iloc[0] or 0) != 0:
            base = float(idx_at.iloc[0])
            bh_equity = [
                [str(d.date()), float(v) / base] for d, v in zip(reb_dates, idx_at)
            ]
        else:
            log.warning("加權報酬指數在再平衡日缺值，略過 buy_hold 基準。")

    span_days = (len(reb_dates) - 1) * 10
    strategy = _equity_stats(
        strat_equity,
        n_trades=len(strat_trades),
        n_profitable=int(np.sum(np.array(strat_trades) > 0)),
        span_trading_days=span_days,
    )
    always_long = _equity_stats(
        long_equity,
        n_trades=len(long_trades),
        n_profitable=int(np.sum(np.array(long_trades) > 0)),
        span_trading_days=span_days,
    )
    buy_hold = (
        _equity_stats(bh_equity, n_trades=1, n_profitable=0, span_trading_days=span_days)
        if bh_equity
        else None
    )
    if buy_hold is not None:
        # buy-and-hold 只有一筆「交易」，勝率不具意義
        buy_hold["hit_rate"] = None

    result = {
        "strategy": strategy,
        "buy_hold_index": buy_hold,
        "always_long": always_long,
        "credible": credible,
        "params": {
            "rebalance_every_days": 10,
            "selection": "預測機率前 30%（3~10 檔）",
            "notional_ntd": notional,
            "costs": {
                "commission_rate": commission_rate,
                "commission_discount": commission_discount,
                "commission_min_ntd": commission_min,
                "securities_tax_rate": tax_rate,
                "slippage_bps": slippage_bps,
            },
        },
    }
    if not credible:
        result["warning"] = (
            "模型未通過可信度檢驗，此處策略績效僅供參考，"
            "不得視為有效的預測能力證據。"
        )
    log.info(
        "回測完成：策略總報酬 %.2f%%（年化 %.2f%%，最大回撤 %.2f%%，交易 %d 筆，勝率 %.2f）",
        strategy["total_return"] * 100,
        strategy["annualized"] * 100,
        strategy["max_drawdown"] * 100,
        strategy["n_trades"],
        (strategy["hit_rate"] or 0) * 100,
    )
    return result
