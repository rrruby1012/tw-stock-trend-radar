"""回測測試：交易成本、最低手續費、基準行為、空資料處理。"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import _trade_net_return, run_backtest


def _cost_kwargs(cfg):
    c = cfg["costs"]
    return dict(
        commission_rate=c["commission_rate"],
        commission_discount=c["commission_discount"],
        commission_min=c["commission_min_ntd"],
        tax_rate=c["securities_tax_rate"],
        slippage_bps=c["slippage_bps"],
    )


def test_零毛報酬時淨報酬為負_成本確實被扣除(base_cfg):
    """毛報酬為 0 的交易，扣完手續費+稅+滑價後淨報酬必須為負。"""
    net = _trade_net_return(100_000, 0.0, **_cost_kwargs(base_cfg))
    assert net < 0, f"零毛報酬交易淨報酬應為負，實際 {net}"


def test_最低手續費20元有生效(base_cfg):
    """小額交易（手續費不足 20 元）應按 20 元計，報酬比無下限時更差。"""
    kw = _cost_kwargs(base_cfg)
    assert 1000 * kw["commission_rate"] * kw["commission_discount"] < 20
    with_min = _trade_net_return(1000, 0.05, **kw)
    kw_no_min = dict(kw, commission_min=0.0)
    without_min = _trade_net_return(1000, 0.05, **kw_no_min)
    assert with_min < without_min, "最低手續費應使小額交易報酬更低"


def test_手續費折扣有作用(base_cfg):
    kw = _cost_kwargs(base_cfg)
    full = _trade_net_return(100_000, 0.05, **dict(kw, commission_discount=1.0))
    disc = _trade_net_return(100_000, 0.05, **kw)
    assert disc > full


def _fake_oos_and_panel(n_stocks=4, n_days=60, seed=0):
    """合成 oos 預測 + 含 fwd_ret_10 的 panel。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    sids = [f"9{i:03d}" for i in range(n_stocks)]
    rows = []
    idx_rows = []
    for i, sid in enumerate(sids):
        for j, d in enumerate(dates):
            proba = float(rng.uniform(0.3, 0.7))
            rows.append({"stock_id": sid, "date": d,
                         "y_true": float(rng.integers(0, 2)), "proba": proba})
            idx_rows.append((sid, d, float(rng.normal(0.002, 0.02))))
    oos_df = pd.DataFrame(rows)
    panel = pd.DataFrame(
        {"fwd_ret_10": [r[2] for r in idx_rows]},
        index=pd.MultiIndex.from_tuples(
            [(r[0], r[1]) for r in idx_rows], names=["stock_id", "date"]),
    )
    index_close = pd.Series(10000 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n_days))),
                            index=dates)
    return oos_df, panel, index_close


def test_buy_hold_index_不含交易成本(base_cfg):
    """buy_hold 基準報酬必須等於指數本身報酬（docstring 已揭露不含成本）。"""
    oos_df, panel, index_close = _fake_oos_and_panel()
    oos = {"targets": {"up_10": {"oos": oos_df}}, "credible": True}
    res = run_backtest(oos, panel, index_close, base_cfg, credible=True)
    bh = res["buy_hold_index"]
    assert bh is not None
    reb_dates = pd.DatetimeIndex(sorted(oos_df["date"].unique()))[::10]
    expected = float(index_close.reindex(reb_dates, method="ffill").iloc[-1] /
                     index_close.reindex(reb_dates, method="ffill").iloc[0] - 1)
    assert bh["total_return"] == pytest.approx(expected, rel=1e-9)


def test_回測輸出結構完整(base_cfg):
    oos_df, panel, index_close = _fake_oos_and_panel()
    oos = {"targets": {"up_10": {"oos": oos_df}}, "credible": True}
    res = run_backtest(oos, panel, index_close, base_cfg, credible=True)
    for key in ["strategy", "buy_hold_index", "always_long"]:
        assert res[key] is not None, key
        for mk in ["total_return", "annualized", "max_drawdown", "n_trades", "hit_rate"]:
            assert mk in res[key], f"{key}.{mk}"
    assert res["strategy"]["max_drawdown"] <= 0, "最大回撤應為負值或零"
    assert res["strategy"]["n_trades"] > 0


def test_空oos回傳warning而非報錯(base_cfg):
    oos = {"targets": {"up_10": {"oos": pd.DataFrame()}}, "credible": False}
    panel = pd.DataFrame({"fwd_ret_10": []})
    res = run_backtest(oos, panel, None, base_cfg, credible=False)
    assert "warning" in res
    assert res["strategy"] is None


def test_不可信模型回測加註警語(base_cfg):
    oos_df, panel, index_close = _fake_oos_and_panel()
    oos = {"targets": {"up_10": {"oos": oos_df}}, "credible": False}
    res = run_backtest(oos, panel, index_close, base_cfg, credible=False)
    assert "warning" in res and "可信度" in res["warning"]
