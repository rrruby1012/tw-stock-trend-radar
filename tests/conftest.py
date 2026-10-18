"""共用 fixture：全合成市場資料（固定 seed，測試可重跑、互不干擾）。

不呼叫任何真實 API。所有隨機皆以 numpy default_rng(seed) 產生。
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import features as F  # noqa: E402
from src import labels as L  # noqa: E402
from src.config import load_config  # noqa: E402

STOCK_IDS = ["1001", "1002", "1003"]


def make_prices(stock_id, n_days, seed, start="2020-01-02",
                start_price=100.0, drift=0.0005, vol=0.02):
    """合成還原權息日線（幾何布朗運動）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rets = rng.normal(drift, vol, n_days)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.005, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
    volume = rng.integers(50, 500, n_days).astype(float) * 1000  # 股
    amount = close * volume
    return pd.DataFrame({
        "date": dates, "stock_id": str(stock_id),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "amount": amount,
    })


def make_inst(stock_id, dates, seed):
    """合成三大法人買賣超（寬表風格：已是淨額欄位）。"""
    rng = np.random.default_rng(seed)
    n = len(dates)
    return pd.DataFrame({
        "date": pd.DatetimeIndex(dates), "stock_id": str(stock_id),
        "foreign_net": rng.normal(0, 2e6, n),
        "trust_net": rng.normal(0, 5e5, n),
        "dealer_net": rng.normal(0, 3e5, n),
    })


def make_margin(stock_id, dates, seed):
    """合成融資融券餘額（隨機漫步，恆正）。"""
    rng = np.random.default_rng(seed)
    n = len(dates)
    mp = 1e7 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    ss = 2e6 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date": pd.DatetimeIndex(dates), "stock_id": str(stock_id),
        "MarginPurchaseTodayBalance": mp, "ShortSaleTodayBalance": ss,
    })


def make_revenue(stock_id, start_ym="2019-01", n_months=40, seed=7):
    """合成月營收（穩定成長 + 噪音）。"""
    rng = np.random.default_rng(seed)
    months = pd.date_range(start_ym, periods=n_months, freq="MS")
    revenue = 1e8 * (1.004 ** np.arange(n_months)) * (1 + rng.normal(0, 0.02, n_months))
    return pd.DataFrame({
        "date": months + pd.offsets.MonthEnd(0),
        "stock_id": str(stock_id),
        "revenue": revenue,
        "revenue_year": months.year,
        "revenue_month": months.month,
    })


def make_financials(stock_id, start_q="2019-03-31", n_quarters=12, seed=9):
    """合成季財報（long format：NetIncome / Revenue / GrossProfit）。"""
    rng = np.random.default_rng(seed)
    quarters = pd.date_range(start_q, periods=n_quarters, freq="Q")
    rows = []
    for i, q in enumerate(quarters):
        rev = 3e8 * (1.01 ** i) * (1 + rng.normal(0, 0.03))
        rows.append({"date": q, "stock_id": str(stock_id), "type": "NetIncome",
                     "value": rev * 0.12 * (1 + rng.normal(0, 0.05))})
        rows.append({"date": q, "stock_id": str(stock_id), "type": "Revenue", "value": rev})
        rows.append({"date": q, "stock_id": str(stock_id), "type": "GrossProfit",
                     "value": rev * 0.3})
    return pd.DataFrame(rows)


def make_index(dates, seed=11):
    """合成加權報酬指數序列。"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, len(dates))
    return pd.Series(10000 * np.exp(np.cumsum(rets)),
                     index=pd.DatetimeIndex(dates), name="index_close")


def build_test_panel(n_days=300, n_stocks=3, seed=42, cfg=None):
    """一站式合成 panel（含特徵 + 標籤），供多個測試檔使用。"""
    cfg = cfg or load_config()
    sids = STOCK_IDS[:n_stocks]
    prices, inst, margin, revenue, fins = {}, {}, {}, {}, {}
    for i, sid in enumerate(sids):
        px = make_prices(sid, n_days, seed + i)
        prices[sid] = px
        inst[sid] = make_inst(sid, px["date"], seed + 100 + i)
        margin[sid] = make_margin(sid, px["date"], seed + 200 + i)
        revenue[sid] = make_revenue(sid, seed=seed + 300 + i)
        fins[sid] = make_financials(sid, seed=seed + 400 + i)
    index_close = make_index(prices[sids[0]]["date"])
    panel = F.build_panel(prices, inst, margin, revenue, fins, index_close, cfg)
    panel = L.add_labels(panel, index_close, [10, 20])
    return panel, index_close, cfg


@pytest.fixture(scope="session")
def base_cfg():
    """真實 config.yaml（測試與正式環境用同一份參數）。"""
    return load_config()


@pytest.fixture(scope="session")
def small_cfg(base_cfg):
    """加速版設定：給 walk-forward 等較重的測試用。"""
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["validation"]["min_train_days"] = 120
    cfg["model"]["validation"]["refit_every_days"] = 40
    cfg["model"]["validation"]["embargo_days"] = 5
    cfg["model"]["validation"]["n_splits_min"] = 2
    cfg["model"]["calibration"]["min_calibration_samples"] = 30
    cfg["credibility"]["min_oos_samples"] = 100
    return cfg


@pytest.fixture(scope="session")
def panel_small(small_cfg):
    """3 檔 × 300 天合成 panel（含特徵與標籤）。"""
    panel, _, _ = build_test_panel(n_days=300, n_stocks=3, seed=42, cfg=small_cfg)
    return panel


@pytest.fixture(scope="session")
def index_small(panel_small):
    dates = panel_small.index.get_level_values("date").unique()
    return make_index(dates)


@pytest.fixture(scope="session")
def watchlist_small():
    return pd.DataFrame({
        "symbol": STOCK_IDS,
        "name": ["測試鋼鐵", "測試金融", "測試電子"],
    })
