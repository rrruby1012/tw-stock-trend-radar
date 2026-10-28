"""Phase 2 第 3、4 項：chip_lag_days 設定化與財報 PIT 截止規則。

只測本任務範圍：設定值傳進籌碼特徵、Q4 年結後 90 天、Q1-Q3 季結後
45 天。不碰 YoY 計算、營收 dedupe、cache 等其他審計問題。
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
from src.config import load_config  # noqa: E402


# ---------- 共用合成資料 ----------

def _prices(dates):
    n = len(dates)
    return pd.DataFrame({
        "date": pd.DatetimeIndex(dates),
        "open": np.full(n, 100.0),
        "high": np.full(n, 100.0),
        "low": np.full(n, 100.0),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 10000.0),
        "amount": np.full(n, 100.0 * 10000.0),
    })


def _inst_impulse(dates, impulse_idx):
    n = len(dates)
    foreign = np.zeros(n)
    foreign[impulse_idx] = 1e8
    return pd.DataFrame({
        "date": pd.DatetimeIndex(dates),
        "foreign_net": foreign,
        "trust_net": np.zeros(n),
        "dealer_net": np.zeros(n),
    })


def _fin_quarter(quarter_end):
    """單季財報（long format），毛利率固定 0.3。"""
    return pd.DataFrame([
        {"date": pd.Timestamp(quarter_end), "type": "NetIncome", "value": 3e7},
        {"date": pd.Timestamp(quarter_end), "type": "Revenue", "value": 3e8},
        {"date": pd.Timestamp(quarter_end), "type": "GrossProfit", "value": 9e7},
    ])


# ---------- (a) chip_lag_days 設定值確實傳進特徵計算 ----------

def test_config_chip_lag_days_default_is_one():
    """設定檔本身帶 chip_lag_days = 1（與舊行為一致）。"""
    cfg = load_config()
    assert cfg["features"]["chip_lag_days"] == 1


def _chip_feats(lag):
    cfg = copy.deepcopy(load_config())
    cfg["features"]["chip_lag_days"] = lag
    dates = pd.bdate_range("2025-01-02", periods=80)
    feats = F.build_features_for_stock(
        _prices(dates), _inst_impulse(dates, 30), None, None, None, None, cfg
    )
    return feats


def _first_positive_index(series):
    pos = series[series > 0]
    assert not pos.empty, "預期籌碼特徵在脈衝後出現正值"
    return series.index.get_loc(pos.index[0])


def test_chip_lag_value_changes_feature_timing():
    """改設定值會改變籌碼特徵的 lag 行為。

    外資淨額在第 30 個交易日單日脈衝：5 日合計在脈衝日及其後 4 日
    非零，shift(chip_lag) 後第一個正值應落在 30 + lag。
    """
    assert _first_positive_index(_chip_feats(1)["inst_foreign_5"]) == 31
    assert _first_positive_index(_chip_feats(3)["inst_foreign_5"]) == 33
    # lag=1 時脈衝隔日已可用，lag=3 時同日仍不可用
    assert _chip_feats(1)["inst_foreign_5"].iloc[31] > 0
    assert _chip_feats(3)["inst_foreign_5"].iloc[31] == 0


def test_chip_lag_data_section_fallback():
    """相容舊寫法：features 未設時讀 data.chip_lag_days。"""
    cfg = copy.deepcopy(load_config())
    del cfg["features"]["chip_lag_days"]
    cfg["data"]["chip_lag_days"] = 3
    dates = pd.bdate_range("2025-01-02", periods=80)
    feats = F.build_features_for_stock(
        _prices(dates), _inst_impulse(dates, 30), None, None, None, None, cfg
    )
    assert _first_positive_index(feats["inst_foreign_5"]) == 33


def test_chip_lag_missing_everywhere_defaults_to_one():
    """兩處皆未設時預設 lag 1（舊行為）。"""
    cfg = copy.deepcopy(load_config())
    cfg["features"].pop("chip_lag_days", None)
    cfg["data"].pop("chip_lag_days", None)
    dates = pd.bdate_range("2025-01-02", periods=80)
    feats = F.build_features_for_stock(
        _prices(dates), _inst_impulse(dates, 30), None, None, None, None, cfg
    )
    assert _first_positive_index(feats["inst_foreign_5"]) == 31


# ---------- (b)(c) 財報 PIT：Q4 90 天、Q1-Q3 45 天 ----------

def test_pit_financials_q4_availability_90_days():
    """Q4（2024-12-31 年結）可用日 = 年結後 90 天 = 2025-03-31。

    舊固定 75 日規則會給 2025-03-16，提前約兩週——新規則必須更晚。
    """
    out = F._pit_financials(_fin_quarter("2024-12-31"))
    assert out["availability_date"].iloc[0] == pd.Timestamp("2025-03-31")
    assert out["availability_date"].iloc[0] != pd.Timestamp("2025-03-16")


def test_pit_financials_q1_q2_q3_availability_45_days():
    """Q1/Q2/Q3 季結後 45 天（各季法定截止日近似）。"""
    fin = pd.concat([
        _fin_quarter("2025-03-31"),
        _fin_quarter("2025-06-30"),
        _fin_quarter("2025-09-30"),
    ])
    out = F._pit_financials(fin).sort_values("availability_date").reset_index(drop=True)
    assert list(out["availability_date"]) == [
        pd.Timestamp("2025-05-15"),  # Q1: 03-31 + 45 天
        pd.Timestamp("2025-08-14"),  # Q2: 06-30 + 45 天
        pd.Timestamp("2025-11-14"),  # Q3: 09-30 + 45 天
    ]


def test_pit_q4_end_to_end_not_available_before_deadline():
    """端到端：Q4 財報在 2025-03-31 前不可用、當日才可用。"""
    cfg = load_config()
    dates = pd.bdate_range("2025-03-03", periods=40)  # 含 03-28（五）與 03-31（一）
    feats = F.build_features_for_stock(
        _prices(dates), None, None, None, _fin_quarter("2024-12-31"), None, cfg
    )
    assert pd.isna(feats.loc[pd.Timestamp("2025-03-28"), "gross_margin"])
    assert feats.loc[pd.Timestamp("2025-03-31"), "gross_margin"] == pytest.approx(0.3)


def test_pit_q1_end_to_end_not_available_before_deadline():
    """端到端：Q1 財報在 2025-05-15 前不可用、當日才可用。"""
    cfg = load_config()
    dates = pd.bdate_range("2025-05-01", periods=20)  # 含 05-14（三）與 05-15（四）
    feats = F.build_features_for_stock(
        _prices(dates), None, None, None, _fin_quarter("2025-03-31"), None, cfg
    )
    assert pd.isna(feats.loc[pd.Timestamp("2025-05-14"), "gross_margin"])
    assert feats.loc[pd.Timestamp("2025-05-15"), "gross_margin"] == pytest.approx(0.3)


def test_pit_financials_q4_leap_year_floor_is_mar31():
    """閏年 Q4 下限：2027-12-31 年結後 90 日為 2028-03-30，但可用日
    不得早於法定 3/31，故應為 2028-03-31（協調者補充，取兩者較晚者）。"""
    out = F._pit_financials(_fin_quarter("2027-12-31"))
    assert out["availability_date"].iloc[0] == pd.Timestamp("2028-03-31")
