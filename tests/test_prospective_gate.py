"""前瞻門檻接線測試（2026-11-11；鎖定規格第 7 條）。

驗證：
- 區塊計數：日期排序後貪婪切分，每 20 筆一區塊，餘數捨去。
- 門檻判定：120 日且 6 區塊才放行；119 日即使統計通過仍 false；
  不足 6 區塊仍 false；前瞻達標但統計失敗／輸出退化仍 false；
  只有統計 ∧ 健康 ∧ 前瞻三者皆通過才放行。
- 保守規則：進度讀取失敗（None）或序列為空時一律不放行。
- 來源過濾：prospective_stats 只計 ledger 中 data_class=prospective 且
  published_at >= start_at 的凍結紀錄；重建與 walk-forward 模擬永不計入。
- 門檻來自 config prospective 區段，不得寫死。

全部使用合成日期／合成 ledger（tmp_path），不讀寫 repo 的真實 ledger。
"""
import json
from datetime import date, timedelta
from pathlib import Path

from src.ledger import count_nonoverlap_blocks, prospective_stats
from src.validation import decide_credibility, evaluate_prospective_gate

# 與正式 config 相同的門檻（測試另以自訂 config 證明門檻來自設定、非寫死）。
CFG = {"prospective": {"min_published_days": 120, "min_nonoverlap_blocks_20d": 6}}
START_AT = "2026-10-22T16:23:27Z"


def _dates(n, start=date(2026, 10, 27)):
    """產生 n 個連續曆日的日期字串（僅供門檻計數測試）。

    區塊計數只看「序列筆數」（排序後貪婪切分），不驗日曆間隔，因此以連續
    曆日充當交易日序列已足夠驗證計數與門檻邏輯；真實交易日的間隔問題
    屬於 ledger 計數口徑，見 count_nonoverlap_blocks 的註解。
    """
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def _gate(dates, cfg=None):
    return evaluate_prospective_gate(cfg or CFG, dates)


def test_區塊計數每20筆一區塊餘數捨去():
    assert count_nonoverlap_blocks([]) == 0
    assert count_nonoverlap_blocks(_dates(19)) == 0
    assert count_nonoverlap_blocks(_dates(20)) == 1
    assert count_nonoverlap_blocks(_dates(39)) == 1
    assert count_nonoverlap_blocks(_dates(40)) == 2
    assert count_nonoverlap_blocks(_dates(119)) == 5
    assert count_nonoverlap_blocks(_dates(120)) == 6
    assert count_nonoverlap_blocks(_dates(125)) == 6
    assert count_nonoverlap_blocks(_dates(140)) == 7
    # 重複日期只計一次
    assert count_nonoverlap_blocks(_dates(20) + _dates(20)) == 1
    # 未排序的輸入仍正確計數
    assert count_nonoverlap_blocks(list(reversed(_dates(40)))) == 2


def test_119日即使統計通過仍不放行():
    gate = _gate(_dates(119))
    assert gate["pass"] is False
    assert gate["published_days"] == 119
    assert gate["nonoverlap_blocks_20d"] == 5
    credible, reasons, status = decide_credibility(
        stats_credible=True, stats_reasons=["統計通過"],
        stats_status="ok", degenerate=False, degenerate_cause=None, gate=gate)
    assert credible is False
    assert status == "failed"
    assert any("119" in r and "120" in r for r in reasons)


def test_不足6區塊仍不放行():
    gate = _gate(_dates(100))  # 100 日＝5 區塊
    assert gate["pass"] is False
    assert gate["nonoverlap_blocks_20d"] == 5
    credible, _, _ = decide_credibility(
        stats_credible=True, stats_reasons=["統計通過"],
        stats_status="ok", degenerate=False, degenerate_cause=None, gate=gate)
    assert credible is False


def test_120日6區塊且統計通過才放行():
    gate = _gate(_dates(120))
    assert gate["pass"] is True
    assert gate["published_days"] == 120
    assert gate["nonoverlap_blocks_20d"] == 6
    credible, reasons, status = decide_credibility(
        stats_credible=True, stats_reasons=["統計通過"],
        stats_status="ok", degenerate=False, degenerate_cause=None, gate=gate)
    assert credible is True
    assert status == "ok"
    assert reasons == ["統計通過"]


def test_前瞻達標但統計失敗仍不放行():
    gate = _gate(_dates(120))
    assert gate["pass"] is True
    credible, reasons, status = decide_credibility(
        stats_credible=False, stats_reasons=["AUC 未達門檻"],
        stats_status="failed", degenerate=False, degenerate_cause=None, gate=gate)
    assert credible is False
    assert status == "failed"
    assert "AUC 未達門檻" in reasons


def test_前瞻達標但輸出退化仍不放行():
    gate = _gate(_dates(120))
    assert gate["pass"] is True
    credible, reasons, status = decide_credibility(
        stats_credible=True, stats_reasons=["統計通過"],
        stats_status="ok", degenerate=True,
        degenerate_cause="calibration_saturation", gate=gate)
    assert credible is False
    assert status == "failed"
    assert any("退化" in r for r in reasons)


def test_進度讀取失敗採保守不放行():
    gate = _gate(None)
    assert gate["pass"] is False
    assert any("讀取失敗" in r for r in gate["reasons"])
    credible, _, _ = decide_credibility(
        stats_credible=True, stats_reasons=["統計通過"],
        stats_status="ok", degenerate=False, degenerate_cause=None, gate=gate)
    assert credible is False


def test_序列為空採保守不放行():
    gate = _gate([])
    assert gate["pass"] is False
    credible, _, _ = decide_credibility(
        stats_credible=True, stats_reasons=["統計通過"],
        stats_status="ok", degenerate=False, degenerate_cause=None, gate=gate)
    assert credible is False


def test_統計資料不足時狀態維持insufficient():
    gate = _gate(_dates(10))
    credible, _, status = decide_credibility(
        stats_credible=False, stats_reasons=["樣本不足"],
        stats_status="insufficient", degenerate=False,
        degenerate_cause=None, gate=gate)
    assert credible is False
    assert status == "insufficient"


def test_門檻來自config而非寫死():
    small_cfg = {"prospective": {"min_published_days": 2,
                                 "min_nonoverlap_blocks_20d": 1}}
    gate = _gate(_dates(20), cfg=small_cfg)  # 20 日＝1 區塊
    assert gate["pass"] is True
    assert gate["min_published_days"] == 2
    assert gate["min_nonoverlap_blocks_20d"] == 1
    gate2 = _gate(_dates(19), cfg=small_cfg)  # 19 日＝0 區塊
    assert gate2["pass"] is False


def _write_record(pub_dir, date_str, data_class, published_at):
    rec = {"data_class": data_class, "data_cutoff_date": date_str,
           "prediction_sha256": "x" * 64, "published_at": published_at}
    (pub_dir / f"{date_str}.json").write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def test_prospective_stats只計前瞻凍結紀錄(tmp_path):
    """合成 ledger：prospective 40 筆（含 1 筆 start_at 之前）、
    reconstructed 20 筆、walkforward_sim 20 筆；只應計入 40 筆 prospective
    中 published_at >= start_at 的 39 筆（39 日＝1 區塊）。"""
    ledger_dir = tmp_path / "ledger"
    pub_dir = ledger_dir / "published"
    pub_dir.mkdir(parents=True)
    index = []

    def add(d, cls, pub):
        _write_record(pub_dir, d, cls, pub)
        index.append({"date": d, "data_cutoff_date": d,
                      "prediction_sha256": "x" * 64, "published_at": pub,
                      "data_class": cls})

    base = date(2026, 10, 27)
    # 40 筆 prospective：第一筆 published_at 在 start_at 之前（不計），其餘 39 筆計入
    for i in range(40):
        d = (base + timedelta(days=i)).isoformat()
        pub = ("2026-10-22T10:00:00+08:00" if i == 0
               else "2026-10-27T19:00:00+08:00")
        add(d, "prospective", pub)
    # 20 筆 reconstructed 與 20 筆 walkforward_sim：皆不計入
    for i in range(20):
        d = (base + timedelta(days=100 + i)).isoformat()
        add(d, "reconstructed", "2026-11-01T19:00:00+08:00")
    for i in range(20):
        d = (base + timedelta(days=200 + i)).isoformat()
        add(d, "walkforward_sim", "2026-11-01T19:00:00+08:00")
    (ledger_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8")

    stats = prospective_stats(ledger_dir, START_AT)
    assert stats["published_days"] == 39
    assert stats["nonoverlap_blocks_20d"] == 1
    assert stats["nonoverlap_blocks_20d"] == count_nonoverlap_blocks(
        stats["dates"], 20)
    assert len(stats["dates"]) == 39


def test_prospective_stats_start_at未設定時不計(tmp_path):
    """start_at 為 None（尚未部署）→ 一律不計，累積尚未開始。"""
    ledger_dir = tmp_path / "ledger"
    pub_dir = ledger_dir / "published"
    pub_dir.mkdir(parents=True)
    _write_record(pub_dir, "2026-10-27", "prospective", "2026-10-27T19:00:00+08:00")
    (ledger_dir / "index.json").write_text(json.dumps(
        [{"date": "2026-10-27", "data_cutoff_date": "2026-10-27",
          "prediction_sha256": "x" * 64, "published_at": "2026-10-27T19:00:00+08:00",
          "data_class": "prospective"}], ensure_ascii=False), encoding="utf-8")
    stats = prospective_stats(ledger_dir, None)
    assert stats["published_days"] == 0
    assert stats["nonoverlap_blocks_20d"] == 0
