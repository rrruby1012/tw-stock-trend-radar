"""發布 ledger 測試：write-once、兩段式證明、勘誤 append-only、prospective 統計分流。

規格（2026-10-22 鎖定）：
- 發布檔不得含自己的 publication commit SHA（自我參照不可能）；
  source_commit_sha / workflow_run_id / prediction_sha256 等欄位齊全。
- 同日同內容冪等；同日不同內容衝突且原檔不變。
- attestation（第二段證明）只 append 到 corrections ledger，不回寫原發布檔。
- prospective 統計只計 ledger 內、class=prospective 且 published_at >= start_at 的紀錄；
  staging/archive 的自我宣稱不算數。
"""
import json
from pathlib import Path

import pytest

from src import ledger


def _preds(v=0.55):
    return [{"symbol": "2330", "name": "台積電", "p_up_10": v},
            {"symbol": "1101", "name": "台泥", "p_up_10": 0.5}]


def _record(date="2026-10-23", v=0.55, published_at="2026-10-23T19:05:00+08:00"):
    return ledger.build_publication_record(
        predictions=_preds(v),
        data_cutoff_date=date,
        generated_at="2026-10-23T19:00:00+08:00",
        published_at=published_at,
        source_commit_sha="srcsha123",
        workflow_run_id="98765",
        model_version="logreg-isotonic-v1",
        config_version="2026-10-23.1",
    )


def test_雜湊正規化與鍵順序無關():
    a = [{"b": 1, "a": 2}]
    b = [{"a": 2, "b": 1}]
    assert ledger.prediction_sha256(a) == ledger.prediction_sha256(b)
    assert ledger.prediction_sha256(a) != ledger.prediction_sha256([{"a": 2, "b": 2}])


def test_發布紀錄欄位齊全且無自我參照():
    rec = _record()
    for k in ["data_class", "data_cutoff_date", "prediction_sha256",
              "source_commit_sha", "workflow_run_id", "model_version",
              "config_version", "generated_at", "published_at", "predictions"]:
        assert k in rec, f"發布紀錄缺欄位 {k}"
    assert rec["data_class"] == "prospective"
    # 兩段式證明：發布檔不得預寫自己的 publication commit SHA
    assert "publication_commit_sha" not in rec
    assert rec["prediction_sha256"] == ledger.prediction_sha256(rec["predictions"])


def test_publish_write_once_冪等與衝突(tmp_path):
    led = tmp_path / "ledger"
    rec = _record()
    out = ledger.publish(led, rec)
    assert out["prediction_sha256"] == rec["prediction_sha256"]
    target = led / "published" / "2026-10-23.json"
    bytes1 = target.read_bytes()
    # 同內容重發 → 冪等、檔案不變、index 不重複
    ledger.publish(led, _record())
    assert target.read_bytes() == bytes1
    idx = ledger.load_index(led)
    assert [e["date"] for e in idx] == ["2026-10-23"]
    # 不同內容 → 衝突，且原檔永不覆寫
    idx_bytes = (led / "index.json").read_bytes()
    with pytest.raises(ledger.LedgerConflict):
        ledger.publish(led, _record(v=0.99))
    assert target.read_bytes() == bytes1
    # 衝突時 index 也不得變動（失敗發生在任何狀態變更完成前）
    assert (led / "index.json").read_bytes() == idx_bytes


def test_兩段式證明_attestation不回寫原檔(tmp_path):
    led = tmp_path / "ledger"
    rec = _record()
    ledger.publish(led, rec)
    target = led / "published" / "2026-10-23.json"
    bytes_before = target.read_bytes()
    entry = ledger.append_attestation(
        led, date="2026-10-23",
        publication_commit_sha="pubsha456",
        attested_at="2026-10-24T09:00:00+08:00",
        prediction_sha256=rec["prediction_sha256"],
    )
    assert entry["kind"] == "attestation"
    assert entry["publication_commit_sha"] == "pubsha456"
    # 原發布檔逐位元不變，且仍不含 publication_commit_sha
    assert target.read_bytes() == bytes_before
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert "publication_commit_sha" not in saved
    # attestation 只存在於 append-only ledger
    entries = ledger.load_corrections(led)
    assert len(entries) == 1 and entries[0]["kind"] == "attestation"


def test_勘誤append_only且原檔不變(tmp_path):
    led = tmp_path / "ledger"
    ledger.publish(led, _record())
    target = led / "published" / "2026-10-23.json"
    bytes_before = target.read_bytes()
    ledger.append_correction(
        led, date="2026-10-23", reason="1101 收盤價來源更正",
        corrected_at="2026-10-24T10:00:00+08:00",
        original={"symbol": "1101", "close_raw": 26.8},
        corrected={"symbol": "1101", "close_raw": 26.85},
    )
    ledger.append_correction(
        led, date="2026-10-23", reason="補充說明",
        corrected_at="2026-10-24T11:00:00+08:00",
    )
    assert target.read_bytes() == bytes_before
    entries = ledger.load_corrections(led)
    assert len(entries) == 2
    assert all(e["kind"] == "correction" for e in entries)
    assert entries[0]["corrected"]["close_raw"] == 26.85


def test_prospective統計只計ledger且受start_at管控(tmp_path):
    led = tmp_path / "ledger"
    ledger.publish(led, _record(date="2026-10-23",
                                published_at="2026-10-23T19:05:00+08:00"))
    ledger.publish(led, _record(date="2026-10-26", v=0.56,
                                published_at="2026-10-26T19:05:00+08:00"))
    # start_at 未設定（尚未部署）→ 一律不計
    s0 = ledger.prospective_stats(led, None)
    assert s0["published_days"] == 0
    # start_at 在兩筆之間 → 只計較晚的一筆
    s1 = ledger.prospective_stats(led, "2026-10-24T00:00:00+08:00")
    assert s1["published_days"] == 1 and s1["dates"] == ["2026-10-26"]
    # start_at 更早 → 兩筆都計
    s2 = ledger.prospective_stats(led, "2026-10-01T00:00:00+08:00")
    assert s2["published_days"] == 2


def test_prospective統計跨時區界線(tmp_path):
    """start_at 以 UTC（Z）記錄、published_at 為 +08:00 時，必須以絕對時間比較。

    界線 2026-10-22T16:00:00Z = 台北 2026-10-23 00:00。
    台北 10/22 23:00（=15:00Z）在界線之前，不得計入——字串比對會誤判為之後。
    """
    led = tmp_path / "ledger"
    ledger.publish(led, _record(date="2026-10-22",
                                published_at="2026-10-22T23:00:00+08:00"))
    ledger.publish(led, _record(date="2026-10-23", v=0.56,
                                published_at="2026-10-23T19:05:00+08:00"))
    s = ledger.prospective_stats(led, "2026-10-22T16:00:00Z")
    assert s["published_days"] == 1 and s["dates"] == ["2026-10-23"]


def test_分類函式_舊資料不得升格():
    # 無欄位無旗標的舊 meta → reconstructed
    assert ledger.classify_meta({"data_dates": {"price": "2026-10-19"}}) == "reconstructed"
    # reconstructed 旗標
    assert ledger.classify_meta({"reconstructed": True}) == "reconstructed"
    # 明確欄位優先
    assert ledger.classify_meta({"data_class": "walkforward_sim"}) == "walkforward_sim"
    assert ledger.classify_meta({"provenance": {"data_class": "prospective"}}) == "prospective"
    # ledger 命中才可 prospective
    m = {"data_dates": {"price": "2026-10-23"}}
    assert ledger.classify_meta(m, ledger_dates={"2026-10-23"}) == "prospective"
    assert ledger.classify_meta(m, ledger_dates=set()) == "reconstructed"
    assert ledger.classify_meta(None) == "reconstructed"
