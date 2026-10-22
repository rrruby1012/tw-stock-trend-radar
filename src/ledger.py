"""真實前瞻發布 ledger（immutable published ledger）。

設計（2026-10-22 鎖定規格）：
- 資料三分類，靠明確欄位分流，禁止以日期/檔名猜測：
  walkforward_sim（回溯樣本外模擬）、reconstructed（事後重建）、
  prospective（真實前瞻發布）。前兩類永遠不進 prospective 統計。
- Staging（docs/data/*.json、archive/）是 mutable 的工作區，同日可重跑覆寫；
  只有「正式發布」動作把當日紀錄凍結進本 ledger（published/），write-once。
- 兩段式證明：發布檔「不可能」預先包含自己的 git commit SHA，因此發布紀錄
  只記錄 source_commit_sha（觸發工作流的程式/設定 commit）、workflow_run_id、
  prediction_sha256、model_version、config_version、generated_at、
  published_at、data_cutoff_date。實際包含發布檔的 publication commit SHA
  由 Git 歷史提供，必要時事後以 append-only 的 attestation 條目引用，
  永不回寫原發布檔。
- 勘誤：append-only correction ledger（corrections.json），原發布檔永不覆寫；
  前端同時顯示原值與更正原因。
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

CLASS_WALKFORWARD = "walkforward_sim"
CLASS_RECONSTRUCTED = "reconstructed"
CLASS_PROSPECTIVE = "prospective"
DATA_CLASSES = (CLASS_WALKFORWARD, CLASS_RECONSTRUCTED, CLASS_PROSPECTIVE)


class LedgerConflict(RuntimeError):
    """同一日期已凍結不同內容的發布紀錄（write-once 衝突）。"""


def canonical_predictions_bytes(predictions) -> bytes:
    """預測內容的正規化序列化（鍵排序、固定分隔符），供雜湊與比對。"""
    return json.dumps(
        predictions, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def prediction_sha256(predictions) -> str:
    return hashlib.sha256(canonical_predictions_bytes(predictions)).hexdigest()


def classify_meta(meta: dict | None, ledger_dates: set[str] | None = None) -> str:
    """由明確欄位判定一份 meta 的資料類別（讀取端共用規則）。

    優先序：明確 data_class 欄位 > reconstructed 旗標 > ledger 命中。
    無任何證明的舊資料一律 reconstructed（不得升格 prospective）。
    """
    if not meta:
        return CLASS_RECONSTRUCTED
    dc = meta.get("data_class") or (meta.get("provenance") or {}).get("data_class")
    if dc in DATA_CLASSES:
        return dc
    if meta.get("reconstructed"):
        return CLASS_RECONSTRUCTED
    date = (meta.get("data_dates") or {}).get("price")
    if ledger_dates and date in ledger_dates:
        return CLASS_PROSPECTIVE
    return CLASS_RECONSTRUCTED


# ---------- 路徑 ----------

def _published_dir(ledger_dir: Path) -> Path:
    return Path(ledger_dir) / "published"


def _index_path(ledger_dir: Path) -> Path:
    return Path(ledger_dir) / "index.json"


def _corrections_path(ledger_dir: Path) -> Path:
    return Path(ledger_dir) / "corrections.json"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- 發布（write-once） ----------

def build_publication_record(
    *,
    predictions: list,
    data_cutoff_date: str,
    generated_at: str,
    published_at: str,
    source_commit_sha: str,
    workflow_run_id: str | None,
    model_version: str,
    config_version: str,
) -> dict:
    """組裝發布紀錄。刻意不含 publication_commit_sha（自我參照不可能）。"""
    return {
        "data_class": CLASS_PROSPECTIVE,
        "data_cutoff_date": data_cutoff_date,
        "prediction_sha256": prediction_sha256(predictions),
        "source_commit_sha": source_commit_sha,
        "workflow_run_id": workflow_run_id,
        "model_version": model_version,
        "config_version": config_version,
        "generated_at": generated_at,
        "published_at": published_at,
        "predictions": predictions,
    }


def publish(ledger_dir: str | Path, record: dict) -> dict:
    """把發布紀錄凍結進 ledger。write-once：

    - 同日期、同雜湊 → 冪等，回傳既有紀錄（同日重跑安全）。
    - 同日期、不同雜湊 → LedgerConflict，絕不覆寫。
    - index.json 只 append 新日期；已存在日期不重複加入。
    """
    ledger_dir = Path(ledger_dir)
    pub_dir = _published_dir(ledger_dir)
    pub_dir.mkdir(parents=True, exist_ok=True)
    date = record["data_cutoff_date"]
    target = pub_dir / f"{date}.json"
    if target.exists():
        existing = _read_json(target, {})
        if existing.get("prediction_sha256") == record["prediction_sha256"]:
            log.info("ledger：%s 已凍結且內容相同，冪等跳過", date)
            return existing
        raise LedgerConflict(
            f"{date} 已有不同內容的凍結發布紀錄"
            f"（既有 {existing.get('prediction_sha256')} vs 新 {record['prediction_sha256']}）；"
            "write-once 禁止覆寫，請改走 correction ledger 勘誤。"
        )
    target.write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    index = _read_json(_index_path(ledger_dir), [])
    if not any(e.get("date") == date for e in index):
        index.append({
            "date": date,
            "data_cutoff_date": date,
            "prediction_sha256": record["prediction_sha256"],
            "published_at": record["published_at"],
            "data_class": CLASS_PROSPECTIVE,
        })
        index.sort(key=lambda e: e["date"])
        _index_path(ledger_dir).write_text(
            json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    log.info("ledger：已凍結 %s 的發布紀錄（%s）", date, record["prediction_sha256"][:12])
    return record


def load_index(ledger_dir: str | Path) -> list:
    return _read_json(_index_path(ledger_dir), [])


def load_published(ledger_dir: str | Path, date: str) -> dict | None:
    return _read_json(_published_dir(Path(ledger_dir)) / f"{date}.json", None)


# ---------- 勘誤 / 證明（append-only） ----------

def _append_entry(ledger_dir: Path, entry: dict) -> dict:
    ledger_dir = Path(ledger_dir)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    entries = _read_json(_corrections_path(ledger_dir), [])
    entries.append(entry)
    _corrections_path(ledger_dir).write_text(
        json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return entry


def append_correction(
    ledger_dir: str | Path,
    *,
    date: str,
    reason: str,
    corrected_at: str,
    original: dict | None = None,
    corrected: dict | None = None,
    references_publication_commit_sha: str | None = None,
) -> dict:
    """追加一筆勘誤。原發布檔不動；前端同時顯示原值、更正值與原因。"""
    entry = {
        "kind": "correction",
        "date": date,
        "reason": reason,
        "corrected_at": corrected_at,
        "original": original,
        "corrected": corrected,
    }
    if references_publication_commit_sha:
        entry["references_publication_commit_sha"] = references_publication_commit_sha
    return _append_entry(Path(ledger_dir), entry)


def append_attestation(
    ledger_dir: str | Path,
    *,
    date: str,
    publication_commit_sha: str,
    attested_at: str,
    prediction_sha256: str | None = None,
) -> dict:
    """兩段式證明・第二段：事後追加引用「包含發布檔的 git commit SHA」。

    只寫入 corrections.json（append-only），永不回寫 published/ 原檔。
    """
    entry = {
        "kind": "attestation",
        "date": date,
        "publication_commit_sha": publication_commit_sha,
        "attested_at": attested_at,
    }
    if prediction_sha256:
        entry["prediction_sha256"] = prediction_sha256
    return _append_entry(Path(ledger_dir), entry)


def load_corrections(ledger_dir: str | Path) -> list:
    return _read_json(_corrections_path(Path(ledger_dir)), [])


# ---------- prospective 統計（只計 ledger，A/B 永不混入） ----------

def _parse_ts(s) -> "datetime | None":
    """解析 ISO 時間戳為 aware datetime；naive 視為 UTC。

    published_at 帶 +08:00、start_at 以 UTC（Z）記錄時，字串比對會判錯，
    必須正規化為絕對時間再比較。
    """
    if not s:
        return None
    from datetime import datetime, timezone

    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def prospective_stats(ledger_dir: str | Path, start_at: str | None) -> dict:
    """真實前瞻累積統計。只讀 ledger 的 index 與 published 紀錄，

    且只計 data_class == prospective 且 published_at >= start_at 者。
    start_at 為 None（尚未部署）→ 一律不計（累積尚未開始）。
    時間比較以絕對時間進行（相容 +08:00 與 Z 兩種記錄格式）。
    """
    ledger_dir = Path(ledger_dir)
    start_dt = _parse_ts(start_at)
    dates: list[str] = []
    for e in load_index(ledger_dir):
        rec = load_published(ledger_dir, e.get("date"))
        if not rec or rec.get("data_class") != CLASS_PROSPECTIVE:
            continue
        if start_dt is None:
            continue
        pub_dt = _parse_ts(rec.get("published_at"))
        if pub_dt is None or pub_dt < start_dt:
            continue
        dates.append(e["date"])
    dates.sort()
    # 不重疊 20 交易日區塊數：以已發布交易日序列貪婪切分（近似，僅供累積進度）
    blocks = 0
    i = 0
    while i + 20 <= len(dates):
        blocks += 1
        i += 20
    return {
        "published_days": len(dates),
        "dates": dates,
        "nonoverlap_blocks_20d": blocks,
        "start_at": start_at,
        "accumulating": True,
    }
