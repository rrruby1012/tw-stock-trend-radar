"""主更新流程：抓取 → 特徵 → 驗證 → 預測 → 回測 → 產生網站。

用法：
    python scripts/update.py [--config config.yaml] [--full]

- 預設增量更新（使用 data/cache 快取）。
- --full：忽略快取，全量重抓。
- 非交易日（無新資料）時正常結束，不報錯。
- API 暫時失敗時保留前一次正確結果（不覆寫 docs/）。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as config_mod
from src import features as feat_mod
from src import labels as labels_mod
from src import ledger as ledger_mod
from src import universe as universe_mod
from src.fetch import fetch_all
from src.sitegen import generate_site

log = logging.getLogger("update")


def parse_args():
    p = argparse.ArgumentParser(description="台股趨勢雷達更新流程")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--full", action="store_true", help="忽略快取，全量重抓")
    p.add_argument("--symbols", default=None, help="僅更新指定股票（逗號分隔），測試用")
    p.add_argument("--force", action="store_true", help="忽略「無新資料」檢查，強制完整重跑")
    return p.parse_args()


def latest_stored_price_date(docs_dir: Path) -> str | None:
    meta = docs_dir / "data" / "meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8"))["data_dates"].get("price")
        except Exception:  # noqa: BLE001
            return None
    return None


def _source_commit_sha(root: Path) -> str:
    """觸發本次工作流的程式/設定 commit（兩段式證明・第一段）。

    GitHub Actions 以 GITHUB_SHA 提供；本機則讀 git HEAD；皆無時為 "unknown"。
    注意：這不是包含發布檔的 publication commit（那由 Git 歷史/attestation 提供）。
    """
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def train_final_and_predict(panel: pd.DataFrame, feature_cols: list[str], cfg: dict):
    """在全量資料上訓練最終模型（留 embargo），預測每個股票最新一列。"""
    from src.models import TARGETS, CalibratedModel, LogisticModel

    # 防禦：排除標籤欄（避免目標洩漏；與 validation.walk_forward 一致）
    label_cols = set(TARGETS) | {"fwd_ret_10", "fwd_ret_20"}
    feature_cols = [c for c in feature_cols if c not in label_cols and c in panel.columns]
    if not feature_cols:
        raise ValueError("沒有可用的特徵欄位。")

    vcfg = cfg["model"]["validation"]
    embargo = vcfg.get("embargo_days", 20)
    lr_cfg = cfg["model"]["logistic_regression"]

    dates = panel.index.get_level_values("date").unique().sort_values()
    max_date = dates.max()
    cutoff = max_date - pd.Timedelta(days=embargo * 1.6)  # 交易日換算寬鬆估計

    latest_idx = panel.groupby(level="stock_id").tail(1).index
    results = {}
    for target in TARGETS:
        df = panel.dropna(subset=[target] + feature_cols).copy()
        if df.empty:
            continue
        train = df[df.index.get_level_values("date") <= cutoff]
        if len(train) < 500:
            log.warning("%s 訓練樣本不足（%d），跳過", target, len(train))
            continue
        # 80/20 按日期切分做校準，fit 與 cal 之間隔 embargo（與 validation
        # 相同慣例：embargo_days，預設 20 個交易日；train/test 的 embargo
        # 數值本身不動；被空出的日期兩邊都不用，不得回退為相鄰切分）。
        # 審計原問題：舊切分 fit/cal 日期相鄰，標籤窗格重疊的樣本會同時出現
        # 在 fit 與 cal 兩側，造成校準洩漏。
        from src.validation import split_fit_cal_dates

        tdates = train.index.get_level_values("date").unique().sort_values()
        fit_dates, cal_dates = split_fit_cal_dates(
            tdates, fit_fraction=0.8, embargo_days=embargo
        )
        if len(fit_dates) == 0 or len(cal_dates) == 0:
            log.warning("%s fit/cal 切分後校準集為空（embargo 後無日期），跳過",
                        target)
            continue
        fit_df = train[train.index.get_level_values("date").isin(fit_dates)]
        cal_df = train[train.index.get_level_values("date").isin(cal_dates)]
        X_fit, y_fit = fit_df[feature_cols], fit_df[target]
        X_cal, y_cal = cal_df[feature_cols], cal_df[target]
        est = LogisticModel(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            class_weight=lr_cfg.get("class_weight", "balanced"),
        )
        model = CalibratedModel(est).fit(X_fit, y_fit, X_cal, y_cal)
        X_latest = panel.loc[latest_idx, feature_cols]
        proba = model.predict_proba(X_latest)
        results[target] = pd.Series(proba, index=X_latest.index.get_level_values("stock_id"))
        log.info("%s 最終模型訓練完成，預測 %d 檔", target, len(results[target]))
    return results, max_date


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(__file__).resolve().parent.parent
    cfg = config_mod.load_config(root / args.config)

    # 1. 宇宙
    watchlist = universe_mod.load_watchlist(root / "watchlist.csv")
    if args.symbols:
        keep = {s.strip().zfill(4) for s in args.symbols.split(",")}
        watchlist = watchlist[watchlist["symbol"].isin(keep)]
    symbols = watchlist["symbol"].tolist()
    log.info("分析宇宙：%d 檔", len(symbols))

    # 2. 抓取
    years = cfg["data"].get("history_years", 5)
    end_date = date.today()
    start_date = end_date - timedelta(days=int(years * 365.25) + 200)
    try:
        fetched = fetch_all(
            cfg, symbols, start_date, end_date, incremental=not args.full
        )
    except Exception as e:  # noqa: BLE001
        log.error("資料抓取失敗，保留前一次結果：%s", e)
        return 0  # 不報錯結束（API 暫時失敗時保留前次正確結果）

    data_dates = fetched["data_dates"]
    price_date = data_dates.get("price_adj")
    log.info("資料日期：%s", data_dates)

    # 3. 非交易日判斷：無新價格資料則正常結束
    docs_dir = root / cfg["site"].get("output_dir", "docs")
    prev_price_date = latest_stored_price_date(docs_dir)
    if not args.force and prev_price_date and price_date and price_date <= prev_price_date:
        log.info("無新交易資料（%s），正常結束", price_date)
        return 0

    # 4. 排除規則
    included, excluded = universe_mod.apply_exclusion_rules(
        watchlist,
        fetched["datasets"]["price_adj"],
        min_history_days=cfg["data"].get("min_history_days", 500),
        max_missing_ratio=cfg["data"].get("max_missing_ratio", 0.05),
        min_avg_turnover=cfg["data"].get("min_avg_turnover", 5_000_000),
    )
    if included.empty:
        log.error("排除後無股票可分析")
        return 1
    inc_symbols = included["symbol"].tolist()
    log.info("納入 %d 檔，排除 %d 檔", len(included), len(excluded))

    # 5. 特徵 + 標籤
    ds = fetched["datasets"]
    panel = feat_mod.build_panel(
        {s: ds["price_adj"][s] for s in inc_symbols if s in ds["price_adj"]},
        {s: ds["institutional"].get(s) for s in inc_symbols},
        {s: ds["margin"].get(s) for s in inc_symbols},
        {s: ds["revenue"].get(s) for s in inc_symbols},
        {s: ds["financials"].get(s) for s in inc_symbols},
        fetched["index_close"],
        cfg,
    )
    panel = labels_mod.add_labels(panel, fetched["index_close"], cfg["horizons"]["days"])
    feature_cols = feat_mod.feature_columns(panel)
    log.info("特徵 panel：%s，特徵數 %d", panel.shape, len(feature_cols))

    # 6. walk-forward 驗證 + 可信度
    from src.validation import credibility_check, walk_forward

    val_results = walk_forward(panel, feature_cols, cfg)
    credible, reasons, val_status = credibility_check(val_results, cfg)
    log.info("模型可信度：%s（%s）[%s]", credible, "; ".join(reasons), val_status)

    # 7. 回測
    from src.backtest import run_backtest

    backtest_results = run_backtest(
        val_results, panel, fetched["index_close"], cfg, credible=credible
    )

    # 8. 最終預測
    proba_series, max_pred_date = train_final_and_predict(panel, feature_cols, cfg)

    # 9. 產生網站
    up10m = val_results["targets"]["up_10"]["metrics"]
    meta = {
        "generated_at": pd.Timestamp.now(tz="Asia/Taipei").isoformat(),
        "data_dates": {
            "price": data_dates.get("price_adj"),
            "institutional": data_dates.get("institutional"),
            "margin": data_dates.get("margin"),
            "revenue": data_dates.get("revenue"),
            "financials": data_dates.get("financials"),
            "index": data_dates.get("index"),
        },
        "sources": ["FinMind", "臺灣證券交易所 OpenAPI（備援）"],
        "universe": {"included": len(included), "excluded": len(excluded)},
        "excluded": excluded.to_dict(orient="records") if not excluded.empty else [],
        "model_valid": bool(credible),
        "validation_status": val_status,
        "credibility_reasons": reasons,
        # 前端「模型驗證摘要」與頁尾樣本數讀取此欄位（缺值時前端顯示「資料暫無」）
        "credibility": {
            "target": "up_10",
            "oos_n": up10m.get("n"),
            "auc": up10m.get("auc"),
            "auc_baseline": up10m.get("auc_always"),
            "brier": up10m.get("brier"),
            "brier_baseline": up10m.get("brier_always"),
        },
        "risk_disclaimer": cfg["site"]["risk_disclaimer"],
        "is_trading_day": True,
        "data_delayed": bool(fetched.get("warnings")),
        "warnings": fetched.get("warnings", [])[:5],
    }
    provenance = {
        "model_version": cfg.get("project", {}).get("model_version"),
        "config_version": cfg.get("project", {}).get("config_version"),
        "source_commit_sha": _source_commit_sha(root),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
    }
    generate_site(
        panel=panel,
        proba_series=proba_series,
        val_results=val_results,
        backtest_results=backtest_results,
        meta=meta,
        watchlist=included,
        cfg=cfg,
        output_dir=docs_dir,
        provenance=provenance,
    )
    log.info("網站已產生至 %s", docs_dir)

    # 9b. 覆核：納入名單的每檔都必須有預測與歷史檔，否則列為排除並寫明原因
    # （不可靜默遺漏，避免網站顯示「排除 0」卻實際缺檔）
    preds_path = docs_dir / "data" / "predictions.json"
    preds = json.loads(preds_path.read_text(encoding="utf-8"))
    pred_symbols = {str(p.get("symbol")) for p in preds}
    name_map = dict(zip(included["symbol"].astype(str), included["name"]))
    missing = [s for s in inc_symbols if str(s) not in pred_symbols]
    hist_missing = [
        s for s in inc_symbols
        if not (docs_dir / "data" / f"history_{s}.json").exists()
    ]
    extra_excluded = []
    for s in sorted(set(missing) | set(hist_missing)):
        why = []
        if str(s) in map(str, missing):
            why.append("未產生預測資料")
        if s in hist_missing:
            why.append("未產生個股歷史檔")
        extra_excluded.append({
            "symbol": str(s),
            "name": name_map.get(str(s), str(s)),
            "排除原因": "；".join(why) + "（待查）",
            "資料天數": None,
        })
        log.warning("覆核發現 %s 缺檔：%s", s, "；".join(why))
    if extra_excluded:
        meta_path = docs_dir / "data" / "meta.json"
        meta_disk = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_disk["excluded"] = list(meta_disk.get("excluded") or []) + extra_excluded
        meta_disk["universe"] = {
            "included": len(pred_symbols),
            "excluded": len(meta_disk["excluded"]),
        }
        meta_path.write_text(
            json.dumps(meta_disk, ensure_ascii=False, default=str), encoding="utf-8"
        )
        # 同步當日存檔的 meta（若已產生）
        arch_date = (meta_disk.get("data_dates") or {}).get("price")
        if arch_date:
            arch_meta = docs_dir / "data" / "archive" / f"meta_{arch_date}.json"
            if arch_meta.exists():
                arch_meta.write_text(
                    json.dumps(meta_disk, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
        log.info("已將 %d 檔缺檔股票列為排除", len(extra_excluded))

    # 9c. 正式發布：把當日預測凍結進 immutable ledger（write-once）。
    # staging（docs/data/*.json、archive）可重跑覆寫，ledger 不可；
    # 同日同內容冪等、同日不同內容視為衝突並讓本次 run 失敗（不靜默覆寫）。
    meta_path = docs_dir / "data" / "meta.json"
    meta_disk = json.loads(meta_path.read_text(encoding="utf-8"))
    final_preds = json.loads(
        (docs_dir / "data" / "predictions.json").read_text(encoding="utf-8")
    )
    published_at = pd.Timestamp.now(tz="Asia/Taipei").isoformat()
    record = ledger_mod.build_publication_record(
        predictions=final_preds,
        data_cutoff_date=(meta_disk.get("data_dates") or {}).get("price"),
        generated_at=meta_disk.get("generated_at"),
        published_at=published_at,
        source_commit_sha=provenance["source_commit_sha"],
        workflow_run_id=provenance["workflow_run_id"],
        model_version=provenance["model_version"],
        config_version=provenance["config_version"],
    )
    ledger_dir = docs_dir / "data" / "ledger"
    try:
        frozen = ledger_mod.publish(ledger_dir, record)
    except ledger_mod.LedgerConflict as e:
        log.error("發布凍結失敗：%s", e)
        return 1
    # 回寫 staging meta 的發布狀態（ledger 原檔不動；此處只補 staging 標記）
    for mp in [meta_path,
               docs_dir / "data" / "archive" / f"meta_{record['data_cutoff_date']}.json"]:
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            prov = dict(m.get("provenance") or {})
            prov["ledger_status"] = "frozen"
            prov["published_at"] = frozen.get("published_at")
            m["provenance"] = prov
            mp.write_text(json.dumps(m, ensure_ascii=False, default=str), encoding="utf-8")
    pstats = ledger_mod.prospective_stats(
        ledger_dir, cfg.get("prospective", {}).get("start_at")
    )
    log.info("前瞻發布累積：%d 日（start_at=%s）",
             pstats["published_days"], pstats["start_at"])
    # 前瞻累積進度寫入 live meta（前端「真實前瞻紀錄」區讀取；未達門檻只能標累積中）
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    m["prospective_progress"] = {
        "published_days": pstats["published_days"],
        "nonoverlap_blocks_20d": pstats["nonoverlap_blocks_20d"],
        "start_at": pstats["start_at"],
        "min_published_days": cfg.get("prospective", {}).get("min_published_days", 120),
        "status": "累積中",
    }
    meta_path.write_text(json.dumps(m, ensure_ascii=False, default=str), encoding="utf-8")

    # 10. 摘要
    up10 = val_results["targets"]["up_10"]["metrics"]
    print("==== 更新完成 ====")
    print(f"資料日期（股價）：{data_dates.get('price_adj')}")
    print(f"納入股票：{len(included)} 檔")
    print(f"模型可信：{credible}")
    print(f"up_10 樣本外：n={up10.get('n')} hit_rate={up10.get('hit_rate')} "
          f"brier={up10.get('brier')} auc={up10.get('auc')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
