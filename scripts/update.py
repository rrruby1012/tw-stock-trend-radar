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
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as config_mod
from src import features as feat_mod
from src import labels as labels_mod
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


def train_final_and_predict(panel: pd.DataFrame, feature_cols: list[str], cfg: dict):
    """在全量資料上訓練最終模型（留 embargo），預測每個股票最新一列。"""
    from src.models import TARGETS, CalibratedModel, LogisticModel

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
        # 80/20 按日期切分做校準
        tdates = train.index.get_level_values("date").unique().sort_values()
        split = tdates[int(len(tdates) * 0.8)]
        fit_df = train[train.index.get_level_values("date") <= split]
        cal_df = train[train.index.get_level_values("date") > split]
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
        "risk_disclaimer": cfg["site"]["risk_disclaimer"],
        "is_trading_day": True,
        "data_delayed": bool(fetched.get("warnings")),
        "warnings": fetched.get("warnings", [])[:5],
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
    )
    log.info("網站已產生至 %s", docs_dir)

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
