"""網站產生器：將分析結果轉為 docs/ 下的 JSON，供靜態前端讀取。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _r(x, nd=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), nd)


def derive_signals(feat: pd.Series) -> tuple[list[str], list[str], str]:
    """由最新特徵列推導有利/風險訊號與失效條件（規則式、可解釋）。"""
    bull, bear = [], []

    def has(k):
        v = feat.get(k)
        return v is not None and not (isinstance(v, float) and np.isnan(v))

    if has("ma_slope_20_60"):
        if feat["ma_slope_20_60"] > 0.005:
            bull.append("20日均線向上穿越60日均線（中期趨勢向上）")
        elif feat["ma_slope_20_60"] < -0.005:
            bear.append("20日均線向下跌破60日均線（中期趨勢向下）")
    if has("ma_ratio_20"):
        if feat["ma_ratio_20"] > 0:
            bull.append("股價站上20日均線")
        else:
            bear.append("股價跌破20日均線")
    if has("inst_foreign_20") and feat["inst_foreign_20"] > 0:
        bull.append("外資近20日買超")
    elif has("inst_foreign_20") and feat["inst_foreign_20"] < 0:
        bear.append("外資近20日賣超")
    if has("inst_trust_20") and feat["inst_trust_20"] > 0:
        bull.append("投信近20日買超")
    if has("revenue_yoy") and feat["revenue_yoy"] > 0:
        bull.append("月營收年增率為正")
    elif has("revenue_yoy") and feat["revenue_yoy"] < 0:
        bear.append("月營收年增率為負")
    if has("rsi_14"):
        if feat["rsi_14"] > 70:
            bear.append("RSI高於70（短線過熱）")
        elif feat["rsi_14"] < 30:
            bull.append("RSI低於30（短線超賣）")
    if has("macd_hist"):
        if feat["macd_hist"] > 0:
            bull.append("MACD動能為正")
        else:
            bear.append("MACD動能為負")
    if has("volatility_20") and has("volatility_60"):
        if feat["volatility_20"] > feat["volatility_60"] * 1.5:
            bear.append("近期波動度明顯放大")
    if has("margin_chg_20") and feat["margin_chg_20"] > 0.15:
        bear.append("融資餘額20日增幅超過15%（追價風險）")
    if has("rel_strength_20"):
        if feat["rel_strength_20"] > 0:
            bull.append("近20日跑贏加權指數")
        else:
            bear.append("近20日落後加權指數")

    invalidation = "若收盤跌破60日均線，或10日上漲機率估計值跌破0.4，本次趨勢判斷失效。"
    return bull[:5], bear[:5], invalidation


def _max_drawdown(close: pd.Series) -> float | None:
    if close.empty or len(close) < 2:
        return None
    roll_max = close.cummax()
    dd = (close / roll_max - 1).min()
    return float(dd)


def generate_site(
    panel: pd.DataFrame,
    proba_series: dict[str, pd.Series],
    val_results: dict,
    backtest_results: dict,
    meta: dict,
    watchlist: pd.DataFrame,
    cfg: dict,
    output_dir: str | Path,
) -> None:
    from .models import TARGETS, classify_trend, price_interval

    out = Path(output_dir)
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    tcfg = cfg["trend"]
    name_map = dict(zip(watchlist["symbol"], watchlist["name"]))
    latest_idx = panel.groupby(level="stock_id").tail(1).index

    predictions = []
    for sid in sorted(panel.index.get_level_values("stock_id").unique()):
        sub = panel.loc[sid].sort_index()
        latest = sub.iloc[-1]
        close = float(latest["close"])
        price_date = sub.index.max().strftime("%Y-%m-%d")

        p = {t: proba_series.get(t, pd.Series()).get(sid) for t in TARGETS}
        vol60 = latest.get("volatility_60")
        iv10 = price_interval(close, vol60, 10) if vol60 else None
        iv20 = price_interval(close, vol60, 20) if vol60 else None
        # 機率缺值時以 0.5（無方向）判定趨勢，避免 float(None) 報錯；
        # 輸出的機率欄位仍保留 None 以誠實呈現缺值。
        trend_p10 = _r(p.get("up_10"))
        trend_p20 = _r(p.get("up_20"))
        trend = classify_trend(
            trend_p10 if trend_p10 is not None else 0.5,
            trend_p20 if trend_p20 is not None else 0.5,
            tcfg["bullish_threshold"], tcfg["bearish_threshold"],
        )
        bull, bear, invalidation = derive_signals(latest)

        # 該股樣本外命中率
        oos = val_results["targets"]["up_10"]["oos"]
        oos_s = oos[oos["stock_id"] == sid]
        hit_rate = (
            float(((oos_s["proba"] >= 0.5) == (oos_s["y_true"] == 1)).mean())
            if len(oos_s) >= 20 else None
        )
        mdd = _max_drawdown(sub["close"].tail(250))

        # 顯示用官方原始收盤價；還原價只供模型計算
        close_raw = float(latest["close_raw"]) if "close_raw" in latest and pd.notna(latest["close_raw"]) else close
        predictions.append({
            "symbol": sid,
            "name": name_map.get(sid, sid),
            "close": round(close, 2),
            "close_raw": round(close_raw, 2),
            "price_date": price_date,
            "p_up_10": _r(p.get("up_10")),
            "p_up_20": _r(p.get("up_20")),
            "p_exc_10": _r(p.get("exc_10")),
            "p_exc_20": _r(p.get("exc_20")),
            "interval_10": {"lo": _r(iv10["lo"], 2), "hi": _r(iv10["hi"], 2)} if iv10 else None,
            "interval_20": {"lo": _r(iv20["lo"], 2), "hi": _r(iv20["hi"], 2)} if iv20 else None,
            "downside_risk_10": _r(iv10["downside_var5"]) if iv10 else None,
            "downside_risk_20": _r(iv20["downside_var5"]) if iv20 else None,
            "trend": trend,
            "bull_signals": bull,
            "bear_signals": bear,
            "invalidation": invalidation,
            "hit_rate_10": _r(hit_rate),
            "oos_n": int(len(oos_s)),
            "max_drawdown": _r(mdd),
            "adjusted_price": True,
            "price_warning": None,
        })

        # 個股歷史檔（顯示用原始收盤價）
        close_raw_s = sub["close_raw"] if "close_raw" in sub.columns else sub["close"]
        hist_prices = [
            [d.strftime("%Y-%m-%d"), round(float(c), 2), int(v)]
            for d, c, v in zip(sub.index, close_raw_s, sub["volume"])
        ][-260:]
        pred_hist = []
        if not oos_s.empty and "date" in oos_s.columns:
            for _, r in oos_s.sort_values("date").iterrows():
                actual = None if pd.isna(r["y_true"]) else bool(r["y_true"] == 1)
                pred_hist.append([r["date"].strftime("%Y-%m-%d"), _r(r["proba"]), actual])
        (data_dir / f"history_{sid}.json").write_text(
            json.dumps({"symbol": sid, "name": name_map.get(sid, sid),
                        "prices": hist_prices, "predictions": pred_hist},
                       ensure_ascii=False),
            encoding="utf-8",
        )

    # 若模型不可信：不顯示推薦排名（前端依 model_valid 隱藏），此處仍輸出原始數值供查核
    (data_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False), encoding="utf-8"
    )

    # 驗證結果
    val_out = {"credible": bool(meta.get("model_valid")), "targets": {}}
    for t, tres in val_results["targets"].items():
        val_out["targets"][t] = {
            "metrics": {k: _r(v) if isinstance(v, float) else v
                        for k, v in tres["metrics"].items()},
            "periods": [
                {**{k: (v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else v)
                      for k, v in p.items() if k != "metrics"},
                 "metrics": {kk: _r(vv) if isinstance(vv, float) else vv
                             for kk, vv in p.get("metrics", {}).items()}}
                for p in tres.get("periods", [])
            ],
        }
    val_out["backtest"] = backtest_results
    (data_dir / "validation.json").write_text(
        json.dumps(val_out, ensure_ascii=False, default=str), encoding="utf-8"
    )

    (data_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log.info("產生 %d 檔預測、%d 個歷史檔", len(predictions), len(predictions))

    # ---------- 每日預測存檔（供歷史回看） ----------
    archive_dir = data_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    arch_date = (meta.get("data_dates") or {}).get("price") or pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d")
    (archive_dir / f"predictions_{arch_date}.json").write_text(
        json.dumps(predictions, ensure_ascii=False), encoding="utf-8"
    )
    (archive_dir / f"meta_{arch_date}.json").write_text(
        json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8"
    )
    # 存檔索引（只保留最近 90 天）
    index_path = archive_dir / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    except Exception:  # noqa: BLE001
        index = []
    if not any(e.get("date") == arch_date for e in index):
        index.append({"date": arch_date, "generated_at": meta.get("generated_at")})
    index = sorted(index, key=lambda e: e.get("date", ""), reverse=True)[:90]
    # 清掉超過 90 天的舊檔
    keep_dates = {e["date"] for e in index}
    for old in list(archive_dir.glob("predictions_*.json")) + list(archive_dir.glob("meta_*.json")):
        stem_date = old.stem.rsplit("_", 1)[-1]
        if stem_date not in keep_dates:
            old.unlink(missing_ok=True)
    index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    log.info("已寫入 %s 的預測存檔（共 %d 天）", arch_date, len(index))
