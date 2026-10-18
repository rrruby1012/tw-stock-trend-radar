# 網站說明（GitHub Pages 靜態儀表板）

> 本目錄為 GitHub Pages 網站根目錄。純 HTML / CSS / JS，無建置步驟，
> 後端只需產生 `data/*.json`，前端即可直接用瀏覽器開啟或部署。

## 檔案一覽

| 檔案 | 用途 |
|---|---|
| `index.html` | 首頁：市場狀態、預測排行榜（可排序）、風險排行榜、模型驗證摘要、頁尾（資料來源/日期/免責聲明） |
| `stock.html` | 個股頁，以 `?symbol=2330` 指定股票：基本資訊、機率估計、價格區間、訊號、走勢圖、預測紀錄 |
| `assets/style.css` | 手機優先 RWD 深色金融風格（漲=紅、跌=綠，依台股慣例） |
| `assets/app.js` | 共用 JS：`App.fetchJSON`、`App.fmtPct/fmtPrice/fmtDate`、`App.drawPriceChart`（SVG 自繪，無外部 CDN）、`App.buildSortableTable`（可排序表格） |
| `data/*.json` | 後端產生的資料（見下方 schema） |

## 後端需產生的 JSON

### `data/meta.json`

```jsonc
{
  "generated_at": "2026-10-18T18:30:00+08:00",   // 本次產生時間（ISO 8601）
  "data_dates": {                                 // 各類資料的實際日期（字串，null 表無資料）
    "price": "2026-10-16",
    "institutional": "2026-10-15",                // 18:30 排程時通常為 T-1
    "margin": "2026-10-15",
    "revenue": "2026-09-30",
    "financials": "2026-06-30"
  },
  "sources": ["FinMind", "TWSE OpenAPI"],          // 實際使用的資料來源
  "universe": { "included": 48, "excluded": 2 },   // 納入/排除檔數
  "model_valid": true,                            // false → 首頁顯示警示且隱藏排行榜
  "credibility": {
    "target": "up_10",
    "oos_n": 12345,                               // 樣本外樣本數
    "auc": 0.56, "auc_baseline": 0.52,
    "brier": 0.241, "brier_baseline": 0.248
  },
  "risk_disclaimer": "本工具僅供研究與教育用途…", // 免責聲明全文
  "is_trading_day": true,
  "data_delayed": false                           // true → 顯示「資料可能延遲」
}
```

### `data/predictions.json`（陣列）

```jsonc
[{
  "symbol": "2330",
  "name": "台積電",
  "close": 1085.0,                 // 最新收盤價（null 表無資料）
  "price_date": "2026-10-16",
  "p_up_10": 0.62,                 // 0~1 分數；前端一律標示「估計值」
  "p_up_20": 0.58,
  "p_exc_10": 0.55,                // 跑贏加權指數機率
  "p_exc_20": 0.51,
  "interval_10": { "lo": 1020.0, "hi": 1150.0 },
  "interval_20": { "lo": 990.0, "hi": 1180.0 },
  "downside_risk_10": 0.06,        // 下行風險（0~1）
  "downside_risk_20": 0.09,
  "trend": "偏多",                 // 偏多 | 盤整 | 偏空
  "bull_signals": ["站上月線…"],
  "bear_signals": ["外資連續賣超…"],
  "invalidation": "若跌破…則判斷失效",
  "hit_rate_10": 0.57,
  "oos_n": 250,
  "max_drawdown": -0.12,
  "adjusted_price": true,          // 是否為還原權息價格
  "price_warning": null            // 有則顯示警示文字（例如除權息跳空提醒）
}]
```

### `data/history_<SYMBOL>.json`

```jsonc
{
  "symbol": "2330",
  "name": "台積電",
  "prices": [["2025-10-20", 1040.0, 25000000], "..."],  // [日期, 收盤, 成交量]，遞增，取近 250 筆
  "predictions": [["2026-09-01", 0.62, 1], "..."]
  // [預測日期, 當時 p_up_10, 實際是否上漲 1/0/null（尚未到期則 null）]
}
```

### `data/validation.json`

```jsonc
{
  "targets": {
    "up_10": {
      "metrics": { "n": 40000, "hit_rate": 0.55, "brier": 0.242, "auc": 0.56 },
      "periods": [{ "period": "2021Q1", "n": 3000, "hit_rate": 0.54, "auc": 0.55 }]
    }
  },
  "backtest": {
    "strategy":    { "total_return": 0.35, "annualized": 0.06, "max_drawdown": -0.18, "n_trades": 120, "hit_rate": 0.55 },
    "buy_hold_index": { "total_return": 0.28, "annualized": 0.05, "max_drawdown": -0.22 },
    "always_long":    { "total_return": 0.30, "annualized": 0.055, "max_drawdown": -0.20 },
    "credible": true
  }
}
```

## 前端行為約定

- 任一欄位為 `null` / 缺失時，前端顯示「資料暫無」，不報錯。
- `model_valid: false` 時：首頁頂部顯示「模型目前不具可靠預測能力」警示，
  隱藏預測排行榜與風險排行榜；個股頁仍可瀏覽，但會加註警示。
- 所有機率一律附「估計值」標籤；全站不得出現「必漲」「保證獲利」等字眼。
- 排行榜預設依「10日上漲機率」由高到低排序；表頭可切換排序。
- 風險排行榜取 `downside_risk_10` 最小的 10 檔。
- 直接以 `file://` 開啟時，`fetch` 會被瀏覽器阻擋；請用本機伺服器
  （如 `python3 -m http.server`）或部署後的 GitHub Pages 網址瀏覽。
