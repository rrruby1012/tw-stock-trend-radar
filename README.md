# 台股兩週至一個月趨勢雷達

針對台灣 50 成分股，估算未來 10 / 20 個交易日的上漲機率、跑贏加權指數機率、價格區間與下行風險，並以時間序列 walk-forward 驗證與回測檢驗模型是否可信。結果呈現在手機版繁體中文 GitHub Pages 儀表板，每週一至週五台灣時間 18:30 自動更新。

> ⚠️ **研究教育用途，不構成投資建議。** 所有機率皆為模型「估計值」。若模型未通過驗證，網站會明確顯示「模型目前不具可靠預測能力」，不會硬產生推薦名單。本工具不連接券商、不會自動下單。

---

## 目錄

- [資料來源](#資料來源)
- [快速開始](#快速開始)
- [GitHub Actions 自動更新](#github-actions-自動更新)
- [Secrets 設定](#secrets-設定)
- [GitHub Pages 設定](#github-pages-設定)
- [方法說明](#方法說明)
- [專案結構](#專案結構)
- [限制與已知問題](#限制與已知問題)
- [授權](#授權)

---

## 資料來源

優先順序：**FinMind（主力）→ TWSE OpenAPI（備援）→ TPEx OpenAPI（第一版未採用）**。
調查日期 2026-10-18，方式為查閱公開文件與官方 repo，未做違反服務條款的爬取。

### FinMind（主力）

- 官方文件：https://finmind.github.io/ ｜ GitHub：https://github.com/FinMind/FinMind（Apache-2.0）
- API：`https://api.finmindtrade.com/api/v4`
- **授權**：免費方案為**非商業授權**；註冊 token 免費（至 finmindtrade.com 註冊驗證信箱後取得）。本專案僅呈現衍生分析結果、不轉散布原始資料，並標示出處 "FinMind"。
- **呼叫限制**：未登入 300 次/小時；註冊 token 後 600 次/小時；超量回傳 **HTTP 402**（非 429）。台灣 50 檔每週更新約 450 次請求，免費註冊 token 足夠。
- **更新時間**（週一至五）：股價 17:30 → PER 18:00 → 法人買賣 20:00 → 融資融券 21:00。**18:30 排程時當日法人與融資券尚未公布，因此籌碼類特徵一律使用 T-1 資料**，網站會標示各類資料日期。
- **維護時段**：每週日 00:00–07:00 停止服務（排程已避開）。
- 本專案使用的資料集：

| 資料集 | 內容 | 歷史回溯 | 備註 |
|---|---|---|---|
| `TaiwanStockPrice`（未還原）＋`TaiwanStockDividendResult`（除權息參考價） | **自行還原權息**日成交 | 1994-10-01 / 2003-05-01 至今 | 價格主來源。注意：FinMind 現成的 `TaiwanStockPriceAdj` 需付費等級，免費方案會被拒絕（已實測）；本專案改以免費資料集自行做向後還原（除權息日因子 r = 參考價/除權息前收盤價，之前交易日累乘），邏輯等同且已驗證無跳空 |
| `TaiwanStockInstitutionalInvestorsBuySellWide` | 三大法人買賣超（寬表） | 2005-01-01 至今 | 自營商欄位 2014-12-01 前後定義不同，程式已合併處理 |
| `TaiwanStockMarginPurchaseShortSale` | 融資融券餘額 | 2001-01-01 至今 | — |
| `TaiwanStockMonthRevenue` | 月營收 | 2002-02-01 至今 | `create_time` 非公告時間；回測採保守規則「月底後第 10 日」視為可取得 |
| `TaiwanStockFinancialStatements` | 綜合損益表 | 1990-03-01 至今 | EPS 為單季原始值不可直接跨季相加；採保守規則「季底後 75 日」視為可取得 |
| `TaiwanStockTotalReturnIndex`（data_id=TAIEX） | 加權**報酬指數**（含息） | 2003-01-01 至今 | 「跑贏大盤」比較基準 |
| `TaiwanStockTradingDate` | 交易日曆 | — | 判斷非交易日 |
| `TaiwanStockInfo` | 股票總覽（名稱、市場別） | 現況表 | 取每檔最新 date 列 |

### 臺灣證券交易所 OpenAPI（備援）

- 文件：https://openapi.twse.com.tw/v1/swagger.json ｜ Base：`https://openapi.twse.com.tw/v1/`
- **授權**：免 token、匿名可呼叫；官方未公開限流數字，實務上需節流（本專案僅在 FinMind 失敗時呼叫）。
- **更新時間**：盤後陸續（約 14:00–20:00 依報表）。
- **關鍵限制**：
  - 日成交資訊為**未還原**價格，除權息日會跳空，僅作備援顯示用，不得直接用於報酬計算。
  - 多數彙總報表僅提供**最新交易日**，不適合做 5 年歷史建庫。
- 備援用端點（端點 ID 建議以 swagger.json 核對）：`STOCK_DAY_ALL`（全體收盤行情）、`TWT48U`（三大法人日報）、`MI_MARGN`（融資融券彙總）。

### 櫃買中心 OpenAPI（第一版未採用）

- 文件：https://www.tpex.org.tw/openapi/
- 免 token；但多為「僅最新一日」快照，歷史回溯能力弱。
- 台灣 50 成分股全為上市公司，第一版用不到；未來若納入上櫃股，優先用 FinMind 同一資料集（已含上櫃），不必另接。

### 實作陷阱（已處理）

1. `TaiwanStockPrice` 無成交價之日開高低收存為 0——本專案改用 Adj 表並以 `close > 0` 過濾。
2. 法人自營商欄位年代切分——程式自動合併 `Dealer` / `Dealer_self` / `Dealer_Hedging`。
3. 月營收無法嚴格 PIT——採「月底後第 10 日」保守規則，並誠實揭露。
4. 財報 EPS 不可跨季直接相加——只用單季年增率與毛利率。
5. FinMind 免費方案查個股必須帶 `data_id`（逐檔查）；配額用盡回 402，程式會重試並在連續失敗時保留前次結果。

---

## 快速開始

```bash
# 1. 安裝依賴（Python 3.11 建議）
pip install -r requirements.txt

# 2. （選用）設定 FinMind token，提高 API 上限
export FINMIND_TOKEN="你的token"   # 未設定時以匿名模式執行

# 3. 執行更新（抓取→特徵→驗證→預測→回測→產生網站）
python scripts/update.py

# 4. 本地預覽網站
python3 -m http.server -d docs 8000
# 瀏覽器開啟 http://localhost:8000
```

測試：

```bash
python -m pytest tests/ -q
```

只更新特定股票（測試用）：

```bash
python scripts/update.py --symbols 2330,2317
```

---

## GitHub Actions 自動更新

`.github/workflows/update.yml`：

- **排程**：每週一至週五 10:30 UTC（= 台灣時間 18:30），cron：`30 10 * * 1-5`
- **手動執行**：Actions 頁面 →「更新趨勢雷達」→ Run workflow
- **非交易日**：無新資料時正常結束、不報錯、不提交
- **API 暫時失敗**：保留前一次正確結果，網站顯示「資料可能延遲」，不發布空白或錯誤數據
- 更新成功後自動提交 `docs/` 並推送，GitHub Pages 隨即更新

## Secrets 設定

| Secret | 必填 | 說明 |
|---|---|---|
| `FINMIND_TOKEN` | 否 | FinMind API token。未設定時以匿名模式執行（300 次/小時）；設定後上限 600 次/小時。**絕不寫入程式碼**，只從環境變數讀取。 |

設定位置：repo → Settings → Secrets and variables → Actions → New repository secret。
申請方式：至 finmindtrade.com 免費註冊並驗證信箱，登入後在使用者資訊頁取得。

## GitHub Pages 設定（只需做一次）

1. repo → Settings → Pages
2. Build and deployment → Deploy from a branch
3. Branch 選 `main`、資料夾選 `/docs` → Save
4. 網址形如 `https://<你的帳號>.github.io/tw-stock-trend-radar/`

---

## 方法說明

### 預測目標

- 10 / 20 個交易日後報酬率 > 0 的機率（估計值）
- 10 / 20 個交易日後跑贏加權**報酬**指數的機率（估計值）
- 未來 10 / 20 個交易日的合理價格區間（80% 區間，常態近似）與下行風險（5% VaR）

### 特徵（全部嚴格 point-in-time）

5/10/20/60/120 日報酬率、均線乖離與斜率、量能變化、波動度、60 日最大回撤、價格缺口、RSI(14)、MACD、外資/投信/自營商買賣超（T-1）、融資融券變化（T-1）、相對大盤強弱、月營收年增率（月底+10日規則）、單季稅後淨利年增率與毛利率（季底+75日規則）。第一版不使用 LLM 讀新聞。

### 模型與驗證

- **基準**：全部猜上漲（歷史上漲比例）、持有大盤、簡單動能規則（20 日報酬 > 0）
- **ML 模型**：邏輯迴歸（標準化＋`class_weight=balanced`，可解釋）＋ isotonic 機率校準
- **驗證**：時間序列 walk-forward（擴張窗口、每 63 交易日重訓、20 天 embargo 防標籤重疊洩漏），絕不隨機切割
- **可信度門檻**：樣本外 AUC 優於基準 ≥ 0.02 且 Brier score 優於基準 ≥ 0.005 且樣本數 ≥ 500，否則網站顯示「模型目前不具可靠預測能力」
- **回測**：虛擬交易，含手續費 0.1425%（可設折扣）、證交稅 0.3%、滑價 10bps；參數集中於 `config.yaml`

### 趨勢判定

10 日與 20 日上漲機率平均 ≥ 0.58 → 偏多；≤ 0.42 → 偏空；其餘為盤整。

---

## 專案結構

```
tw-stock-trend-radar/
├── README.md                  本文件
├── requirements.txt
├── config.yaml                中央參數（模型、回測成本、排程）
├── watchlist.csv              分析名單（可自行增減股票代號）
├── LICENSE                    MIT ＋ 資料授權說明
├── src/
│   ├── config.py              設定載入
│   ├── datasources/           FinMind / TWSE 客戶端
│   ├── fetch.py               抓取編排與 parquet 快取
│   ├── universe.py            名單載入與排除規則
│   ├── features.py            特徵工程（嚴格 PIT）
│   ├── labels.py              標籤建立
│   ├── models.py              基準＋邏輯迴歸＋校準＋價格區間
│   ├── validation.py          walk-forward 驗證與可信度檢驗
│   ├── backtest.py            虛擬交易回測（含成本）
│   └── sitegen.py             網站 JSON 產生器
├── scripts/update.py          主更新流程
├── tests/                     pytest 自動測試
├── .github/workflows/         自動更新 workflow
├── docs/                      GitHub Pages（index.html、stock.html、data/*.json）
└── data/cache/                原始快取（不進版控）
```

---

## 限制與已知問題

1. **18:30 排程的籌碼為 T-1**：FinMind 法人資料 20:00、融資券 21:00 才公布，18:30 執行時只能用前一交易日的籌碼。網站會標示各類資料日期。若希望籌碼為當日，可將排程改為 21:30 之後（需修改 workflow 的 cron）。
2. **月營收 PIT 為保守估計**：FinMind 未提供嚴格公告時間戳，回測以「月底後第 10 日」為可用日，可能晚於實際公告日數天。
3. **還原權息基準漂移**：FinMind Adj 表以後復權計算，基準日會隨時間前移；入庫後固定使用入庫版本，不重算歷史。
4. **台灣 50 名單**：第一版以 `watchlist.csv` 為準（2026-10 手動整理）；成分股調整時請手動更新，或未來串接 TWSE 成分股 API。
5. **模型可能不可信**：台股短週期預測難度高，若 walk-forward 未通過門檻，網站會誠實顯示警示而不給推薦名單——這是設計，不是 bug。
6. **免費 FinMind 為非商業授權**：請勿將本專案用於商業用途；網站需保留 "FinMind" 出處標示（已內建於頁尾）。

---

## 授權

程式碼採用 MIT License（見 LICENSE）。資料來源授權見 LICENSE 末段與本文件「資料來源」章節。
