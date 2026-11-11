"""舊存檔顯示相容測試（2026-11-11；項目 2）。

驗證：
- 交易日旗標三態：true→交易日、false→非交易日、缺失→未記錄；
  舊存檔缺 is_trading_day 時絕不可顯示為「非交易日」。
- 股價日期：新 meta 取 data_dates.price；舊 meta 只有頂層 price_date
  時以後者為準；兩者皆無時回傳 null（顯示「資料暫無」，不得補造）。
- 以 Node 實際執行 app.js 的 helper（非僅字串比對），並以真實舊存檔
  meta_2026-09-01.json 為例驗證。
- reconstructed／prospective 分類邏輯不動（見 test_frontend_honesty）。

注意：本測試不改寫任何歷史 JSON；測試產物只放 tmp_path。
"""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
APPJS = (ROOT / "docs" / "assets" / "app.js").read_text(encoding="utf-8")
APPJS_PATH = ROOT / "docs" / "assets" / "app.js"
OLD_META = ROOT / "docs" / "data" / "archive" / "meta_2026-09-01.json"

# 以 Node 執行 app.js 並呼叫 helper 的小腳本（寫入 tmp_path 再執行）。
NODE_SCRIPT = r"""
const fs = require("fs");
const [,, appPath, oldMetaPath] = process.argv;
const src = fs.readFileSync(appPath, "utf8");
// app.js 以 IIFE 回傳 App；以 eval 取回（載入時不執行 document/window 操作）。
const App = eval(src + "\nApp;");
const failures = [];
function check(name, got, want) {
  if (got !== want) failures.push({ name, got, want });
}
// 交易日三態
check("true", App.tradingDayState({ is_trading_day: true }), "trading");
check("false", App.tradingDayState({ is_trading_day: false }), "non_trading");
check("missing", App.tradingDayState({}), "unknown");
check("null", App.tradingDayState({ is_trading_day: null }), "unknown");
check("no meta", App.tradingDayState(null), "unknown");
check("string", App.tradingDayState({ is_trading_day: "yes" }), "unknown");
// 股價日期
check("new", App.displayPriceDate({ data_dates: { price: "2026-11-10" } }), "2026-11-10");
check("old", App.displayPriceDate({ price_date: "2026-09-01" }), "2026-09-01");
check("new wins", App.displayPriceDate({ data_dates: { price: "2026-11-10" }, price_date: "2026-09-01" }), "2026-11-10");
check("none", App.displayPriceDate({}), null);
check("none meta", App.displayPriceDate(null), null);
// 真實舊存檔（2026-09-01）：只有頂層 price_date，無 is_trading_day
const oldMeta = JSON.parse(fs.readFileSync(oldMetaPath, "utf8"));
check("real old tradingDayState", App.tradingDayState(oldMeta), "unknown");
check("real old displayPriceDate", App.displayPriceDate(oldMeta), "2026-09-01");
if (failures.length) {
  console.log(JSON.stringify({ ok: false, failures }, null, 2));
  process.exit(1);
}
console.log(JSON.stringify({ ok: true }));
"""


def test_交易日旗標改為三態且舊二元邏輯移除():
    assert "tradingDayState" in APPJS
    assert "tradingDayState" in INDEX
    assert "交易日未記錄" in INDEX
    # 舊的二元寫法必須消失：缺欄位時不得直接當非交易日
    assert 'meta.is_trading_day ? "交易日" : "非交易日"' not in INDEX
    assert 'meta.is_trading_day ? "ok" : "muted"' not in INDEX


def test_股價日期改讀price_date相容舊存檔():
    assert "displayPriceDate" in APPJS
    assert "displayPriceDate" in INDEX
    # 其餘類別缺值時仍顯示資料暫無（不得補造）：renderFooter 不得對
    # institutional/margin/revenue/financials 做 price_date 回退
    seg = INDEX.split("function renderFooter")[1].split("function renderInvalidView")[0]
    assert "displayPriceDate" in seg
    assert 'p[0] === "price"' in seg


def test_舊存檔分類與重建標示不動():
    # 相容修正不得動到分類邏輯：舊存檔仍為 reconstructed，不得升格
    assert "resolveArchiveClass" in APPJS
    assert "事後重建，不代表當日實際發布" in INDEX


def test_node實際執行helper(tmp_path):
    """以 Node 載入 app.js，實際呼叫 tradingDayState / displayPriceDate。"""
    script = tmp_path / "check_archive_compat.js"
    script.write_text(NODE_SCRIPT, encoding="utf-8")
    r = subprocess.run(
        ["node", str(script), str(APPJS_PATH), str(OLD_META)],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"Node 檢查失敗：{r.stdout}\n{r.stderr}"
    assert json.loads(r.stdout)["ok"] is True


def test_真實舊存檔結構仍為舊形狀():
    """回歸錨點：2026-09-01 存檔確實是舊結構（只有頂層 price_date），
    本修正不得以重寫歷史 JSON 來「修好」它。"""
    meta = json.loads(OLD_META.read_text(encoding="utf-8"))
    assert meta["price_date"] == "2026-09-01"
    assert "data_dates" not in meta
    assert "is_trading_day" not in meta
    assert "sources" not in meta
    assert meta.get("reconstructed") is True


# ---------- 無效 JSON 存檔明示錯誤（2026-11-11 補充） ----------

STOCK = (ROOT / "docs" / "stock.html").read_text(encoding="utf-8")

# 以 Node 起本地 HTTP 伺服器驗證 fetchJSON：壞內容為內嵌的非標準 JSON
# （重現 2026-10-14/15 存檔的 close_raw: NaN 寫法），不依賴本機快照的
# 檔案內容（本機 docs/data 為舊快照，與乾淨 main 的該檔內容不同）。
NODE_FETCH_SCRIPT = r"""
const fs = require("fs");
const http = require("http");
const [,, appPath] = process.argv;
const src = fs.readFileSync(appPath, "utf8");
const App = eval(src + "\nApp;");
// 重現乾淨 main predictions_2026-10-14.json 的失效模式：close_raw 為裸 NaN
const badBody = '[{"symbol":"3711","name":"日月光投控","close":100.0,"close_raw":NaN,"p_up_10":0.6}]';
const server = http.createServer((req, res) => {
  if (req.url === "/bad.json") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(badBody);
  } else if (req.url === "/good.json") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify([{ symbol: "2330", close_raw: 100.0 }]));
  } else {
    res.writeHead(404);
    res.end("not found");
  }
});
server.listen(0, "127.0.0.1", async () => {
  const port = server.address().port;
  const base = "http://127.0.0.1:" + port;
  const failures = [];
  function check(name, got, want) {
    if (got !== want) failures.push({ name, got, want });
  }
  const r1 = await App.fetchJSON(base + "/bad.json");
  check("bad ok", r1.ok, false);
  check("bad invalid_json", r1.invalid_json, true);
  const r2 = await App.fetchJSON(base + "/good.json");
  check("good ok", r2.ok, true);
  check("good data", r2.data[0].close_raw, 100.0);
  const r3 = await App.fetchJSON(base + "/missing.json");
  check("404 ok", r3.ok, false);
  check("404 invalid_json", r3.invalid_json, false);
  server.close();
  if (failures.length) {
    console.log(JSON.stringify({ ok: false, failures }, null, 2));
    process.exit(1);
  }
  console.log(JSON.stringify({ ok: true }));
});
"""


def test_fetchJSON區分無效JSON與一般讀取失敗():
    assert "invalid_json" in APPJS
    # 無效 JSON 時首頁與個股頁皆明示「含有無效數值」，不得誤報為尚未產生
    assert "含有無效數值" in INDEX
    assert "含有無效數值" in STOCK
    assert "原始存檔保留未更動" in INDEX
    assert "原始存檔保留未更動" in STOCK
    # 個股頁在 meta 無效時必須停止，不得以空 meta 繼續渲染
    seg = STOCK.split("Promise.all([App.fetchJSON(metaPath)")[1].split("App.fetchJSON(predPath)")[0]
    assert "mr.invalid_json" in seg


def test_node_fetchJSON無效JSON(tmp_path):
    """以 Node 實際執行 fetchJSON：裸 NaN 內容 → ok:false 且 invalid_json:true；
    合法 JSON → ok:true；404 → ok:false 且 invalid_json:false。"""
    script = tmp_path / "check_fetch_invalid_json.js"
    script.write_text(NODE_FETCH_SCRIPT, encoding="utf-8")
    r = subprocess.run(
        ["node", str(script), str(APPJS_PATH)],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, f"Node 檢查失敗：{r.stdout}\n{r.stderr}"
    assert json.loads(r.stdout)["ok"] is True
