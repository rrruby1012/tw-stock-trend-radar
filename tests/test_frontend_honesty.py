"""前端誠實顯示回歸測試（靜態檢查，沿用 test_stock_page 模式）。

鎖定規格（2026-10-22）：
- credible=false 預設畫面：無精確百分比/排名/機率條/個股方向；
  定性標示只進「實驗區・未驗證」且不可排序。
- reconstructed 三處標示（首頁橫幅、存檔選單、個股頁橫幅），文字含
  「事後重建，不代表當日實際發布」。
- 資料分類靠欄位 + ledger 證明（app.js resolveArchiveClass / fetchLedger）。
- 個股頁模擬紀錄明確標示 walkforward_sim，非實際發布。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
STOCK = (ROOT / "docs" / "stock.html").read_text(encoding="utf-8")
APPJS = (ROOT / "docs" / "assets" / "app.js").read_text(encoding="utf-8")


def test_首頁未驗證預設不提供方向():
    assert "模型未通過驗證，暫不提供個股方向" in INDEX
    assert "renderInvalidView" in INDEX
    # 舊研究表（含精確機率欄）的入口必須移除
    assert "10日上漲機率（未驗證）" not in INDEX
    assert "renderResearchTable" not in INDEX


def test_實驗區存在且不可排序():
    assert "實驗區・未驗證" in INDEX
    assert "lab-section" in INDEX
    # 實驗區表格欄位全部 sortable: false —— 取 renderInvalidView 段落檢查
    seg = INDEX.split("function renderInvalidView")[1].split("function renderForward")[0]
    assert "sortable: true" not in seg, "實驗區表格不得可排序"
    assert "probCell" not in seg, "實驗區不得顯示精確機率"
    assert "fmtPct" not in seg, "實驗區不得顯示精確百分比"


def test_reconstructed標示三處齊全():
    assert "事後重建，不代表當日實際發布" in INDEX  # 首頁橫幅
    assert "事後重建" in INDEX and "正式發布" in INDEX  # 存檔選單標記
    assert "事後重建，不代表當日實際發布" in STOCK  # 個股頁橫幅
    # 橫幅改讀正確欄位（舊 bug：reconstructed_note 與資料的 note 不一致）
    assert "meta.note || meta.reconstructed_note" in INDEX


def test_分類靠欄位與ledger證明():
    assert "resolveArchiveClass" in APPJS
    assert "fetchLedger" in APPJS
    assert "data/ledger/index.json" in APPJS
    assert "data/ledger/corrections.json" in APPJS
    # 命中判定在首頁只限 prospective
    assert 'clsInfo.cls === "prospective"' in INDEX
    assert 'clsInfo && clsInfo.cls === "prospective"' in STOCK


def test_個股頁未驗證走實驗區且不顯示精確機率():
    assert "實驗區・未驗證" in STOCK
    assert "renderLab" in STOCK
    # 未驗證分支走實驗區、驗證分支才走精確機率（以主流程 labCard 為錨點切分）
    assert "var labCard = renderLab(p);" in STOCK
    seg = STOCK.split("var labCard = renderLab(p);")[1].split("renderActual")[0]
    assert "renderProbs(p)" not in seg.split("} else {")[0], "未驗證分支不得渲染精確機率"
    assert "renderProbs(p);" in seg.split("} else {")[1], "驗證分支應渲染精確機率"
    # 價格區間/下行風險（精確估計）未驗證時不顯示
    assert 'if (window.__modelValid !== false) renderInterval(p);' in STOCK


def test_個股頁模擬紀錄標示():
    assert "樣本外模擬紀錄（walk-forward）" in STOCK
    assert "walkforward_sim" in STOCK
    assert "模擬命中率" in STOCK
    assert "非本站實際發布" in STOCK
    # 舊標題不得再出現（會被誤認為真實績）
    assert "預測紀錄與歷史表現" not in STOCK
    assert "10日上漲預測歷史命中率" not in STOCK


def test_前瞻累積狀態有顯示():
    assert "真實前瞻紀錄（prospective）" in INDEX
    assert "資料累積中" in INDEX
    assert "prospective_progress" in INDEX


def test_個股頁規則式訊號未驗證時全進實驗區():
    # renderSignals 支援 targetCard：未驗證分支必須把訊號塞進實驗區卡片，
    # 預設區不得獨立渲染「訊號與判斷條件」section
    assert "function renderSignals(p, targetCard)" in STOCK
    assert "renderSignals(p, labCard)" in STOCK
    seg = STOCK.split("var labCard = renderLab(p);")[1].split("renderActual")[0]
    assert "renderSignals(p, labCard)" in seg, "未驗證分支未把訊號收進實驗區"
    # 有利/風險訊號的渲染只能經 renderSignals（不得有第二條直出路徑）
    assert STOCK.count("sigList(p.bull_signals") == 1
    assert STOCK.count("sigList(p.bear_signals") == 1
