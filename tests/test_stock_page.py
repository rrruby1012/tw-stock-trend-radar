"""stock.html 回歸測試：歷史結果須同時接受布林 true/false 與數字 1/0。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOCK_HTML = (ROOT / "docs" / "stock.html").read_text(encoding="utf-8")


def test_歷史結果接受布林值():
    # 必須有正規化函式同時處理 true/false 與 1/0
    assert "normActual" in STOCK_HTML, "stock.html 缺少 normActual 正規化函式"
    assert "v === true" in STOCK_HTML and "v === false" in STOCK_HTML, (
        "stock.html 未處理布林 true/false")
    # 舊的寫法（只比對 1/0）不得再單獨作為篩選條件
    assert "return r[2] === 1 || r[2] === 0" not in STOCK_HTML, (
        "stock.html 仍只接受數字 1/0，布林資料會被誤判為無紀錄")


def test_close_raw缺失時不得標示官方原始():
    # 缺失時應顯示警告文字，不得直接標示「官方原始」
    assert "官方原始價缺失" in STOCK_HTML, (
        "stock.html 未處理 close_raw 缺失的警告標示")
    assert "hasRaw" in STOCK_HTML


def test_還原權息說明不混淆():
    # 標章文字須說明上方顯示的是官方原始收盤，模型才用還原價
    assert "上方顯示仍為官方原始收盤價" in STOCK_HTML, (
        "還原權息標章未釐清顯示值與模型計算值的區別")
