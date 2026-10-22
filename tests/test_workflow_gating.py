"""工作流閘門測試：同日 --force 產生不同內容時，失敗必須發生在任何
commit/push 之前，staging 漂移不得部署上公開站。

保證鏈：
1. update.py 在 ledger 衝突時 return 1（執行更新步驟失敗）。
2. update.yml 的提交步驟排在更新步驟之後，且無 `if: always()`、
   無 continue-on-error——GitHub Actions 預設跳過失敗步驟的後續步驟，
   因此失敗 run 不會產生任何 commit/push。
3. 已凍結的 ledger 在衝突後逐位元不變（見 test_ledger）。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YML = (ROOT / ".github" / "workflows" / "update.yml").read_text(encoding="utf-8")
UPDATE = (ROOT / "scripts" / "update.py").read_text(encoding="utf-8")


def test_提交步驟在更新步驟之後且無always():
    i_update = YML.index("python scripts/update.py")
    i_commit = YML.index("git commit")
    assert i_update < i_commit, "提交步驟必須在更新步驟之後"
    assert "if: always()" not in YML, "不得有 if: always() 繞過失敗閘門"
    assert "continue-on-error" not in YML, "不得有 continue-on-error 繞過失敗閘門"


def test_衝突時update回傳失敗():
    assert "except ledger_mod.LedgerConflict" in UPDATE
    seg = UPDATE.split("except ledger_mod.LedgerConflict")[1].split("# 10.")[0]
    assert "return 1" in seg, "ledger 衝突時 update.py 必須回傳非零結束碼"
