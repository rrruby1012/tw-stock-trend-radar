"""中央設定載入：讀取 config.yaml，環境變數可覆寫敏感值。"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    cfg_path = Path(path) if path else _DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 環境變數覆寫（token 絕不寫入檔案）
    cfg.setdefault("data", {})["finmind_token"] = os.environ.get(
        cfg["data"].get("finmind_token_env", "FINMIND_TOKEN"), ""
    )
    return cfg


def get(cfg: dict, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
