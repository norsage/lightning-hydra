"""Smoke: the skeleton builds and runs one batch of train/val/test."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_train_debug(tmp_path):
    subprocess.run(
        [sys.executable, "scripts/train.py", "trainer=debug", f"hydra.run.dir={tmp_path}"],
        cwd=ROOT,
        check=True,
    )
    assert (tmp_path / ".hydra" / "config.yaml").exists()
