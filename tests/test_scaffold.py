"""scaffold.py and install.sh; the deployed project's own smoke behind LH_E2E=1."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD = ROOT / "scripts" / "scaffold.py"


def scaffold(project: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCAFFOLD), str(project), *args], capture_output=True, text=True, check=False
    )


def test_renders_package_name(tmp_path):
    r = scaffold(tmp_path, "--pkg", "demo")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "src" / "demo" / "model" / "lightning.py").exists()
    assert (tmp_path / ".gitignore").exists()
    leftovers = [
        p for p in tmp_path.rglob("*") if "__pkg__" in str(p) or (p.is_file() and b"__pkg__" in p.read_bytes())
    ]
    assert leftovers == []
    assert "_target_: demo.model.lightning.Regressor" in (tmp_path / "configs" / "model" / "default.yaml").read_text()
    assert 'name = "demo"' in (tmp_path / "pyproject.toml").read_text()
    assert not list(tmp_path.rglob(".ruff_cache"))


def test_existing_files_kept(tmp_path):
    (tmp_path / "pyproject.toml").write_text("mine\n")
    r = scaffold(tmp_path, "--pkg", "demo")
    assert r.returncode == 0
    assert "exists       pyproject.toml" in r.stdout
    assert (tmp_path / "pyproject.toml").read_text() == "mine\n"
    assert (tmp_path / "scripts" / "train.py").exists()


def test_dry_run_writes_nothing(tmp_path):
    r = scaffold(tmp_path / "p", "--pkg", "demo", "--dry-run")
    assert r.returncode == 0
    assert "would write  scripts/train.py" in r.stdout
    assert not (tmp_path / "p").exists()


def test_compare(tmp_path):
    scaffold(tmp_path, "--pkg", "demo")
    (tmp_path / "configs" / "train.yaml").write_text("changed\n")
    (tmp_path / "scripts" / "train.py").unlink()
    out = scaffold(tmp_path, "--pkg", "demo", "--compare").stdout
    assert "differs      configs/train.yaml" in out
    assert "missing      scripts/train.py" in out
    assert "same         pyproject.toml" in out
    assert not (tmp_path / "scripts" / "train.py").exists()


@pytest.mark.parametrize("name", ["My-Pkg", "1abc", "class", "__pkg__"])
def test_bad_package_name(tmp_path, name):
    r = scaffold(tmp_path, "--pkg", name)
    assert r.returncode == 2
    assert not any(tmp_path.iterdir())


def test_python_default_pinned(tmp_path):
    assert scaffold(tmp_path, "--pkg", "demo").returncode == 0
    assert (tmp_path / ".python-version").read_text() == "3.12\n"
    pyproject = (tmp_path / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.12,<3.14"' in pyproject
    assert 'pythonVersion = "3.12"' in pyproject


def test_python_chosen(tmp_path):
    assert scaffold(tmp_path, "--pkg", "demo", "--python", "3.13").returncode == 0
    assert (tmp_path / ".python-version").read_text() == "3.13\n"
    assert 'requires-python = ">=3.13,<3.14"' in (tmp_path / "pyproject.toml").read_text()
    # --compare takes the project's own version
    out = scaffold(tmp_path, "--pkg", "demo", "--compare").stdout
    assert "same         pyproject.toml" in out
    assert "same         .python-version" in out


@pytest.mark.parametrize(
    ("version", "why"),
    [("3.14", "Hydra 1.3 fails"), ("3.10", "older than"), ("3.99", "newer than"), ("3.12.1", "without the patch")],
)
def test_python_refused(tmp_path, version, why):
    r = scaffold(tmp_path, "--pkg", "demo", "--python", version)
    assert r.returncode == 2
    assert why in r.stderr and "supported:" in r.stderr or why == "without the patch"
    assert not any(tmp_path.iterdir())


def test_install_project_copy(tmp_path):
    """Anchored excludes: templates/project/ keeps its pyproject.toml, tests/ and .gitignore."""
    subprocess.run(["bash", str(ROOT / "install.sh"), "--project"], cwd=tmp_path, check=True, stdin=subprocess.DEVNULL)
    skill = tmp_path / ".agents" / "skills" / "lightning-hydra"
    for rel in [
        "SKILL.md",
        "scripts/scaffold.py",
        "references/torch-gpu.md",
        "templates/project/pyproject.toml",
        "templates/project/tests/test_train.py",
        "templates/project/.gitignore",
    ]:
        assert (skill / rel).is_file(), rel
    for rel in ["docs", "tests", "probes", "pyproject.toml", ".git"]:
        assert not (skill / rel).exists(), rel
    assert list(skill.rglob(".ruff_cache")) == []
    link = tmp_path / ".claude" / "skills" / "lightning-hydra"
    assert link.is_symlink() and (link / "SKILL.md").is_file()


@pytest.mark.skipif(
    not os.environ.get("LH_E2E"),
    reason="LH_E2E=1: uv lock/sync from data.pyg.org, ruff, pyright and pytest in the project; minutes",
)
def test_deployed_project_smoke(tmp_path):
    uv = shutil.which("uv")
    assert uv, "uv not found"
    assert scaffold(tmp_path, "--pkg", "demo").returncode == 0
    subprocess.run([uv, "sync"], cwd=tmp_path, check=True)
    for check in (["ruff", "check", "."], ["ruff", "format", "--check", "."], ["pyright"], ["pytest", "-q"]):
        subprocess.run([uv, "run", *check], cwd=tmp_path, check=True)
