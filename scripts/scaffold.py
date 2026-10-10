#!/usr/bin/env python3
"""Deploy the lightning-hydra skeleton into a project, or compare a project with it.

    python3 scaffold.py <project> --pkg <name>                 # write missing files
    python3 scaffold.py <project> --pkg <name> --python 3.13   # another Python
    python3 scaffold.py <project> --pkg <name> --dry-run       # show what would be written
    python3 scaffold.py <project> --pkg <name> --compare       # project vs. current templates

Renders templates/project/: the marker __pkg__ becomes <name> in paths and in
file contents. The templates carry the default Python version in
.python-version, requires-python and pyright's pythonVersion; --python
replaces it there. --python is checked against the versions the skeleton runs on; one
outside them is refused with the reason. Without --python: the project's
.python-version if it has one, else the default. Existing files are never overwritten; they are listed as
`exists` and merged by hand (pyproject.toml of an existing project, say).
--compare writes nothing and prints `same`, `differs` or `missing` per file:
the list to look at when updating a deployed project.

Exit 0 on success, 2 on bad arguments. Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import keyword
import re
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "project"
MARKER = "__pkg__"
IGNORED = {".ruff_cache", "__pycache__", ".pytest_cache", ".DS_Store", ".venv"}

# Python versions the skeleton runs on, and why the others do not. uv.lock pins
# packages, not the interpreter: .python-version does, locally and in remote
# tasks, where uv would otherwise download the newest version requires-python
# allows.
DEFAULT_PYTHON = "3.12"
SUPPORTED_PYTHON = ("3.11", "3.12", "3.13")
UNSUPPORTED_PYTHON = {
    "3.14": "Hydra 1.3 fails on it before reading the config (argparse: badly formed help string)",
}
OLDEST_REASON = "older than the oldest version the skeleton is tested on"


def check_pkg(name: str) -> str | None:
    """Why `name` is not a usable package name, or None."""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        return "lowercase letters, digits and underscores, not starting with a digit"
    if keyword.iskeyword(name):
        return "a Python keyword"
    if name == MARKER:
        return "the template marker itself"
    return None


def check_python(version: str) -> str | None:
    """Why the skeleton cannot use Python `version`, or None."""
    if not re.fullmatch(r"3\.\d+", version):
        return "a minor version such as 3.12, without the patch"
    if version in SUPPORTED_PYTHON:
        return None
    supported = ", ".join(SUPPORTED_PYTHON)
    minor = int(version.split(".")[1])
    if minor < int(SUPPORTED_PYTHON[0].split(".")[1]):
        return f"{OLDEST_REASON}; supported: {supported}"
    why = UNSUPPORTED_PYTHON.get(version, "newer than the newest version the skeleton is tested on")
    return f"{why}; supported: {supported}"


def python_below() -> str:
    """The first version after the supported ones, for requires-python."""
    return f"3.{int(SUPPORTED_PYTHON[-1].split('.')[1]) + 1}"


def set_python(rel: Path, content: bytes, python: str) -> bytes:
    """The default Python version of the templates replaced by `python`."""
    if rel == Path(".python-version"):
        return f"{python}\n".encode()
    if rel == Path("pyproject.toml"):
        text = content.decode()
        for old, new in [
            (
                f'requires-python = ">={DEFAULT_PYTHON},<{python_below()}"',
                f'requires-python = ">={python},<{python_below()}"',
            ),
            (f'pythonVersion = "{DEFAULT_PYTHON}"', f'pythonVersion = "{python}"'),
        ]:
            if old not in text:
                raise SystemExit(
                    f"templates/project/pyproject.toml: no line {old!r}; scaffold.py and the template disagree"
                )
            text = text.replace(old, new)
        return text.encode()
    return content


def render(pkg: str, python: str = DEFAULT_PYTHON) -> dict[Path, bytes]:
    """Relative path in the project -> content, with the markers replaced."""
    out: dict[Path, bytes] = {}
    for src in sorted(TEMPLATE.rglob("*")):
        rel = src.relative_to(TEMPLATE)
        if src.is_dir() or IGNORED & set(rel.parts) or src.suffix == ".pyc":
            continue
        dest = Path(*(part.replace(MARKER, pkg) for part in rel.parts))
        out[dest] = set_python(rel, src.read_bytes().replace(MARKER.encode(), pkg.encode()), python)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", type=Path, help="project root (created if missing)")
    ap.add_argument("--pkg", required=True, help="Python package name, e.g. affinity")
    ap.add_argument(
        "--python",
        help=f"Python minor version: one of {', '.join(SUPPORTED_PYTHON)} "
        f"(default: the project's .python-version, else {DEFAULT_PYTHON})",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print what would be written, write nothing")
    mode.add_argument("--compare", action="store_true", help="compare the project with the templates")
    args = ap.parse_args(argv)

    if why := check_pkg(args.pkg):
        print(f"error: --pkg {args.pkg!r}: {why}", file=sys.stderr)
        return 2

    root = args.project
    python = args.python
    if python is None:
        pinned = root / ".python-version"
        python = pinned.read_text().strip() if pinned.is_file() else DEFAULT_PYTHON
        python = ".".join(python.split(".")[:2])  # a pinned patch version counts as its minor
    if why := check_python(python):
        print(f"error: Python {python}: {why}", file=sys.stderr)
        return 2

    files = render(args.pkg, python)
    counts: dict[str, int] = {}
    for rel, content in files.items():
        dest = root / rel
        if args.compare:
            state = "missing" if not dest.exists() else "same" if dest.read_bytes() == content else "differs"
        elif dest.exists():
            state = "exists"
        else:
            state = "would write" if args.dry_run else "wrote"
            if not args.dry_run:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
        counts[state] = counts.get(state, 0) + 1
        print(f"{state:12} {rel}")

    print("; ".join(f"{n} {s}" for s, n in counts.items()), file=sys.stderr)
    if counts.get("exists"):
        print("existing files were left as they are: merge them with the templates by hand", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
