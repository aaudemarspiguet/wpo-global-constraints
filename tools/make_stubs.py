"""Regenerate src/ (exercises) from solutions/ (reference).

Every block between
    # >>> SOLUTION: <hint>
    ...
    # <<< SOLUTION
is replaced by `raise NotImplementedError("TODO: <hint>")` at the same indent.

    python tools/make_stubs.py --force    # (re)writes src/wpo from solutions/wpo
    python tools/make_stubs.py --check    # exit 1 if src/ is out of date
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC, SOL = ROOT / "src" / "wpo", ROOT / "solutions" / "wpo"
BEGIN = re.compile(r"^(\s*)# >>> SOLUTION:?\s*(.*)$")
END = re.compile(r"^\s*# <<< SOLUTION\s*$")


def stub_text(text: str) -> str:
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        if not skipping:
            m = BEGIN.match(line)
            if m:
                indent, hint = m.group(1), m.group(2).strip() or "implement me"
                hint = hint.replace('"', "'")
                out.append(f'{indent}raise NotImplementedError("TODO: {hint}")\n')
                skipping = True
            else:
                out.append(line)
        elif END.match(line):
            skipping = False
    if skipping:
        raise ValueError("unterminated SOLUTION block")
    return "".join(out)


def build(dest: pathlib.Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    for f in SOL.rglob("*"):
        if "__pycache__" in f.parts:
            continue
        target = dest / f.relative_to(SOL)
        if f.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif f.suffix == ".py":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(stub_text(f.read_text()))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing src/wpo (DESTROYS your work)")
    args = ap.parse_args()
    if args.check:
        tmp = ROOT / ".stub_check" / "wpo"
        build(tmp)
        stale = [p for p in tmp.rglob("*.py")
                 if not (SRC / p.relative_to(tmp)).exists()
                 or (SRC / p.relative_to(tmp)).read_text() != p.read_text()]
        shutil.rmtree(tmp.parent)
        if stale:
            print("src/ differs from freshly generated stubs:", *stale, sep="\n  ")
            sys.exit(1)
        print("src/ is up to date")
        return
    if SRC.exists() and not args.force:
        print(f"{SRC} exists. Refusing to overwrite your work; pass --force to regenerate.")
        sys.exit(1)
    build(SRC)
    print(f"wrote stubs to {SRC}")


if __name__ == "__main__":
    main()
