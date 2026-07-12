#!/usr/bin/env python3
from __future__ import annotations

import argparse
import py_compile
import re
import shutil
from pathlib import Path


NEW_RENDER_BLOCK = r'''_PROMPT_FIELD_RE = re.compile(
    r"\{([A-Za-z_][A-Za-z0-9_]*)\}"
)


def render(template: str, **kwargs: Any) -> str:
    """Replace named placeholders while preserving literal JSON braces."""
    values = {
        key: (
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
            )
            if isinstance(value, (dict, list))
            else str(value if value is not None else "")
        )
        for key, value in kwargs.items()
    }

    required = set(_PROMPT_FIELD_RE.findall(template))
    missing = sorted(required.difference(values))
    if missing:
        raise KeyError(
            "Missing prompt render values: "
            + ", ".join(missing)
        )

    return _PROMPT_FIELD_RE.sub(
        lambda match: values[match.group(1)],
        template,
    )


'''


def patch_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    original = path.read_text(encoding="utf-8")

    if "_PROMPT_FIELD_RE = re.compile" in original:
        print(f"Already patched: {path}")
        return

    updated = original

    if not re.search(r"(?m)^import re\s*$", updated):
        updated, import_count = re.subn(
            r"(?m)^import json\s*$",
            "import json\nimport re",
            updated,
            count=1,
        )
        if import_count != 1:
            raise SystemExit(
                "Could not locate a standalone `import json` line."
            )

    render_pattern = re.compile(
        r"def render\(template:\s*str,\s*\*\*kwargs:\s*Any\)\s*->\s*str:"
        r".*?"
        r"(?=^DIMMEM_V1_EXTRACTION_SYSTEM\s*=)",
        flags=re.DOTALL | re.MULTILINE,
    )
    updated, render_count = render_pattern.subn(
        NEW_RENDER_BLOCK,
        updated,
        count=1,
    )
    if render_count != 1:
        raise SystemExit(
            "Could not locate exactly one old render() block. "
            "No file was changed."
        )

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")

    try:
        py_compile.compile(str(path), doraise=True)
    except Exception:
        shutil.copy2(backup, path)
        raise

    print(f"Patched: {path}")
    print(f"Backup:  {backup}")
    print("Python compile check: passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="longmemeval/graph_memory_v2/prompts.py",
        help="Path to prompts.py",
    )
    args = parser.parse_args()
    patch_file(Path(args.path))


if __name__ == "__main__":
    main()
