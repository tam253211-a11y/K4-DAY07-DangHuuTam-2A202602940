#!/usr/bin/env python3
"""Checkpoint 2 corpus check (checklist in docs/DATA_COLLECTION.md, section 6).

Usage: python scripts/check_corpus.py [data/<ten-chu-de>]

Unlike the one-liner in the lab guide, this parses the frontmatter at the closing
"\\n---\\n" (URLs may contain "---") and strips the quotes the crawler adds to values.
Exit code is 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REQUIRED = ["doc_id", "title", "source_url", "retrieved_at", "document_version", "audience"]
FILTER_FIELDS = ["department", "category", "language"]
AUDIENCES = {"student", "faculty", "staff", "all"}
LEFTOVER_MENU = ["hidden", "Trang chủ", "Tiếng anh", "Tổng lượt truy cập", "Thành viên online", "Phát triển bởi"]


def parse(path: Path) -> tuple[dict[str, str], str]:
    head, body = path.read_text(encoding="utf-8").split("\n---\n", 1)
    meta = {key: value.strip().strip('"') for key, value in re.findall(r"^(\w+):\s*(.+)$", head, re.M)}
    return meta, body


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/thu-vien-vnulib")
    files = sorted(folder.glob("*.md"))
    failures: list[str] = []

    def check(label: str, passed: bool, detail: str = "") -> None:
        print(f"{'OK  ' if passed else 'FAIL'} {label}{(' - ' + detail) if detail else ''}")
        if not passed:
            failures.append(label)

    metas: dict[str, dict[str, str]] = {}
    for path in files:
        meta, body = parse(path)
        metas[path.stem] = meta
        missing = [key for key in REQUIRED if key not in meta]
        problems = []
        if missing:
            problems.append("thiếu " + ",".join(missing))
        if meta.get("doc_id") != path.stem:
            problems.append("doc_id khác tên file")
        if not any(key in meta for key in FILTER_FIELDS):
            problems.append("thiếu trường lọc (department/category/language)")
        if meta.get("audience") not in AUDIENCES:
            problems.append(f"audience không hợp lệ: {meta.get('audience')}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", meta.get("retrieved_at", "")):
            problems.append("retrieved_at không phải YYYY-MM-DD")
        if len(body.strip()) < 200:
            problems.append("nội dung quá ngắn")
        leftovers = [word for word in LEFTOVER_MENU if word in body]
        if leftovers:
            problems.append("còn sót menu/footer: " + ",".join(leftovers))
        check(path.name, not problems, "; ".join(problems))

    check("Số file 5-10", 5 <= len(files) <= 10, f"{len(files)} file")
    check("doc_id không trùng", len({m.get('doc_id') for m in metas.values()}) == len(metas))

    manifest_path = folder / "sources.csv"
    if manifest_path.exists():
        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
        check("sources.csv khớp 1-1 với file .md", sorted(r["doc_id"] for r in rows) == sorted(metas))
        mismatches = [
            f"{row['doc_id']}.{key}"
            for row in rows
            if row["doc_id"] in metas
            for key in ("title", "source_url", "retrieved_at", "document_version")
            if metas[row["doc_id"]].get(key) != row[key]
        ]
        check("sources.csv khớp với frontmatter", not mismatches, ", ".join(mismatches))
    else:
        check("có sources.csv", False)

    audiences: dict[str, int] = {}
    for meta in metas.values():
        audiences[meta.get("audience", "?")] = audiences.get(meta.get("audience", "?"), 0) + 1
    check("audience có ít nhất 2 giá trị", len(audiences) >= 2, str(audiences))
    check('có tài liệu audience=student (K4 cần cho metadata_filter của benchmark)', audiences.get("student", 0) >= 1)

    print("\nKẾT QUẢ:", "ĐẠT" if not failures else f"CHƯA ĐẠT ({len(failures)} mục)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
