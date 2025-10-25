from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from scripts.company_rag_store import CompanyRAGStore


def list_playbooks(store: CompanyRAGStore) -> None:
    playbook_dir = store.playbook_directory
    entries = []
    for path in sorted(playbook_dir.glob("*.md")):
        if path.name == "archive":
            continue
        stat = path.stat()
        entries.append(
            {
                "file": path.name,
                "company": path.stem,
                "size_kb": round(stat.st_size / 1024, 2),
                "updated": stat.st_mtime,
            }
        )
    print(json.dumps(entries, ensure_ascii=False, indent=2))


def show_playbook(store: CompanyRAGStore, company: str, version: str | None = None) -> None:
    if version:
        path = store.playbook_archive_directory / store._slugify(company) / f"{version}.md"
    else:
        path = store.playbook_directory / f"{store._slugify(company)}.md"
    if not path.exists():
        raise FileNotFoundError(f"未找到 Playbook：{path}")
    print(path.read_text(encoding="utf-8"))


def prune_archives(store: CompanyRAGStore, company: str, keep: int) -> None:
    archive_dir = store.playbook_archive_directory / store._slugify(company)
    if not archive_dir.exists():
        print("没有归档内容，无需清理。")
        return
    files = sorted(archive_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    for obsolete in files[keep:]:
        obsolete.unlink()
    print(f"已清理 {max(0, len(files) - keep)} 个归档，保留最新 {min(len(files), keep)} 个。")


def prune_all(store: CompanyRAGStore, keep: int) -> None:
    for archive_dir in store.playbook_archive_directory.glob("*"):
        if archive_dir.is_dir():
            prune_archives(store, archive_dir.name, keep)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="管理 Playbook 归档与浏览的工具。")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出当前所有 Playbook 概览。")

    show_parser = sub.add_parser("show", help="查看指定公司 Playbook 或归档版本。")
    show_parser.add_argument("--company", required=True, help="公司名称")
    show_parser.add_argument("--version", help="归档版本时间戳")

    prune_parser = sub.add_parser("prune", help="清理某个公司的归档文件。")
    prune_parser.add_argument("--company", required=True)
    prune_parser.add_argument("--keep", type=int, default=5, help="保留最新 N 个归档")

    prune_all_parser = sub.add_parser("prune-all", help="批量清理所有公司的归档。")
    prune_all_parser.add_argument("--keep", type=int, default=5)

    args = parser.parse_args(argv)
    store = CompanyRAGStore()

    if args.cmd == "list":
        list_playbooks(store)
    elif args.cmd == "show":
        show_playbook(store, args.company, version=args.version)
    elif args.cmd == "prune":
        prune_archives(store, args.company, keep=args.keep)
    elif args.cmd == "prune-all":
        prune_all(store, keep=args.keep)


if __name__ == "__main__":
    main()
