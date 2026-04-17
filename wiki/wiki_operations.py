#!/usr/bin/env python3
"""Utility operations for managing this repository's wiki markdown files."""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REQUIRED_PAGES = [
    "Home.md",
    "AlexNet-Architecture-Theory.md",
    "Advanced-Training-Configuration.md",
    "Data-Preparation-Guide.md",
    "Hyperparameter-Tuning.md",
    "Model-Performance-and-Metrics.md",
    "Advanced-Troubleshooting.md",
    "TensorFlow-2x-Migration-Notes.md",
]


def wiki_pages(wiki_dir: Path):
    return sorted(
        [p for p in wiki_dir.glob("*.md") if p.name not in {"README.md"}],
        key=lambda p: p.name,
    )


def cmd_copy(args):
    wiki_dir = args.wiki_dir.resolve()
    dest_dir = args.dest.resolve()

    if not wiki_dir.is_dir():
        print(f"ERROR: wiki directory not found: {wiki_dir}", file=sys.stderr)
        return 1
    if not dest_dir.is_dir():
        print(f"ERROR: destination directory not found: {dest_dir}", file=sys.stderr)
        return 1

    copied = 0
    for page in wiki_pages(wiki_dir):
        target = dest_dir / page.name
        if args.dry_run:
            print(f"[dry-run] copy {page} -> {target}")
        else:
            shutil.copy2(page, target)
            print(f"copied {page.name} -> {target}")
        copied += 1

    print(f"Done. Pages processed: {copied}")
    return 0


def cmd_gh_create(args):
    wiki_dir = args.wiki_dir.resolve()
    if not wiki_dir.is_dir():
        print(f"ERROR: wiki directory not found: {wiki_dir}", file=sys.stderr)
        return 1

    if not shutil.which("gh"):
        print("ERROR: GitHub CLI (gh) was not found in PATH.", file=sys.stderr)
        return 1

    failures = 0
    for page in wiki_pages(wiki_dir):
        title = page.stem
        command = ["gh", "wiki", "create", title, "-F", str(page)]
        if args.repo:
            command.extend(["-R", args.repo])

        if args.dry_run:
            print("[dry-run] " + " ".join(command))
            continue

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            failures += 1
            print(f"FAILED: {page.name}", file=sys.stderr)
            if result.stderr:
                print(result.stderr.strip(), file=sys.stderr)
        else:
            print(f"created page: {title}")

    if failures:
        print(f"Completed with {failures} failures.", file=sys.stderr)
        return 1

    print("Done. All wiki pages created successfully.")
    return 0


def check_internal_links(file_path: Path, wiki_dir: Path):
    issues = []
    content = file_path.read_text(encoding="utf-8")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", content)
    for link in links:
        link = link.strip()
        if not link or link.startswith(("http://", "https://", "#", "mailto:")):
            continue

        target = link.split("#", 1)[0]
        if not target:
            continue

        target_path = (file_path.parent / target).resolve()
        if target_path.exists():
            continue

        md_target = (file_path.parent / (target + ".md")).resolve()
        if md_target.exists():
            continue

        wiki_slug_target = (wiki_dir / (Path(target).name + ".md")).resolve()
        if wiki_slug_target.exists():
            continue

        issues.append(f"{file_path.name}: unresolved link '{link}'")
    return issues


def cmd_verify(args):
    wiki_dir = args.wiki_dir.resolve()
    if not wiki_dir.is_dir():
        print(f"ERROR: wiki directory not found: {wiki_dir}", file=sys.stderr)
        return 1

    issues = []

    required_pages = DEFAULT_REQUIRED_PAGES if not args.required_pages else args.required_pages
    for page_name in required_pages:
        if not (wiki_dir / page_name).exists():
            issues.append(f"missing required page: {page_name}")

    for page in wiki_pages(wiki_dir):
        issues.extend(check_internal_links(page, wiki_dir))

    if issues:
        print("Verification failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Verification passed.")
    return 0


def parse_args():
    default_wiki_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Wiki operations helper")
    parser.add_argument(
        "--wiki-dir",
        type=Path,
        default=default_wiki_dir,
        help=f"Path to wiki markdown files (default: {default_wiki_dir})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_copy = subparsers.add_parser("copy", help="Copy wiki pages to a local .wiki git repository")
    parser_copy.add_argument("--dest", type=Path, required=True, help="Destination wiki repository directory")
    parser_copy.add_argument("--dry-run", action="store_true", help="Show actions without copying files")
    parser_copy.set_defaults(func=cmd_copy)

    parser_gh = subparsers.add_parser("gh-create", help="Create wiki pages using GitHub CLI")
    parser_gh.add_argument("-R", "--repo", help="Repository in OWNER/REPO form (optional)")
    parser_gh.add_argument("--dry-run", action="store_true", help="Show gh commands without executing")
    parser_gh.set_defaults(func=cmd_gh_create)

    parser_verify = subparsers.add_parser("verify", help="Verify required pages and internal links")
    parser_verify.add_argument(
        "--required-pages",
        nargs="*",
        help="Optional custom required pages list (filenames with .md)",
    )
    parser_verify.set_defaults(func=cmd_verify)

    return parser.parse_args()


def main():
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
