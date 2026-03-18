#!/usr/bin/env python3
"""GitHub 자동 푸시 에이전트

작업 내역을 자동으로 커밋하고 GitHub에 푸시합니다.

사용법:
  python3 scripts/git_push_agent.py
  python3 scripts/git_push_agent.py --message "커밋 메시지 직접 입력"
  python3 scripts/git_push_agent.py --dry-run   # 실제 푸시 없이 확인만
"""
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=REPO_ROOT, check=check,
    )


def get_status() -> str:
    return run("git status --short").stdout.strip()


def get_diff_summary() -> str:
    result = run("git diff --stat HEAD 2>/dev/null || git diff --stat", check=False)
    return result.stdout.strip()


def get_changed_files() -> list[str]:
    result = run("git status --short")
    files = []
    for line in result.stdout.strip().splitlines():
        if line.strip():
            files.append(line.strip())
    return files


def auto_commit_message(files: list[str]) -> str:
    """변경 파일 목록 기반으로 커밋 메시지 자동 생성."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 변경 카테고리 분류
    categories = []
    file_names = [f.split()[-1] for f in files]

    if any("config/" in f for f in file_names):
        categories.append("config 업데이트")
    if any("run_live_bot" in f for f in file_names):
        categories.append("봇 로직 수정")
    if any("src/execution/" in f for f in file_names):
        categories.append("실행 모듈 수정")
    if any("src/" in f for f in file_names) and "src/execution/" not in str(file_names):
        categories.append("ML 모듈 수정")
    if any("scripts/" in f for f in file_names):
        categories.append("스크립트 추가/수정")
    if any("data/" in f for f in file_names):
        categories.append("데이터 업데이트")
    if not categories:
        categories.append("기타 수정")

    summary = ", ".join(categories)
    return f"[{now}] {summary}"


def push(message: str | None = None, dry_run: bool = False) -> bool:
    """변경사항 스테이징 → 커밋 → 푸시."""

    # 1. 상태 확인
    status = get_status()
    if not status:
        print("변경사항 없음 — 푸시할 내용이 없습니다.")
        return False

    print("=" * 50)
    print("변경 파일 목록:")
    changed = get_changed_files()
    for f in changed:
        print(f"  {f}")
    print()

    # 2. 스테이징 (data/reports, 로그, DB 제외)
    run("git add config/ run_live_bot_v2.py src/ scripts/", check=False)
    run("git add docs/ experiments/ README.md CLAUDE.md", check=False)

    # 스테이지된 파일 확인
    staged = run("git diff --cached --name-only").stdout.strip()
    if not staged:
        print("스테이지된 파일 없음 (data/reports 등 제외 후).")
        # 전체 추가 시도
        run("git add -A")
        staged = run("git diff --cached --name-only").stdout.strip()
        if not staged:
            print("커밋할 변경사항 없음.")
            return False

    print("커밋 대상 파일:")
    for f in staged.splitlines():
        print(f"  ✓ {f}")
    print()

    # 3. 커밋 메시지
    if not message:
        staged_files = [f"M {f}" for f in staged.splitlines()]
        message = auto_commit_message(staged_files)

    print(f"커밋 메시지: {message}")

    if dry_run:
        print("\n[DRY-RUN] 실제 커밋/푸시는 하지 않습니다.")
        return True

    # 4. 커밋
    result = run(f'git commit -m "{message}"', check=False)
    if result.returncode != 0:
        print(f"커밋 실패: {result.stderr}")
        return False
    print(f"커밋 완료: {result.stdout.strip()}")

    # 5. 원격 최신화 후 푸시
    print("원격 변경사항 동기화 중...")
    pull = run("git pull --rebase origin main", check=False)
    if pull.returncode != 0:
        print(f"Pull 경고: {pull.stderr.strip()}")

    push_result = run("git push origin main", check=False)
    if push_result.returncode != 0:
        print(f"푸시 실패: {push_result.stderr}")
        return False

    print("✅ GitHub 푸시 완료!")
    # 최신 커밋 확인
    log = run("git log --oneline -3").stdout.strip()
    print(f"\n최근 커밋:\n{log}")
    return True


def main():
    parser = argparse.ArgumentParser(description="GitHub 자동 푸시 에이전트")
    parser.add_argument("--message", "-m", type=str, default=None, help="커밋 메시지")
    parser.add_argument("--dry-run", action="store_true", help="실제 푸시 없이 확인만")
    args = parser.parse_args()

    success = push(message=args.message, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
