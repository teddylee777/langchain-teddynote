#!/usr/bin/env python
"""
PyPI 패키지 배포 자동화 스크립트

사용법:
    python publish.py              # patch 버전 증가 (기본)
    python publish.py --minor      # minor 버전 증가
    python publish.py --major      # major 버전 증가
    python publish.py --test       # TestPyPI로 배포
    python publish.py --dry-run    # 실제 배포 없이 테스트
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple
from dotenv import load_dotenv

load_dotenv(".env", override=True)

# 버전 파일 경로
VERSION_FILE = Path("langchain_teddynote/__init__.py")
PYPROJECT_FILE = Path("pyproject.toml")

# 버전 제한
MAX_PATCH = 99
MAX_MINOR = 99


def run_command(cmd: str, dry_run: bool = False) -> subprocess.CompletedProcess:
    """명령어 실행"""
    print(f"🔧 실행: {cmd}")
    if dry_run:
        print("  (dry-run: 실행하지 않음)")
        return subprocess.CompletedProcess(args=cmd, returncode=0)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ 오류 발생:\n{result.stderr}")
        sys.exit(1)
    return result


def get_current_version() -> Tuple[int, int, int]:
    """현재 버전 읽기"""
    if not VERSION_FILE.exists():
        print(f"❌ 버전 파일을 찾을 수 없습니다: {VERSION_FILE}")
        sys.exit(1)
    
    content = VERSION_FILE.read_text()
    match = re.search(r'__version__\s*=\s*["\'](\d+)\.(\d+)\.(\d+)["\']', content)
    
    if not match:
        print("❌ 버전 정보를 찾을 수 없습니다")
        sys.exit(1)
    
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def calculate_new_version(current: Tuple[int, int, int], bump_type: str) -> Tuple[int, int, int]:
    """새 버전 계산 (최대치 제한 적용)"""
    major, minor, patch = current
    
    if bump_type == "patch":
        patch += 1
        if patch > MAX_PATCH:
            patch = 0
            minor += 1
            if minor > MAX_MINOR:
                minor = 0
                major += 1
    elif bump_type == "minor":
        minor += 1
        patch = 0
        if minor > MAX_MINOR:
            minor = 0
            major += 1
    elif bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    
    return major, minor, patch


def update_version_files(new_version: str, dry_run: bool = False):
    """버전 파일 업데이트"""
    print(f"📝 버전 업데이트: {new_version}")
    
    # __init__.py 업데이트
    if VERSION_FILE.exists():
        content = VERSION_FILE.read_text()
        new_content = re.sub(
            r'__version__\s*=\s*["\'][0-9.]+["\']',
            f'__version__ = "{new_version}"',
            content
        )
        if not dry_run:
            VERSION_FILE.write_text(new_content)
        print(f"  ✅ {VERSION_FILE}")
    
    # pyproject.toml 업데이트
    if PYPROJECT_FILE.exists():
        content = PYPROJECT_FILE.read_text()
        new_content = re.sub(
            r'version\s*=\s*["\'][0-9.]+["\']',
            f'version = "{new_version}"',
            content
        )
        if not dry_run:
            PYPROJECT_FILE.write_text(new_content)
        print(f"  ✅ {PYPROJECT_FILE}")


def clean_build_dirs():
    """빌드 디렉토리 정리"""
    print("🧹 빌드 디렉토리 정리")
    dirs_to_clean = ["dist", "build", "*.egg-info"]
    
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  삭제: {path}")


def build_package(dry_run: bool = False):
    """패키지 빌드"""
    print("\n📦 패키지 빌드")
    
    # UV 사용 가능 여부 확인
    uv_check = subprocess.run("uv --version", shell=True, capture_output=True)
    
    if uv_check.returncode == 0:
        # UV로 빌드
        run_command("uv build", dry_run)
    else:
        # fallback: python build 모듈 사용
        print("  ⚠️  UV를 찾을 수 없습니다. python -m build 사용")
        run_command("python -m build", dry_run)


def upload_package(test: bool = False, dry_run: bool = False):
    """패키지 업로드"""
    if test:
        print("\n🧪 TestPyPI로 업로드")
        repository_url = "--repository-url https://test.pypi.org/legacy/"
    else:
        print("\n🚀 PyPI로 업로드")
        repository_url = ""
    
    # twine 업로드
    cmd = f"python -m twine upload {repository_url} dist/*"
    
    if dry_run:
        print(f"  (dry-run: {cmd})")
    else:
        if not test and not os.environ.get("TWINE_TOKEN"):
            print("  ⚠️  TWINE_TOKEN 환경 변수가 설정되지 않았습니다")
            print("     ~/.pypirc 파일 또는 대화형 인증을 사용합니다")
        
        run_command(cmd, dry_run)


def create_git_tag(version: str, dry_run: bool = False):
    """Git 태그 생성"""
    print(f"\n🏷️  Git 태그 생성: v{version}")
    
    # 변경사항 커밋
    run_command("git add -A", dry_run)
    run_command(f'git commit -m "Release v{version}"', dry_run)
    
    # 태그 생성
    run_command(f"git tag -a v{version} -m 'Release v{version}'", dry_run)
    
    if not dry_run:
        print("  💡 태그를 원격 저장소에 푸시하려면:")
        print(f"     git push origin v{version}")


def main():
    parser = argparse.ArgumentParser(description="PyPI 패키지 배포 자동화")
    parser.add_argument(
        "bump_type",
        nargs="?",
        default="patch",
        choices=["patch", "minor", "major"],
        help="버전 증가 타입 (기본: patch)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="TestPyPI로 배포"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 실행 없이 테스트"
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Git 태그 생성 건너뛰기"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="빌드 디렉토리 정리 건너뛰기"
    )
    
    args = parser.parse_args()
    
    print("🚀 PyPI 패키지 배포 시작")
    print("=" * 50)
    
    # 현재 버전 확인
    current_version = get_current_version()
    print(f"📌 현재 버전: {'.'.join(map(str, current_version))}")
    
    # 새 버전 계산
    new_version_tuple = calculate_new_version(current_version, args.bump_type)
    new_version = ".".join(map(str, new_version_tuple))
    print(f"📌 새 버전: {new_version} ({args.bump_type} bump)")
    
    if args.dry_run:
        print("\n⚠️  DRY-RUN 모드: 실제 변경사항 없음")
    
    # 버전 파일 업데이트
    update_version_files(new_version, args.dry_run)
    
    # 빌드 디렉토리 정리
    if not args.no_clean:
        clean_build_dirs()
    
    # 패키지 빌드
    build_package(args.dry_run)
    
    # 패키지 업로드
    upload_package(args.test, args.dry_run)
    
    # Git 태그 생성
    if not args.no_tag:
        create_git_tag(new_version, args.dry_run)
    
    print("\n" + "=" * 50)
    if args.dry_run:
        print("✅ DRY-RUN 완료")
    else:
        print(f"✅ 배포 완료: v{new_version}")
        if args.test:
            print(f"   확인: https://test.pypi.org/project/langchain-teddynote/{new_version}/")
        else:
            print(f"   확인: https://pypi.org/project/langchain-teddynote/{new_version}/")


if __name__ == "__main__":
    main()