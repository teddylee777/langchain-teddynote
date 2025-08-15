# 배포 가이드

이 문서는 `langchain-teddynote` 패키지의 PyPI 배포 프로세스를 설명합니다.

## 목차
- [환경 설정](#환경-설정)
- [버전 관리 정책](#버전-관리-정책)
- [배포 프로세스](#배포-프로세스)
- [명령어 레퍼런스](#명령어-레퍼런스)
- [트러블슈팅](#트러블슈팅)

## 환경 설정

### 1. 필수 도구 설치

```bash
# UV 설치 (권장)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Homebrew 사용
brew install uv

# Twine 설치 (이미 dependencies에 포함)
pip install twine
```

### 2. PyPI 인증 설정

#### 옵션 A: 환경 변수 사용 (권장)

```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxxxx  # PyPI API 토큰

# TestPyPI용
export TWINE_TEST_USERNAME=__token__
export TWINE_TEST_PASSWORD=pypi-xxxxx  # TestPyPI API 토큰
```

#### 옵션 B: ~/.pypirc 파일 사용

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-xxxxx

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-xxxxx
```

### 3. PyPI API 토큰 발급

1. [PyPI 계정 설정](https://pypi.org/manage/account/) 접속
2. "API tokens" 섹션에서 "Add API token" 클릭
3. 토큰 이름 입력 및 범위 설정
4. 생성된 토큰 안전하게 보관

## 버전 관리 정책

### 버전 체계
- **형식**: `major.minor.patch` (예: 0.4.0)
- **버전 소스**: `langchain_teddynote/__init__.py`의 `__version__` 변수

### 버전 제한 규칙
- **Patch 버전**: 0-99 (100이 되면 자동으로 minor 증가)
- **Minor 버전**: 0-99 (100이 되면 자동으로 major 증가)
- **Major 버전**: 제한 없음

### 버전 증가 가이드라인
- **Patch** (기본): 버그 수정, 작은 개선
- **Minor**: 새로운 기능 추가 (하위 호환성 유지)
- **Major**: 큰 변경사항, API 변경 (하위 호환성 깨짐)

## 배포 프로세스

### 1. 기본 배포 (Patch 버전 증가)

```bash
python publish.py
```

자동으로 수행되는 작업:
1. 버전 0.4.0 → 0.4.1로 증가
2. 빌드 디렉토리 정리
3. 패키지 빌드 (wheel & sdist)
4. PyPI 업로드
5. Git 커밋 및 태그 생성

### 2. Minor/Major 버전 증가

```bash
# Minor 버전 증가 (0.4.0 → 0.5.0)
python publish.py minor

# Major 버전 증가 (0.4.0 → 1.0.0)
python publish.py major
```

### 3. TestPyPI로 테스트 배포

```bash
# TestPyPI로 배포하여 테스트
python publish.py --test

# 설치 테스트
pip install -i https://test.pypi.org/simple/ langchain-teddynote==0.4.1
```

### 4. Dry-run 모드

```bash
# 실제 실행 없이 프로세스 확인
python publish.py --dry-run

# Minor 버전 증가 dry-run
python publish.py minor --dry-run
```

### 5. Git 태그 푸시

배포 스크립트는 자동으로 Git 태그를 생성하지만, 원격 저장소에는 수동으로 푸시해야 합니다:

```bash
# 특정 태그 푸시
git push origin v0.4.1

# 모든 태그 푸시
git push origin --tags
```

## 명령어 레퍼런스

### publish.py 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| (없음) | Patch 버전 증가 (기본) | `python publish.py` |
| `minor` | Minor 버전 증가 | `python publish.py minor` |
| `major` | Major 버전 증가 | `python publish.py major` |
| `--test` | TestPyPI로 배포 | `python publish.py --test` |
| `--dry-run` | 실제 실행 없이 테스트 | `python publish.py --dry-run` |
| `--no-tag` | Git 태그 생성 건너뛰기 | `python publish.py --no-tag` |
| `--no-clean` | 빌드 디렉토리 정리 건너뛰기 | `python publish.py --no-clean` |

### 수동 배포 명령어

필요시 수동으로 배포할 수 있습니다:

```bash
# 1. 빌드 디렉토리 정리
rm -rf dist/ build/ *.egg-info

# 2. 패키지 빌드
uv build

# 3. 배포 전 확인
twine check dist/*

# 4. TestPyPI 업로드
twine upload --repository testpypi dist/*

# 5. PyPI 업로드
twine upload dist/*
```

## 트러블슈팅

### 일반적인 문제 해결

#### 1. "uv: command not found"
```bash
# UV 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 또는 Python build 모듈 사용
pip install build
python -m build
```

#### 2. "twine: command not found"
```bash
pip install twine
# 또는
pip install -e .  # dependencies에 포함되어 있음
```

#### 3. 인증 실패
- API 토큰이 올바른지 확인
- 사용자명이 `__token__`인지 확인
- TestPyPI와 PyPI 토큰을 혼동하지 않았는지 확인

#### 4. 버전 충돌
```bash
# 이미 존재하는 버전인 경우
python publish.py minor  # 다음 버전으로 증가
```

#### 5. Git 태그 충돌
```bash
# 기존 태그 삭제 (주의!)
git tag -d v0.4.1
git push origin :refs/tags/v0.4.1

# 다시 배포
python publish.py
```

### 롤백 방법

배포에 문제가 있는 경우:

1. **PyPI에서는 버전을 삭제할 수 없습니다** (영구 보관)
2. 대신 새로운 수정 버전을 배포:
   ```bash
   python publish.py patch  # hotfix 버전 배포
   ```

3. Git에서 태그 제거 (선택사항):
   ```bash
   git tag -d v0.4.1
   git push origin :refs/tags/v0.4.1
   ```

### 체크리스트

배포 전 확인사항:

- [ ] 모든 테스트 통과
- [ ] CHANGELOG 업데이트
- [ ] 문서 업데이트
- [ ] 의존성 버전 확인
- [ ] TestPyPI에서 테스트 완료

## 추가 리소스

- [PyPI 공식 문서](https://pypi.org/help/)
- [UV 문서](https://github.com/astral-sh/uv)
- [Twine 문서](https://twine.readthedocs.io/)
- [Python Packaging 가이드](https://packaging.python.org/)

## 지원

문제가 있거나 질문이 있으시면:
- Issue: https://github.com/teddylee777/langchain-teddynote/issues
- Email: teddylee777@gmail.com