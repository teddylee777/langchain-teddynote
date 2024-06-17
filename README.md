# LangChain TeddyNote

랭체인 한국어 튜토리얼에 사용되는 다양한 유틸 파이썬 패키지

## 설치

```bash
pip install langchain-teddynote
```

## 사용법

### 스트리밍 출력

스트리밍 출력을 위한 `stream_response` 함수를 제공합니다.

```python
from langchain_teddynote.messages import stream_response
from langchain_openai import ChatOpenAI

# 객체 생성
llm = ChatOpenAI(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o",  # 모델명
)
answer = llm.stream("대한민국의 아름다운 관장지 10곳과 주소를 알려주세요!")

# 스트리밍 출력만 하는 경우
stream_response(answer)

# 출력된 답변을 반환 값으로 받는 경우
# final_answer = stream_response(answer, return_output=True)
```
### LangSmith 추적

```python
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# 환경변수 설정은 되어 있다고 가정합니다.
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("프로젝트명 기입")
```
출력
```
LangSmith 추적을 시작합니다.
[프로젝트명]
CH01-Basic
```

### load_prompt 인코딩 설정 적용

```python
from langchain_teddynote.prompts import load_prompt

# UTF-8로 인코딩을 설정합니다.(기본값)
load_prompt("prompts/capital.yaml", encoding="utf-8")

# Windows에서는 cp949로 인코딩을 변경합니다.
load_prompt("prompts/capital.yaml", encoding="cp949")
```