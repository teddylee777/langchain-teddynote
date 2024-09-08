# langchain-teddynote

랭체인 한국어 튜토리얼에 사용되는 다양한 유틸 파이썬 패키지.

LangChain 을 사용하면서 불편한 기능이나, 추가적인 기능을 제공합니다.

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
(기입한 프로젝트명)
```

### OpenAI Assistant V2 

```python
from langchain_teddynote.models import OpenAIAssistant


# RAG 시스템 프롬프트 입력
_DEFAULT_RAG_INSTRUCTIONS = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean."""


# 설정(configs)
configs = {
    "OPENAI_API_KEY": openai_api_key,  # OpenAI API 키
    "instructions": _DEFAULT_RAG_INSTRUCTIONS,  # RAG 시스템 프롬프트
    "PROJECT_NAME": "PDF-RAG-TEST",  # 프로젝트 이름(자유롭게 설정)
    "model_name": "gpt-4o",  # 사용할 OpenAI 모델 이름(gpt-4o, gpt-4o-mini, ...)
    "chunk_size": 1000,  # 청크 크기
    "chunk_overlap": 100,  # 청크 중복 크기
}


# 인스턴스 생성
assistant = OpenAIAssistant(configs)

# 업로드할 파일 경로
data = "파일이름.pdf"

# 파일 업로드 후 file_id 는 잘 보관해 두세요. (대시보드에서 나중에 확인 가능)
file_id = assistant.upload_file(data)

# 업로드한 파일의 ID 리스트 생성
file_ids = [file_id]

# 새로운 어시스턴트 생성 및 ID 받기
assistant_id, vector_id = assistant.create_new_assistant(file_ids)

# 어시스턴트 설정
assistant.setup_assistant(assistant_id)

# 벡터 스토어 설정
assistant.setup_vectorstore(vector_id)
```

스트리밍 출력

```python
for token in assistant.stream("삼성전자가 개발한 생성형 AI의 이름은?"):
    print(token, end="", flush=True)
```
혹은

```python
from langchain_teddynote.messages import stream_response

stream_response(assistant.stream("이전 답변을 영어로"))
```

일반 출력

```python
# 질문
print(assistant.invoke("삼성전자가 개발한 생성형 AI의 이름은?"))
```

대화 목록을 조회

```python
# 대화 목록 조회
assistant.list_chat_history()
```

대화 초기화

```python
# 대화 초기화
assistant.clear_chat_history()
```

