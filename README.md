# langchain-teddynote

랭체인 한국어 튜토리얼에 사용되는 다양한 유틸 파이썬 패키지.

LangChain 을 사용하면서 불편한 기능이나, 추가적인 기능을 제공합니다.

- [다운로드 통계](https://pypistats.org/packages/langchain-teddynote)

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
## 멀티모달 모델(이미지 입력)

```python
from langchain_teddynote.models import MultiModal
from langchain_teddynote.messages import stream_response

# 객체 생성
llm = ChatOpenAI(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o",  # 모델명
)

# 멀티모달 객체 생성
system_prompt = """당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다. 
당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다."""

user_prompt = """당신에게 주어진 표는 회사의 재무제표 입니다. 흥미로운 사실을 정리하여 답변하세요."""

# 멀티모달 객체 생성
multimodal_llm = MultiModal(
    llm, system_prompt=system_prompt, user_prompt=user_prompt
)

# 샘플 이미지 주소(웹사이트로 부터 바로 인식)
IMAGE_URL = "https://storage.googleapis.com/static.fastcampus.co.kr/prod/uploads/202212/080345-661/kwon-01.png"

# 로컬 PC 에 저장되어 있는 이미지의 경로 입력
# IMAGE_URL = "./images/sample-image.png"

# 이미지 파일로 부터 질의
answer = multimodal_llm.stream(IMAGE_URL)
# 스트리밍 방식으로 각 토큰을 출력합니다. (실시간 출력)
stream_response(answer)
```

## DeepL 번역기

```python
from langchain_teddynote.translate import Translator

# api키 설정
deepl_api_key = os.getenv("DEEPL_API_KEY")

# 번역 객체 생성(source_lang, target_lang)
translator = Translator(deepl_api_key, "EN", "KO")

# 번역 실행
translated_text = translator("hello, nice to meet you")
print(translated_text)
```

## Kiwi 형태소 분석기

```python
from langchain_teddynote.community.kiwi_tokenizer import KiwiTokenizer

# 토크나이저 선언
kiwi_tokenizer = KiwiTokenizer()

sent1 = "안녕하세요. 반갑습니다. 내 이름은 테디입니다."
sent2 = "안녕하세용 반갑습니다~^^ 내 이름은 테디입니다!!"

# 토큰화
print(kiwi_tokenizer.tokenize(sent1))
print(kiwi_tokenizer.tokenize(sent2))
```

## Synapsoft DocuAnalyzer

```python
from langchain_teddynote.document_parser import SynapsoftDocuAnalyzer

api = SynapsoftDocuAnalyzer(api_key="API_KEY 를 입력해 주세요")

# markdown 형식으로 변환(반환 형식: List[str])
markdown = api.convert_to_markdown("sample.pdf")

# xml 형식으로 변환(반환 형식: List[str])
xml = api.convert_to_xml("sample.pdf")

# json 형식으로 변환(반환 형식: List[str])
json = api.convert_to_json("sample.pdf")
```


## OpenAI Assistant V2 

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

## 튜토리얼

- [Pinecone 커스텀 구현(한글 형태소 분석기)](https://wikidocs.net/252407)
- [한글 형태소 분석기(Kiwi)가 적용된 BM25 Retriever](https://wikidocs.net/251980)

