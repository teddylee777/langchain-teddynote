from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional, Sequence

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor


def default_get_input(query: str, doc: Document) -> Dict[str, Any]:
    """압축 체인의 입력을 반환합니다."""
    return {"question": query, "context": doc.page_content}


class NoOutputParser(BaseOutputParser[str]):
    """널 문자열을 반환할 수 있는 출력을 파싱합니다."""

    no_output_str: str = "NO_OUTPUT"

    def parse(self, text: str) -> str:
        """
        텍스트를 파싱하여 출력을 반환합니다.

        Args:
            text (str): 파싱할 텍스트

        Returns:
            str: 파싱된 텍스트 또는 빈 문자열
        """
        cleaned_text = text.strip()
        if cleaned_text == self.no_output_str:
            return ""
        return cleaned_text


class LLMChainExtractor(BaseDocumentCompressor):
    """LLM 체인을 사용하여 문서의 관련 부분을 추출하는 문서 압축기입니다."""

    llm_chain: Runnable
    """문서 압축에 사용할 LLM 래퍼입니다."""

    get_input: Callable[[str, Document], dict] = default_get_input
    """쿼리와 Document로부터 체인 입력을 구성하는 호출 가능한 객체입니다."""

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        원본 문서의 페이지 내용을 압축합니다.

        Args:
            documents (Sequence[Document]): 압축할 문서 시퀀스
            query (str): 압축 시 사용할 쿼리
            callbacks (Optional[Callbacks]): 콜백 객체

        Returns:
            Sequence[Document]: 압축된 문서 시퀀스
        """
        compressed_docs = []
        for doc in documents:
            _input = self.get_input(query, doc)
            output = self.llm_chain.invoke(_input, config={"callbacks": callbacks})
            if len(output) == 0:
                continue
            compressed_docs.append(
                Document(page_content=str(output), metadata=doc.metadata)
            )
        return compressed_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        원본 문서의 페이지 내용을 비동기적으로 압축합니다.

        Args:
            documents (Sequence[Document]): 압축할 문서 시퀀스
            query (str): 압축 시 사용할 쿼리
            callbacks (Optional[Callbacks]): 콜백 객체

        Returns:
            Sequence[Document]: 압축된 문서 시퀀스
        """
        outputs = await asyncio.gather(
            *[
                self.llm_chain.ainvoke(
                    self.get_input(query, doc), config={"callbacks": callbacks}
                )
                for doc in documents
            ]
        )
        compressed_docs = []
        for i, doc in enumerate(documents):
            output = outputs[i]
            if len(output) == 0:
                continue
            compressed_docs.append(
                Document(page_content=str(output), metadata=doc.metadata)
            )
        return compressed_docs

    @classmethod
    def from_llm(
        cls,
        llm,
        prompt: Optional[PromptTemplate] = None,
    ) -> LLMChainExtractor:
        """
        LLM으로부터 LLMChainExtractor를 초기화합니다.

        Args:
            llm: 사용할 언어 모델
            prompt (Optional[PromptTemplate]): 사용할 프롬프트 템플릿

        Returns:
            LLMChainExtractor: 초기화된 LLMChainExtractor 객체
        """
        if prompt is None:
            prompt_template = """Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return {no_output_str}. 

Remember, *DO NOT* edit the extracted parts of the context.

#Question: {question}

#Context:
>>>
{context}
>>>

Extracted relevant parts:"""
            _prompt = PromptTemplate.from_template(prompt_template)
        else:
            _prompt = PromptTemplate.from_template(prompt)

        _prompt = _prompt.partial(no_output_str="NO_OUTPUT")
        _get_input = default_get_input
        llm_chain = _prompt | llm | NoOutputParser()
        return cls(llm_chain=llm_chain, get_input=_get_input)
