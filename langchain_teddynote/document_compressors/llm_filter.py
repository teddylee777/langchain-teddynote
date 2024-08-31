from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Optional, Sequence

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor


def default_get_input(query: str, doc: Document) -> Dict[str, Any]:
    """압축 체인의 입력을 반환합니다."""
    return {"question": query, "context": doc.page_content}


def boolean_output_parser(output: str) -> bool:
    """출력을 불리언 값으로 파싱합니다."""
    return output.lower() == "yes"


class LLMChainFilter(BaseDocumentCompressor):
    """쿼리와 관련 없는 문서를 제거하는 필터입니다."""

    llm_chain: Runnable
    """문서 필터링에 사용할 LLM 래퍼입니다."""

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
        """쿼리와의 관련성에 따라 문서를 필터링합니다."""
        filtered_docs = []
        for doc in documents:
            _input = self.get_input(query, doc)
            output = self.llm_chain.invoke(_input, config={"callbacks": callbacks})
            if output:
                filtered_docs.append(doc)
        return filtered_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """쿼리와의 관련성에 따라 문서를 비동기적으로 필터링합니다."""
        outputs = await asyncio.gather(
            *[
                self.llm_chain.ainvoke(
                    self.get_input(query, doc), config={"callbacks": callbacks}
                )
                for doc in documents
            ]
        )
        return [doc for doc, output in zip(documents, outputs) if output]

    @classmethod
    def from_llm(
        cls,
        llm,
        prompt: Optional[BasePromptTemplate] = None,
        **kwargs: Any,
    ) -> LLMChainFilter:
        """언어 모델로부터 LLMChainFilter를 생성합니다."""
        if prompt is None:
            prompt_template = """Given the following question and context, return YES if the context is relevant to the question and NO if it isn't.

> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant (YES / NO):"""
            _prompt = PromptTemplate.from_template(prompt_template)
        else:
            _prompt = PromptTemplate.from_template(prompt)

        # LLM 체인을 구성합니다: 프롬프트 -> LLM -> 문자열 파서 -> 불리언 파서
        llm_chain = _prompt | llm | StrOutputParser() | boolean_output_parser
        return cls(llm_chain=llm_chain, **kwargs)
