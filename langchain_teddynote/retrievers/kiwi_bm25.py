from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional
from operator import itemgetter
import numpy as np

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field
from langchain_core.retrievers import BaseRetriever

try:
    from kiwipiepy import Kiwi
except ImportError:
    raise ImportError(
        "Could not import kiwipiepy, please install with `pip install " "kiwipiepy`."
    )

kiwi_tokenizer = Kiwi()


def kiwi_preprocessing_func(text: str) -> List[str]:
    return [token.form for token in kiwi_tokenizer.tokenize(text)]


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


class KiwiBM25Retriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = kiwi_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = kiwi_preprocessing_func,
        **kwargs: Any,
    ) -> KiwiBM25Retriever:
        """
        Create a KiwiBM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A KiwiBM25Retriever instance.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )

        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = kiwi_preprocessing_func,
        **kwargs: Any,
    ) -> KiwiBM25Retriever:
        """
        Create a KiwiBM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A KiwiBM25Retriever instance.
        """
        texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        processed_query = self.preprocess_func(query)
        return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        return return_docs

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def argsort(seq, reverse):
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

    def search_with_score(self, query: str, top_k=None):
        normalized_score = KiwiBM25Retriever.softmax(
            self.vectorizer.get_scores(self.preprocess_func(query))
        )

        if top_k is None:
            top_k = self.k

        score_indexes = KiwiBM25Retriever.argsort(normalized_score, True)

        docs_with_scores = []
        for i, doc in enumerate(self.docs):
            document = Document(
                page_content=doc.page_content, metadata={"score": normalized_score[i]}
            )
            docs_with_scores.append(document)

        score_indexes = score_indexes[:top_k]

        # Creating an itemgetter object
        getter = itemgetter(*score_indexes)

        # Using itemgetter to get items
        selected_elements = getter(docs_with_scores)
        return selected_elements
