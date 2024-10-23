"""
Ensemble retriever that ensemble the results of
multiple retrievers by using weighted  Reciprocal Rank Fusion
"""

import asyncio
from collections import defaultdict
from collections.abc import Hashable
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    cast,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import ensure_config, patch_config
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    get_unique_config_specs,
)
from pydantic import model_validator
from enum import Enum

T = TypeVar("T")
H = TypeVar("H", bound=Hashable)


class EnsembleMethod(str, Enum):
    RRF = "rrf"
    CC = "cc"


def unique_by_key(iterable: Iterable[T], key: Callable[[T], H]) -> Iterator[T]:
    """Yield unique elements of an iterable based on a key function.

    Args:
        iterable: The iterable to filter.
        key: A function that returns a hashable key for each element.

    Yields:
        Unique elements of the iterable based on the key function.
    """
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e


class EnsembleRetriever(BaseRetriever):
    """Retriever that ensembles the multiple retrievers.

    It uses either Reciprocal Rank Fusion (RRF) or Convex Combination (CC) method.

    Args:
        retrievers: A list of retrievers to ensemble.
        weights: A list of weights corresponding to the retrievers.
                 Must sum to 1 for CC method.
        method: The ensemble method to use. Either "rrf" or "cc".
        c: A constant for RRF method. Default is 60.
        id_key: The key in the document's metadata used to determine unique documents.
            If not specified, page_content is used.
    """

    retrievers: List[RetrieverLike]
    weights: List[float]
    method: EnsembleMethod = EnsembleMethod.RRF
    c: int = 60
    id_key: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def validate_weights(cls, values: Dict[str, Any]) -> Any:
        weights = values.get("weights")
        method = values.get("method", EnsembleMethod.RRF)

        if not weights:
            n_retrievers = len(values["retrievers"])
            values["weights"] = [1 / n_retrievers] * n_retrievers
        elif method == EnsembleMethod.CC and abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0 for CC method")

        return values

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        """List configurable fields for this runnable."""
        return get_unique_config_specs(
            spec for retriever in self.retrievers for spec in retriever.config_specs
        )

    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        from langchain_core.callbacks import CallbackManager

        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            **kwargs,
        )
        try:
            result = self.rank_fusion(input, run_manager=run_manager, config=config)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    async def ainvoke(
        self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> List[Document]:
        from langchain_core.callbacks import AsyncCallbackManager

        config = ensure_config(config)
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            **kwargs,
        )
        try:
            result = await self.arank_fusion(
                input, run_manager=run_manager, config=config
            )
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e
        else:
            await run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get fused result of the retrievers.
        fused_documents = self.rank_fusion(query, run_manager)

        return fused_documents

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Asynchronously get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get fused result of the retrievers.
        fused_documents = await self.arank_fusion(query, run_manager)

        return fused_documents

    def ensemble_results(self, doc_lists: List[List[Document]]) -> List[Document]:
        """
        Ensemble the results using either RRF or CC method.

        Args:
            doc_lists: A list of rank lists, where each rank list contains unique items.

        Returns:
            list: The final aggregated list of items sorted by their scores in descending order.
        """
        if len(doc_lists) != len(self.weights):
            raise ValueError(
                "Number of rank lists must be equal to the number of weights."
            )

        if self.method == EnsembleMethod.RRF:
            return self.reciprocal_rank_fusion(doc_lists)
        elif self.method == EnsembleMethod.CC:
            return self.convex_combination(doc_lists)
        else:
            raise ValueError("Invalid ensemble method")

    def reciprocal_rank_fusion(self, doc_lists: List[List[Document]]) -> List[Document]:
        """
        Perform Reciprocal Rank Fusion on multiple rank lists.
        """
        rrf_score: Dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights):
            for rank, doc in enumerate(doc_list, start=1):
                doc_id = (
                    doc.page_content
                    if self.id_key is None
                    else doc.metadata[self.id_key]
                )
                rrf_score[doc_id] += weight / (rank + self.c)

        all_docs = chain.from_iterable(doc_lists)
        sorted_docs = sorted(
            unique_by_key(
                all_docs,
                lambda doc: (
                    doc.page_content
                    if self.id_key is None
                    else doc.metadata[self.id_key]
                ),
            ),
            key=lambda doc: rrf_score[
                doc.page_content if self.id_key is None else doc.metadata[self.id_key]
            ],
            reverse=True,
        )
        return sorted_docs

    def convex_combination(self, doc_lists: List[List[Document]]) -> List[Document]:
        """
        Perform Convex Combination on multiple rank lists.
        """
        cc_scores: Dict[str, float] = defaultdict(float)

        for doc_list, weight in zip(doc_lists, self.weights):
            max_score = max(doc.metadata.get("score", 0) for doc in doc_list) or 1
            for doc in doc_list:
                doc_id = (
                    doc.page_content
                    if self.id_key is None
                    else doc.metadata[self.id_key]
                )
                normalized_score = doc.metadata.get("score", 0) / max_score
                cc_scores[doc_id] += weight * normalized_score

        all_docs = list(
            unique_by_key(
                chain.from_iterable(doc_lists),
                lambda doc: (
                    doc.page_content
                    if self.id_key is None
                    else doc.metadata[self.id_key]
                ),
            )
        )

        sorted_docs = sorted(
            all_docs,
            key=lambda doc: cc_scores[
                doc.page_content if self.id_key is None else doc.metadata[self.id_key]
            ],
            reverse=True,
        )

        return sorted_docs

    def rank_fusion(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        # Get the results of all retrievers.
        retriever_docs = [
            retriever.invoke(
                query,
                patch_config(
                    config, callbacks=run_manager.get_child(tag=f"retriever_{i+1}")
                ),
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # Enforce that retrieved docs are Documents for each list in retriever_docs
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=cast(str, doc)) if isinstance(doc, str) else doc
                for doc in retriever_docs[i]
            ]

        # apply ensemble method
        fused_documents = self.ensemble_results(retriever_docs)

        return fused_documents

    async def arank_fusion(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        *,
        config: Optional[RunnableConfig] = None,
    ) -> List[Document]:
        # Get the results of all retrievers.
        retriever_docs = await asyncio.gather(
            *[
                retriever.ainvoke(
                    query,
                    patch_config(
                        config, callbacks=run_manager.get_child(tag=f"retriever_{i+1}")
                    ),
                )
                for i, retriever in enumerate(self.retrievers)
            ]
        )

        # Enforce that retrieved docs are Documents for each list in retriever_docs
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=doc) if not isinstance(doc, Document) else doc  # type: ignore[arg-type]
                for doc in retriever_docs[i]
            ]

        # apply ensemble method
        fused_documents = self.ensemble_results(retriever_docs)

        return fused_documents
