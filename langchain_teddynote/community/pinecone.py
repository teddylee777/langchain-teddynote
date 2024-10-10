import os
import time
import pickle
import secrets
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import ConfigDict, model_validator
from langchain_core.retrievers import BaseRetriever
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.embeddings import Embeddings

from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse import BM25Encoder

try:
    from pinecone.exceptions import PineconeException
except ImportError:
    try:
        from pinecone.core.client.exceptions import PineconeException
    except ImportError:
        from pinecone.exceptions.exceptions import PineconeException


from .kiwi_tokenizer import KiwiBM25Tokenizer


def generate_hash() -> str:
    """24자리 무작위 hex 값을 생성하고 6자리씩 나누어 '-'로 연결합니다."""
    random_hex = secrets.token_hex(12)
    return "-".join(random_hex[i : i + 6] for i in range(0, 24, 6))


def create_index(
    api_key: str,
    index_name: str,
    dimension: int,
    metric: str = "dotproduct",
    pod_spec=None,
) -> Any:
    """Pinecone 인덱스를 생성하고 반환합니다."""
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        if pod_spec is None:
            pod_spec = ServerlessSpec(cloud="aws", region="us-east-1")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=pod_spec,
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    print(f"[create_index]\n{index.describe_index_stats()}")
    return index


def create_sparse_encoder(stopwords: List[str], mode: str = "kiwi") -> BM25Encoder:
    """BM25Encoder를 생성하고 반환합니다."""
    bm25 = BM25Encoder(language="english")
    if mode == "kiwi":
        bm25._tokenizer = KiwiBM25Tokenizer(stop_words=stopwords)
    return bm25


def preprocess_documents(
    split_docs: List[Document],
    metadata_keys: List[str] = ["source", "page"],
    min_length: int = 2,
    use_basename: bool = False,
) -> tuple:
    """문서를 전처리하고 내용과 메타데이터를 반환합니다."""
    contents = []
    metadatas = {key: [] for key in metadata_keys}
    for doc in tqdm(split_docs):
        content = doc.page_content.strip()
        if content and len(content) >= min_length:
            contents.append(content)
            for k in metadata_keys:
                value = doc.metadata.get(k)
                if k == "source" and use_basename:
                    value = os.path.basename(value)
                try:
                    metadatas[k].append(int(value))
                except (ValueError, TypeError):
                    metadatas[k].append(value)
    return contents, metadatas


def fit_sparse_encoder(
    sparse_encoder: BM25Encoder, contents: List[str], save_path: str
) -> str:
    """Sparse Encoder 를 학습하고 저장합니다."""
    sparse_encoder.fit(contents)
    with open(save_path, "wb") as f:
        pickle.dump(sparse_encoder, f)
    print(f"[fit_sparse_encoder]\nSaved Sparse Encoder to: {save_path}")
    return save_path


def load_sparse_encoder(file_path: str) -> Any:
    """저장된 스파스 인코더를 로드합니다."""
    try:
        with open(file_path, "rb") as f:
            loaded_file = pickle.load(f)
        print(f"[load_sparse_encoder]\nLoaded Sparse Encoder from: {file_path}")
        return loaded_file
    except Exception as e:
        print(f"[load_sparse_encoder]\n{e}")
        return None


def upsert_documents(
    index: Any,
    namespace: str,
    contents: List[str],
    metadatas: Dict[str, List[Any]],
    sparse_encoder: BM25Encoder,
    embedder: Embeddings,
    batch_size: int = 32,
):
    """문서를 Pinecone 인덱스에 업서트합니다."""
    keys = list(metadatas.keys())

    for i in tqdm(range(0, len(contents), batch_size)):
        i_end = min(i + batch_size, len(contents))
        context_batch = contents[i:i_end]
        metadata_batches = {key: metadatas[key][i:i_end] for key in keys}

        batch_result = [
            {"context": context, **{key: metadata_batches[key][j] for key in keys}}
            for j, context in enumerate(context_batch)
        ]

        ids = [generate_hash() for _ in range(i, i_end)]
        dense_embeds = embedder.embed_documents(context_batch)
        sparse_embeds = sparse_encoder.encode_documents(context_batch)

        vectors = [
            {
                "id": _id,
                "sparse_values": sparse,
                "values": dense,
                "metadata": metadata,
            }
            for _id, sparse, dense, metadata in zip(
                ids, sparse_embeds, dense_embeds, batch_result
            )
        ]

        index.upsert(vectors=vectors, namespace=namespace)

    print(f"[upsert_documents]\n{index.describe_index_stats()}")


def upsert_documents_parallel(
    index,
    namespace,
    contents,
    metadatas,
    sparse_encoder,
    embedder,
    batch_size=100,  # 배치 크기를 줄임
    max_workers=30,
):
    keys = list(metadatas.keys())

    def chunks(iterable, size):
        it = iter(iterable)
        chunk = list(itertools.islice(it, size))
        while chunk:
            yield chunk
            chunk = list(itertools.islice(it, size))

    def process_batch(batch):
        context_batch = [contents[i] for i in batch]
        metadata_batches = {key: [metadatas[key][i] for i in batch] for key in keys}

        batch_result = [
            {
                "context": context[:1000],
                **{key: metadata_batches[key][j] for key in keys},
            }  # 컨텍스트 길이 제한
            for j, context in enumerate(context_batch)
        ]

        ids = [generate_hash() for _ in range(len(batch))]
        dense_embeds = embedder.embed_documents(context_batch)
        sparse_embeds = sparse_encoder.encode_documents(context_batch)

        vectors = [
            {
                "id": _id,
                "sparse_values": sparse,
                "values": dense,
                "metadata": metadata,
            }
            for _id, sparse, dense, metadata in zip(
                ids, sparse_embeds, dense_embeds, batch_result
            )
        ]

        try:
            return index.upsert(vectors=vectors, namespace=namespace, async_req=False)
        except Exception as e:
            print(f"Upsert 중 오류 발생: {e}")
            return None

    batches = list(chunks(range(len(contents)), batch_size))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]

        results = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="문서 Upsert 중"
        ):
            result = future.result()
            if result:
                results.append(result)

    total_upserted = sum(result.upserted_count for result in results if result)
    print(f"총 {total_upserted}개의 Vector 가 Upsert 되었습니다.")
    print(f"{index.describe_index_stats()}")


def delete_namespace(pinecone_index: Any, namespace: str):
    try:
        # 네임스페이스 존재 여부 확인
        namespace_stats = pinecone_index.describe_index_stats()
        if namespace in namespace_stats["namespaces"]:
            pinecone_index.delete(delete_all=True, namespace=namespace)
            print(f"네임스페이스 '{namespace}'의 모든 데이터가 삭제되었습니다.")
        else:
            print(f"네임스페이스 '{namespace}'가 존재하지 않습니다.")

    except PineconeException as e:
        print(f"Pinecone 작업 중 오류 발생:\n{e}")


def delete_by_filter(pinecone_index: Any, namespace: str, filter: Dict):
    # 필터를 사용한 삭제 시도
    try:
        pinecone_index.delete(filter=filter, namespace=namespace)
    except PineconeException as e:
        print(f"필터를 사용한 삭제 중 오류 발생:\n{e}")


def init_pinecone_index(
    index_name: str,
    namespace: str,
    api_key: str,
    sparse_encoder_path: str = None,
    stopwords: List[str] = None,
    tokenizer: str = "kiwi",
    embeddings: Embeddings = None,
    top_k: int = 10,
    alpha: float = 0.5,
) -> Dict:
    """Pinecone 인덱스를 초기화하고 필요한 구성 요소를 반환합니다."""
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    print(f"[init_pinecone_index]\n{index.describe_index_stats()}")

    try:
        with open(sparse_encoder_path, "rb") as f:
            bm25 = pickle.load(f)
        if tokenizer == "kiwi":
            bm25._tokenizer = KiwiBM25Tokenizer(stop_words=stopwords)
    except Exception as e:
        print(e)
        return {}

    namespace_keys = index.describe_index_stats()["namespaces"].keys()
    if namespace not in namespace_keys:
        raise ValueError(
            f"`{namespace}` 를 `{list(namespace_keys)}` 에서 찾지 못했습니다."
        )

    return {
        "index": index,
        "namespace": namespace,
        "sparse_encoder": bm25,
        "embeddings": embeddings,
        "top_k": top_k,
        "alpha": alpha,
    }


class PineconeKiwiHybridRetriever(BaseRetriever):
    """
    Pinecone과 Kiwi를 결합한 하이브리드 검색기 클래스입니다.

    이 클래스는 밀집 벡터와 희소 벡터를 모두 사용하여 문서를 검색합니다.
    Pinecone 인덱스와 Kiwi 토크나이저를 활용하여 효과적인 하이브리드 검색을 수행합니다.

    매개변수:
        embeddings (Embeddings): 문서와 쿼리를 밀집 벡터로 변환하는 임베딩 모델
        sparse_encoder (Any): 문서와 쿼리를 희소 벡터로 변환하는 인코더 (예: BM25Encoder)
        index (Any): 검색에 사용할 Pinecone 인덱스 객체
        top_k (int): 검색 결과로 반환할 최대 문서 수 (기본값: 10)
        alpha (float): 밀집 벡터와 희소 벡터의 가중치를 조절하는 파라미터 (0 에서 1 사이, 기본값: 0.5),  alpha=0.75로 설정한 경우, (0.75: Dense Embedding, 0.25: Sparse Embedding)
        namespace (Optional[str]): Pinecone 인덱스 내에서 사용할 네임스페이스 (기본값: None)
    """

    embeddings: Embeddings
    sparse_encoder: Any
    index: Any
    top_k: int = 10
    alpha: float = 0.5
    namespace: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_environment(cls, values: Dict) -> Dict:
        """
        필요한 패키지가 설치되어 있는지 확인하는 메서드입니다.

        Returns:
            Dict: 유효성 검사를 통과한 값들의 딕셔너리
        """
        try:
            from pinecone_text.hybrid import hybrid_convex_scale
            from pinecone_text.sparse.base_sparse_encoder import BaseSparseEncoder
        except ImportError:
            raise ImportError(
                "Could not import pinecone_text python package. "
                "Please install it with `pip install pinecone_text`."
            )
        return values

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **search_kwargs,
    ) -> List[Document]:
        """
        주어진 쿼리에 대해 관련 문서를 검색하는 메인 메서드입니다.

        Args:
            query (str): 검색 쿼리
            run_manager (CallbackManagerForRetrieverRun): 콜백 관리자
            **search_kwargs: 추가 검색 매개변수

        Returns:
            List[Document]: 관련 문서 리스트
        """
        alpha = self._get_alpha(search_kwargs)
        dense_vec, sparse_vec = self._encode_query(query, alpha)
        query_params = self._build_query_params(
            dense_vec, sparse_vec, search_kwargs, include_metadata=True
        )

        query_response = self.index.query(**query_params)
        # print("namespace", self.namespace)

        documents = self._process_query_response(query_response)

        # Rerank 옵션이 있는 경우 rerank 수행
        if (
            "search_kwargs" in search_kwargs
            and "rerank" in search_kwargs["search_kwargs"]
        ):
            documents = self._rerank_documents(query, documents, **search_kwargs)

        return documents

    def _get_alpha(self, search_kwargs: Dict[str, Any]) -> float:
        """
        알파 값을 가져오는 메서드입니다.

        Args:
            search_kwargs (Dict[str, Any]): 검색 매개변수

        Returns:
            float: 알파 값
        """
        if (
            "search_kwargs" in search_kwargs
            and "alpha" in search_kwargs["search_kwargs"]
        ):
            return search_kwargs["search_kwargs"]["alpha"]
        return self.alpha

    def _encode_query(
        self, query: str, alpha: float
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        쿼리를 인코딩하는 메서드입니다.

        Args:
            query (str): 인코딩할 쿼리
            alpha (float): 하이브리드 스케일링에 사용할 알파 값

        Returns:
            Tuple[List[float], Dict[str, Any]]: 밀집 벡터와 희소 벡터의 튜플
        """
        sparse_vec = self.sparse_encoder.encode_queries(query)
        dense_vec = self.embeddings.embed_query(query)
        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, alpha=alpha)
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]
        return dense_vec, sparse_vec

    def _build_query_params(
        self,
        dense_vec: List[float],
        sparse_vec: Dict[str, Any],
        search_kwargs: Dict[str, Any],
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """
        쿼리 파라미터를 구성하는 메서드입니다.

        Args:
            dense_vec (List[float]): 밀집 벡터
            sparse_vec (Dict[str, Any]): 희소 벡터
            search_kwargs (Dict[str, Any]): 검색 매개변수
            include_metadata (bool): 메타데이터 포함 여부

        Returns:
            Dict[str, Any]: 구성된 쿼리 파라미터
        """
        query_params = {
            "vector": dense_vec,
            "sparse_vector": sparse_vec,
            "top_k": self.top_k,
            "include_metadata": include_metadata,
            "namespace": self.namespace,
        }

        if "search_kwargs" in search_kwargs:
            kwargs = search_kwargs["search_kwargs"]
            query_params.update(
                {
                    "filter": kwargs.get("filter", query_params.get("filter")),
                    "top_k": kwargs.get("top_k")
                    or kwargs.get("k", query_params["top_k"]),
                }
            )

        return query_params

    def _process_query_response(self, query_response: Dict[str, Any]) -> List[Document]:
        """
        쿼리 응답을 처리하는 메서드입니다.

        Args:
            query_response (Dict[str, Any]): Pinecone 쿼리 응답

        Returns:
            List[Document]: 처리된 문서 리스트
        """
        return [
            Document(
                page_content=r.metadata["context"],
                metadata={
                    "page": r.metadata.get("page"),
                    "source": r.metadata.get("source"),
                    "score": r.get("score"),
                },
            )
            for r in query_response["matches"]
        ]

    def _rerank_documents(
        self, query: str, documents: List[Document], **kwargs
    ) -> List[Document]:
        """
        검색된 문서를 재정렬하는 메서드입니다.

        Args:
            query (str): 검색 쿼리
            documents (List[Document]): 재정렬할 문서 리스트
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 재정렬된 문서 리스트
        """
        print("[rerank_documents]")
        rerank_model = kwargs.get("rerank_model", "bge-reranker-v2-m3")
        top_n = kwargs.get("top_n", len(documents))

        rerank_docs = [
            {"id": str(i), "text": doc.page_content} for i, doc in enumerate(documents)
        ]

        result = self.index.inference.rerank(
            model=rerank_model,
            query=query,
            documents=rerank_docs,
            top_n=top_n,
            return_documents=True,
        )

        # 재정렬된 결과를 기반으로 문서 리스트 재구성
        reranked_documents = []
        for item in result:
            original_doc = documents[int(item["id"])]
            reranked_doc = Document(
                page_content=original_doc.page_content,
                metadata={**original_doc.metadata, "rerank_score": item["score"]},
            )
            reranked_documents.append(reranked_doc)

        return reranked_documents
