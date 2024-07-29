import time
import pickle
import secrets
from tqdm.auto import tqdm

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator
from langchain_core.retrievers import BaseRetriever
from typing import List, Dict, Any, Optional
from langchain_core.embeddings import Embeddings

from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse import BM25Encoder

from .kiwi_tokenizer import KiwiBM25Tokenizer


def generate_hash():
    # 24자리 무작위 hex 값 생성
    random_hex = secrets.token_hex(12)  # 12바이트는 24자리 hex로 변환됩니다.

    # 6자리씩 나누어 '-'로 연결
    formatted_hash = "-".join(random_hex[i : i + 6] for i in range(0, 24, 6))

    return formatted_hash


def create_index(api_key, index_name, dimension, metric="dotproduct"):
    """
    Args:
        api_key (str): Pinecone API key.
        index_name (str): Pinecone index name.
        dimension (int): Pinecone index dimension.
        metric (str, optional): 'cosine', 'euclidean', 'dotproduct'. Defaults to "dotproduct".
    """
    # Pinecone Client Init
    pc = Pinecone(api_key=api_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,  # Index 이름
            dimension=dimension,  # Embedding 모델 dimension
            metric=metric,  # Metric
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # for free tier
        )

        # wait for index to be initialized
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    # connect to index
    index = pc.Index(index_name)
    # view index stats
    print(f"[create_index]\n{index.describe_index_stats()}")
    return index


def create_sparse_encoder(stopwords: List[str], mode: str = "kiwi") -> BM25Encoder:
    bm25 = BM25Encoder(language="english")
    if mode == "kiwi":
        bm25._tokenizer = KiwiBM25Tokenizer(stop_words=stopwords)
    return bm25


def preprocess_documents(split_docs, metadata_keys: List[str] = ["source", "page"]):
    contents = []

    metadatas = dict()
    for key in metadata_keys:
        metadatas[key] = []

    for doc in tqdm(split_docs):
        content = doc.page_content.strip()
        metadata = doc.metadata
        if content:
            contents.append(content)
            for k in metadata.keys():
                if k in metadata_keys:
                    metadatas[k].append(metadata[k])
    return contents, metadatas


def fit_save_sparse_encoder(
    sparse_encoder: BM25Encoder, contents: List[str], save_path: str
) -> str:
    sparse_encoder.fit(contents)
    with open(save_path, "wb") as f:
        pickle.dump(sparse_encoder, f)

    print(f"[fit_save_sparse_encoder] Saved sparse encoder to {save_path}")
    return save_path


def load_sparse_encoder(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            loaded_file = pickle.load(f)

        print(f"[load_sparse_encoder] load sparse encoder from {file_path}")
        return loaded_file
    except Exception as e:
        print(f"[load_sparse_encoder] {e}")
        return None


def upsert_documents(
    index, namespace, contents, metadatas, sparse_encoder, embedder, batch_size=16
):

    keys = list(metadatas.keys())

    for i in tqdm(range(0, len(contents), batch_size)):
        # 배치의 끝 인덱스
        i_end = min(i + batch_size, len(contents))

        # 현재 배치에 해당하는 데이터 슬라이싱
        context_batch = contents[i:i_end]
        metadata_batches = {key: metadatas[key][i:i_end] for key in keys}

        # 현재 배치 데이터로 결과 생성
        batch_result = [
            {"context": context, **{key: metadata_batches[key][j] for key in keys}}
            for j, context in enumerate(context_batch)
        ]

        # create unique IDs
        ids = [generate_hash() for _ in range(i, i_end)]

        # create dense vectors
        dense_embeds = embedder.embed_documents(context_batch)
        # create sparse vectors
        sparse_embeds = sparse_encoder.encode_documents(context_batch)

        vectors = []
        # loop through the data and create dictionaries for uploading documents to pinecone index
        for _id, sparse, dense, metadata in zip(
            ids, sparse_embeds, dense_embeds, batch_result
        ):
            vectors.append(
                {
                    "id": _id,
                    "sparse_values": sparse,
                    "values": dense,
                    "metadata": metadata,
                }
            )

        # upload the documents to the new hybrid index
        index.upsert(vectors=vectors, namespace=namespace)

    # show index description after uploading the documents
    print(f"[upsert_documents]\n{index.describe_index_stats()}")


def init_pinecone_index(
    index_name: str,
    namespace: str,
    api_key: str,
    sparse_encoder_pkl_path: str = None,
    stopwords: List[str] = None,
    tokenizer: str = "kiwi",
    embeddings: Embeddings = None,
    top_k: int = 10,
    alpha: float = 0.5,
) -> Dict:

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    print(f"[init_pinecone_index]\n{index.describe_index_stats()}")
    try:
        with open(sparse_encoder_pkl_path, "rb") as f:
            bm25 = pickle.load(f)
        if tokenizer == "kiwi":
            bm25._tokenizer = KiwiBM25Tokenizer(stop_words=stopwords)
    except Exception as e:
        print(e)
        return {}
    namespace_keys = index.describe_index_stats()["namespaces"].keys()
    if namespace not in namespace_keys:
        raise ValueError(f"`{namespace}` not found in one of `{list(namespace_keys)}`")

    return {
        "index": index,
        "namespace": namespace,
        "sparse_encoder": bm25,
        "embeddings": embeddings,
        "top_k": top_k,
        "alpha": alpha,
    }


class PineconeKiwiHybridRetriever(BaseRetriever):
    embeddings: Embeddings
    """Embeddings model to use."""
    """description"""
    sparse_encoder: Any
    """Sparse encoder to use."""
    index: Any
    """Pinecone index to use."""
    top_k: int = 10
    """Number of documents to return."""
    alpha: float = 0.5
    """Alpha value for hybrid search."""
    namespace: Optional[str] = None
    """Namespace value for index partition."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            from pinecone_text.hybrid import hybrid_convex_scale  # noqa:F401
            from pinecone_text.sparse.base_sparse_encoder import (
                BaseSparseEncoder,  # noqa:F401
            )
        except ImportError:
            raise ImportError(
                "Could not import pinecone_text python package. "
                "Please install it with `pip install pinecone_text`."
            )
        return values

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        sparse_vec = self.sparse_encoder.encode_queries(query)
        dense_vec = self.embeddings.embed_query(query)

        dense_vec, sparse_vec = hybrid_convex_scale(
            dense_vec, sparse_vec, alpha=self.alpha
        )
        sparse_vec["values"] = [float(s1) for s1 in sparse_vec["values"]]

        query_response = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
            namespace=self.namespace,
        )
        print("namespace", self.namespace)
        final_result = []
        for r in query_response["matches"]:
            metadata = dict()
            if "page" in r.metadata:
                metadata["page"] = r.metadata["page"]
            if "source" in r.metadata:
                metadata["source"] = r.metadata["source"]
            if "score" in r:
                metadata["score"] = r["score"]
            doc = Document(page_content=r.metadata["context"], metadata=metadata)
            final_result.append(doc)
        return final_result
