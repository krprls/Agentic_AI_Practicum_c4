"""Open-source vector store: ChromaDB + local HuggingFace sentence-transformer embeddings."""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def collection_count() -> int:
    try:
        return get_vectorstore()._collection.count()
    except Exception:  # noqa: BLE001
        return 0


def reset_collection() -> None:
    vs = get_vectorstore()
    vs.delete_collection()
    get_vectorstore.cache_clear()
