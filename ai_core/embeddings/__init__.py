"""Local, reproducible profile embeddings using BAAI BGE-M3 (1024d)."""

from ai_core.embeddings.bge_m3 import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    MODEL_REVISION,
    TEMPLATE_VERSION,
    BgeM3Embedder,
    GteMultilingualEmbedder,
    build_profile_text,
    semantic_profile_hash,
)

__all__ = [
    "EMBEDDING_DIMENSION",
    "MODEL_REVISION",
    "MODEL_NAME",
    "TEMPLATE_VERSION",
    "BgeM3Embedder",
    "GteMultilingualEmbedder",
    "build_profile_text",
    "semantic_profile_hash",
]
