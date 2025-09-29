import logging
from typing import List, Union

from sentence_transformers import SentenceTransformer

from core.config import get_settings
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk

logger = logging.getLogger(__name__)


class SentenceTransformersEmbeddingModel(BaseEmbeddingModel):
    """
    Direct sentence-transformers embedding model implementation for local models.
    Bypasses LiteLLM for local sentence-transformers models.
    """

    def __init__(self, model_key: str):
        """
        Initialize sentence-transformers embedding model with a model key from registered_models.

        Args:
            model_key: The key of the model in the registered_models config
        """
        settings = get_settings()
        self.model_key = model_key

        # Get the model configuration from registered_models
        if not hasattr(settings, "REGISTERED_MODELS") or model_key not in settings.REGISTERED_MODELS:
            raise ValueError(f"Model '{model_key}' not found in registered_models configuration")

        self.model_config = settings.REGISTERED_MODELS[model_key]
        self.dimensions = settings.VECTOR_DIMENSIONS

        # Extract model path from config
        if "model_path" in self.model_config:
            model_path = self.model_config["model_path"]
        elif "model_name" in self.model_config and self.model_config["model_name"].startswith("/"):
            model_path = self.model_config["model_name"]
        else:
            # Fallback to model_name if no explicit path
            model_path = self.model_config.get("model_name", model_key)

        logger.info(f"Loading sentence-transformers model from: {model_path}")

        try:
            self.model = SentenceTransformer(model_path)
            logger.info(f"Successfully loaded sentence-transformers model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers model from {model_path}: {e}")
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents using sentence-transformers.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors (one per document)
        """
        if not texts:
            return []

        try:
            # Use sentence-transformers to encode the texts
            embeddings = self.model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)

            # Convert to list of lists for consistency
            if embeddings.ndim == 1:
                embeddings = [embeddings.tolist()]
            else:
                embeddings = embeddings.tolist()

            # Validate dimensions
            if embeddings and len(embeddings[0]) != self.dimensions:
                logger.warning(
                    f"Embedding dimension mismatch: got {len(embeddings[0])}, expected {self.dimensions}. "
                    f"Please update your VECTOR_DIMENSIONS setting to match the actual dimension."
                )

            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with sentence-transformers: {e}")
            raise

    async def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query using sentence-transformers.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        result = await self.embed_documents([text])
        if not result:
            # In case of error, return zero vector
            return [0.0] * self.dimensions
        return result[0]

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[List[float]]:
        """
        Generate embeddings for chunks to be ingested into the vector store.

        Args:
            chunks: Single chunk or list of chunks to embed

        Returns:
            List of embedding vectors (one per chunk)
        """
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        texts = [chunk.content for chunk in chunks]
        # Batch embedding to respect memory limits
        settings = get_settings()
        batch_size = getattr(settings, "EMBEDDING_BATCH_SIZE", 100)
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = await self.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
        return embeddings

    async def embed_for_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return await self.embed_query(text)