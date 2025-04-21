import base64
import io
from typing import List, Union

import numpy as np
import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from PIL.Image import Image, open as open_image

from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk
import logging

logger = logging.getLogger(__name__)


class ColpaliEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        model_name = "vidore/colSmol-256M"
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,  # "cuda:0",  # or "mps" if on Apple Silicon
            attn_implementation="eager",  # "flash_attention_2" if is_flash_attn_2_available() else None,  # or "eager" if "mps"
        ).eval()
        self.processor = ColIdefics3Processor.from_pretrained(model_name)
        self.batch_size = 10  # Setting batch size to 10 as requested

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[np.ndarray]:
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        # Separate images and texts immediately
        images = []
        texts = []
        
        for chunk in chunks:
            if chunk.metadata.get("is_image"):
                try:
                    # Handle data URI format "data:image/png;base64,..."
                    content = chunk.content
                    if content.startswith("data:"):
                        # Extract the base64 part after the comma
                        content = content.split(",", 1)[1]

                    # Now decode the base64 string
                    image_bytes = base64.b64decode(content)
                    image = open_image(io.BytesIO(image_bytes))
                    images.append(image)
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    # Fall back to using the content as text
                    texts.append(chunk.content)
            else:
                texts.append(chunk.content)

        # Process in batches
        embeddings = []
        
        # Process image batches
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_embeddings = await self.generate_embeddings_batch_images(batch)
            embeddings.extend(batch_embeddings)
            
        # Process text batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self.generate_embeddings_batch_texts(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings

    async def embed_for_query(self, text: str) -> torch.Tensor:
        return await self.generate_embeddings(text)

    async def generate_embeddings(self, content: str | Image) -> np.ndarray:
        if isinstance(content, Image):
            processed = self.processor.process_images([content]).to(self.model.device)
        else:
            processed = self.processor.process_queries([content]).to(self.model.device)

        with torch.no_grad():
            embeddings: torch.Tensor = self.model(**processed)

        return embeddings.to(torch.float32).numpy(force=True)[0]
        
    async def generate_embeddings_batch_images(self, images: List[Image]) -> List[np.ndarray]:
        processed_images = self.processor.process_images(images).to(self.model.device)
        with torch.no_grad():
            image_embeddings = self.model(**processed_images)
        image_embeddings = image_embeddings.to(torch.float32).numpy(force=True)
        return [emb for emb in image_embeddings]
    
    async def generate_embeddings_batch_texts(self, texts: List[str]) -> List[np.ndarray]:
        processed_texts = self.processor.process_queries(texts).to(self.model.device)
        with torch.no_grad():
            text_embeddings = self.model(**processed_texts)
        text_embeddings = text_embeddings.to(torch.float32).numpy(force=True)
        return [emb for emb in text_embeddings]
