import base64
import io
import logging
import time
from typing import List, Union

import numpy as np
import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from PIL.Image import Image
from PIL.Image import open as open_image

from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.chunk import Chunk

logger = logging.getLogger(__name__)


class ColpaliEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        start_time = time.time()
        model_name = "vidore/colSmol-256M"
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ColpaliEmbeddingModel with device: {device}")
        self.model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,  # "cuda:0",  # or "mps" if on Apple Silicon
            attn_implementation="flash_attention_2",
        ).eval()
        model_load_time = time.time() - start_time
        logger.info(f"Model loading took {model_load_time:.2f} seconds")

        processor_start = time.time()
        self.processor = ColIdefics3Processor.from_pretrained(model_name)
        processor_time = time.time() - processor_start
        logger.info(f"Processor loading took {processor_time:.2f} seconds")

        self.batch_size = 8
        total_init_time = time.time() - start_time
        logger.info(f"Total initialization time: {total_init_time:.2f} seconds")

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[np.ndarray]:
        start_time = time.time()
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        logger.info(f"Processing {len(chunks)} chunks for embedding")

        # Separate images and texts immediately
        images = []
        texts = []

        sorting_start = time.time()
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

        sorting_time = time.time() - sorting_start
        logger.info(f"Chunk sorting took {sorting_time:.2f}s - Found {len(images)} images and {len(texts)} text chunks")

        # Process in batches
        embeddings = []

        # Process image batches
        if images:
            img_start = time.time()
            for i in range(0, len(images), self.batch_size):
                batch = images[i : i + self.batch_size]
                logger.debug(
                    f"Processing image batch {i//self.batch_size + 1}/"
                    f"{(len(images)-1)//self.batch_size + 1} with {len(batch)} images"
                )
                batch_start = time.time()
                batch_embeddings = await self.generate_embeddings_batch_images(batch)
                embeddings.extend(batch_embeddings)
                batch_time = time.time() - batch_start
                logger.debug(
                    f"Image batch {i//self.batch_size + 1} processing took {batch_time:.2f}s "
                    f"({batch_time/len(batch):.2f}s per image)"
                )
            img_time = time.time() - img_start
            logger.info(f"All image embedding took {img_time:.2f}s ({img_time/len(images):.2f}s per image)")

        # Process text batches
        if texts:
            text_start = time.time()
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                logger.debug(
                    f"Processing text batch {i//self.batch_size + 1}/"
                    f"{(len(texts)-1)//self.batch_size + 1} with {len(batch)} texts"
                )
                batch_start = time.time()
                batch_embeddings = await self.generate_embeddings_batch_texts(batch)
                embeddings.extend(batch_embeddings)
                batch_time = time.time() - batch_start
                logger.debug(
                    f"Text batch {i//self.batch_size + 1} processing took {batch_time:.2f}s "
                    f"({batch_time/len(batch):.2f}s per text)"
                )
            text_time = time.time() - text_start
            logger.info(f"All text embedding took {text_time:.2f}s ({text_time/len(texts):.2f}s per text)")

        total_time = time.time() - start_time
        logger.info(
            f"Total embed_for_ingestion took {total_time:.2f}s for {len(chunks)} chunks "
            f"({total_time/len(chunks):.2f}s per chunk)"
        )

        return embeddings

    async def embed_for_query(self, text: str) -> torch.Tensor:
        start_time = time.time()
        result = await self.generate_embeddings(text)
        elapsed = time.time() - start_time
        logger.info(f"Query embedding took {elapsed:.2f}s")
        return result

    async def generate_embeddings(self, content: str | Image) -> np.ndarray:
        start_time = time.time()
        content_type = "image" if isinstance(content, Image) else "text"

        process_start = time.time()
        if isinstance(content, Image):
            processed = self.processor.process_images([content]).to(self.model.device)
        else:
            processed = self.processor.process_queries([content]).to(self.model.device)
        process_time = time.time() - process_start
        logger.debug(f"Processing {content_type} took {process_time:.2f}s")

        model_start = time.time()
        with torch.no_grad():
            embeddings: torch.Tensor = self.model(**processed)
        model_time = time.time() - model_start
        logger.debug(f"Model inference for {content_type} took {model_time:.2f}s")

        convert_start = time.time()
        result = embeddings.to(torch.float32).numpy(force=True)[0]
        convert_time = time.time() - convert_start
        logger.debug(f"Tensor conversion took {convert_time:.2f}s")

        total_time = time.time() - start_time
        logger.debug(f"Total generate_embeddings for {content_type} took {total_time:.2f}s")

        return result

    async def generate_embeddings_batch_images(self, images: List[Image]) -> List[np.ndarray]:
        start_time = time.time()

        process_start = time.time()
        processed_images = self.processor.process_images(images).to(self.model.device)
        process_time = time.time() - process_start
        logger.debug(
            f"Processing {len(images)} images took {process_time:.2f}s " f"({process_time/len(images):.2f}s per image)"
        )

        model_start = time.time()
        with torch.no_grad():
            image_embeddings = self.model(**processed_images)
        model_time = time.time() - model_start
        logger.debug(
            f"Model inference for {len(images)} images took {model_time:.2f}s "
            f"({model_time/len(images):.2f}s per image)"
        )

        convert_start = time.time()
        image_embeddings = image_embeddings.to(torch.float32).numpy(force=True)
        result = [emb for emb in image_embeddings]
        convert_time = time.time() - convert_start
        logger.debug(f"Tensor conversion took {convert_time:.2f}s")

        total_time = time.time() - start_time
        logger.debug(
            f"Total batch processing for {len(images)} images took {total_time:.2f}s "
            f"({total_time/len(images):.2f}s per image)"
        )

        return result

    async def generate_embeddings_batch_texts(self, texts: List[str]) -> List[np.ndarray]:
        start_time = time.time()

        process_start = time.time()
        processed_texts = self.processor.process_queries(texts).to(self.model.device)
        process_time = time.time() - process_start
        logger.debug(
            f"Processing {len(texts)} texts took {process_time:.2f}s " f"({process_time/len(texts):.2f}s per text)"
        )

        model_start = time.time()
        with torch.no_grad():
            text_embeddings = self.model(**processed_texts)
        model_time = time.time() - model_start
        logger.debug(
            f"Model inference for {len(texts)} texts took {model_time:.2f}s " f"({model_time/len(texts):.2f}s per text)"
        )

        convert_start = time.time()
        text_embeddings = text_embeddings.to(torch.float32).numpy(force=True)
        result = [emb for emb in text_embeddings]
        convert_time = time.time() - convert_start
        logger.debug(f"Tensor conversion took {convert_time:.2f}s")

        total_time = time.time() - start_time
        logger.debug(
            f"Total batch processing for {len(texts)} texts took {total_time:.2f}s "
            f"({total_time/len(texts):.2f}s per text)"
        )

        return result
