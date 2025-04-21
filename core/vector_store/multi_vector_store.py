from typing import List, Optional, Tuple, Union, ContextManager
import logging
import torch
import numpy as np
import time
from contextlib import contextmanager
import psycopg
from pgvector.psycopg import Bit, register_vector
from core.models.chunk import DocumentChunk
from .base_vector_store import BaseVectorStore
from time import perf_counter

logger = logging.getLogger(__name__)

# Create a dedicated logger for worker ingestion profiling
worker_logger = logging.getLogger("core.workers.ingestion_worker")


class MultiVectorStore(BaseVectorStore):
    """PostgreSQL implementation for storing and querying multi-vector embeddings using psycopg."""

    def __init__(
        self,
        uri: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize PostgreSQL connection for multi-vector storage.

        Args:
            uri: PostgreSQL connection URI
            max_retries: Maximum number of connection retry attempts
            retry_delay: Delay in seconds between retry attempts
        """
        # Convert SQLAlchemy URI to psycopg format if needed
        if uri.startswith("postgresql+asyncpg://"):
            uri = uri.replace("postgresql+asyncpg://", "postgresql://")
        self.uri = uri
        self.conn = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Don't initialize here - initialization will be handled separately
        
    @contextmanager
    def get_connection(self):
        """Get a PostgreSQL connection with retry logic.
        
        Yields:
            A PostgreSQL connection object
            
        Raises:
            psycopg.OperationalError: If all connection attempts fail
        """
        attempt = 0
        last_error = None
        
        # Try to establish a new connection with retries
        while attempt < self.max_retries:
            try:
                # Always create a fresh connection for critical operations
                conn = psycopg.connect(self.uri, autocommit=True)
                # Register vector extension on every new connection
                register_vector(conn)
                
                try:
                    yield conn
                    return
                finally:
                    # Always close connections after use
                    try:
                        conn.close()
                    except Exception:
                        pass
            except psycopg.OperationalError as e:
                last_error = e
                attempt += 1
                if attempt < self.max_retries:
                    logger.warning(f"Connection attempt {attempt} failed: {str(e)}. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
        
        # If we get here, all retries failed
        logger.error(f"All connection attempts failed after {self.max_retries} retries: {str(last_error)}")
        raise last_error

    def initialize(self):
        """Initialize database tables and max_sim function."""
        try:
            # Use the connection with retry logic
            with self.get_connection() as conn:
                # Register vector extension
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                register_vector(conn)

            # First check if the table exists and if it has the required columns
            with self.get_connection() as conn:
                check_table = conn.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'multi_vector_embeddings'
                    );
                """
                ).fetchone()[0]

                if check_table:
                    # Check if document_id column exists
                    has_document_id = conn.execute(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_name = 'multi_vector_embeddings' AND column_name = 'document_id'
                        );
                    """
                    ).fetchone()[0]

                    # If the table exists but doesn't have document_id, we need to add the required columns
                    if not has_document_id:
                        logger.info("Updating multi_vector_embeddings table with required columns")
                        conn.execute(
                            """
                            ALTER TABLE multi_vector_embeddings 
                            ADD COLUMN document_id TEXT,
                            ADD COLUMN chunk_number INTEGER,
                            ADD COLUMN content TEXT,
                            ADD COLUMN chunk_metadata TEXT
                        """
                        )
                        conn.execute(
                            """
                            ALTER TABLE multi_vector_embeddings 
                            ALTER COLUMN document_id SET NOT NULL
                        """
                        )

                        # Add a commit to ensure changes are applied
                        conn.commit()
                else:
                    # Create table if it doesn't exist with all required columns
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS multi_vector_embeddings (
                            id BIGSERIAL PRIMARY KEY,
                            document_id TEXT NOT NULL,
                            chunk_number INTEGER NOT NULL,
                            content TEXT NOT NULL,
                            chunk_metadata TEXT,
                            embeddings BIT(128)[]
                        )
                    """
                    )

                # Add a commit to ensure table creation is complete
                conn.commit()

            try:
                # Create index on document_id
                with self.get_connection() as conn:
                    conn.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_multi_vector_document_id 
                        ON multi_vector_embeddings (document_id)
                    """
                    )
            except Exception as e:
                # Log index creation failure but continue
                logger.warning(f"Failed to create index: {str(e)}")

            try:
                # First, try to drop the existing function if it exists
                with self.get_connection() as conn:
                    conn.execute(
                        """
                        DROP FUNCTION IF EXISTS max_sim(bit[], bit[])
                    """
                    )
                    logger.info("Dropped existing max_sim function")

                    # Create max_sim function
                    conn.execute(
                        """
                        CREATE OR REPLACE FUNCTION max_sim(document bit[], query bit[]) RETURNS double precision AS $$
                            WITH queries AS (
                                SELECT row_number() OVER () AS query_number, * FROM (SELECT unnest(query) AS query) AS foo
                            ),
                            documents AS (
                                SELECT unnest(document) AS document
                            ),
                            similarities AS (
                                SELECT 
                                    query_number, 
                                    1.0 - (bit_count(document # query)::float / greatest(bit_length(query), 1)::float) AS similarity
                                FROM queries CROSS JOIN documents
                            ),
                            max_similarities AS (
                                SELECT MAX(similarity) AS max_similarity FROM similarities GROUP BY query_number
                            )
                            SELECT SUM(max_similarity) FROM max_similarities
                        $$ LANGUAGE SQL
                    """
                    )
                    logger.info("Created max_sim function successfully")
            except Exception as e:
                logger.error(f"Error creating max_sim function: {str(e)}")
                # Continue even if function creation fails - it might already exist and be usable

            logger.info("MultiVectorStore initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing MultiVectorStore: {str(e)}")
            return False

    def _binary_quantize(self, embeddings: Union[np.ndarray, torch.Tensor, List]) -> List[Bit]:
        """Convert embeddings to binary format for PostgreSQL BIT[] arrays."""
        # Start profiling
        start_time = perf_counter()
        
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(embeddings, list) and not isinstance(embeddings[0], np.ndarray):
            embeddings = np.array(embeddings)
        
        # Add this check to ensure pgvector is registered for the connection
        with self.get_connection() as conn:
            register_vector(conn)
        
        result = [Bit(embedding > 0) for embedding in embeddings]
        
        # End profiling and log if called from store_embeddings
        end_time = perf_counter()
        self._last_quantize_time = end_time - start_time
        
        return result

    async def store_embeddings(self, chunks: List[DocumentChunk]) -> Tuple[bool, List[str]]:
        """Store document chunks with their multi-vector embeddings."""
        # Start overall profiling
        total_start_time = perf_counter()
        
        # Initialize timing variables
        quantize_total_time = 0
        db_operation_total_time = 0
        connection_total_time = 0
        
        # Track per-chunk times to identify outliers
        chunk_times = []
        chunk_db_times = []
        chunk_sizes = []
        
        if not chunks:
            worker_logger.info("MultiVectorStore.store_embeddings: No chunks to store")
            return True, []

        stored_ids = []
        
        # Start loop profiling
        loop_start_time = perf_counter()
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_start_time = perf_counter()
            
            # Ensure embeddings exist
            if not hasattr(chunk, "embedding") or chunk.embedding is None:
                logger.error(
                    f"Missing embeddings for chunk {chunk.document_id}-{chunk.chunk_number}"
                )
                continue

            # For multi-vector embeddings, we expect a list of vectors
            embeddings = chunk.embedding
            
            # Track embedding size for analysis
            embedding_size = len(embeddings) if isinstance(embeddings, list) else 1
            chunk_sizes.append(embedding_size)

            # Create binary representation for each vector
            # Reset the last quantize time tracker
            self._last_quantize_time = 0
            binary_embeddings = self._binary_quantize(embeddings)
            # Capture the time from the internal tracker
            quantize_time = self._last_quantize_time
            quantize_total_time += quantize_time

            # Insert into database with retry logic - track time
            db_start_time = perf_counter()
            
            # Track connection establishment time
            conn_start_time = perf_counter()
            with self.get_connection() as conn:
                conn_time = perf_counter() - conn_start_time
                connection_total_time += conn_time
                
                # Track actual query execution time
                query_start_time = perf_counter()
                conn.execute(
                    """
                    INSERT INTO multi_vector_embeddings 
                    (document_id, chunk_number, content, chunk_metadata, embeddings) 
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        chunk.document_id,
                        chunk.chunk_number,
                        chunk.content,
                        str(chunk.metadata),
                        binary_embeddings,
                    ),
                )
                query_time = perf_counter() - query_start_time
            
            db_time = perf_counter() - db_start_time
            db_operation_total_time += db_time
            
            # Track chunk processing time
            chunk_time = perf_counter() - chunk_start_time
            chunk_times.append(chunk_time)
            chunk_db_times.append(db_time)

            stored_ids.append(f"{chunk.document_id}-{chunk.chunk_number}")
            
            # Log detailed metrics for specific chunks (first, middle, last)
            if chunk_idx == 0 or chunk_idx == len(chunks) // 2 or chunk_idx == len(chunks) - 1:
                worker_logger.info(f"MultiVectorStore chunk {chunk_idx+1}/{len(chunks)} profiling: " 
                                 f"Total={chunk_time:.4f}s, "
                                 f"Quantize={quantize_time:.4f}s ({quantize_time*100/chunk_time:.1f}%), "
                                 f"DB={db_time:.4f}s ({db_time*100/chunk_time:.1f}%), "
                                 f"Conn={conn_time:.4f}s ({conn_time*100/db_time:.1f}% of DB), "
                                 f"Query={query_time:.4f}s ({query_time*100/db_time:.1f}% of DB), "
                                 f"Vectors={embedding_size}")

        loop_time = perf_counter() - loop_start_time
        total_time = perf_counter() - total_start_time
        
        # Calculate statistics for chunk processing times
        avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0
        max_chunk_time = max(chunk_times) if chunk_times else 0
        min_chunk_time = min(chunk_times) if chunk_times else 0
        
        avg_db_time = sum(chunk_db_times) / len(chunk_db_times) if chunk_db_times else 0
        max_db_time = max(chunk_db_times) if chunk_db_times else 0
        
        avg_vectors = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        max_vectors = max(chunk_sizes) if chunk_sizes else 0
        
        # Log detailed profiling information
        worker_logger.info(f"MultiVectorStore.store_embeddings profiling: Total time={total_time:.4f}s for {len(chunks)} chunks")
        worker_logger.info(f"  - Binary quantization: {quantize_total_time:.4f}s ({quantize_total_time*100/total_time:.1f}%)")
        worker_logger.info(f"  - Database operations: {db_operation_total_time:.4f}s ({db_operation_total_time*100/total_time:.1f}%)")
        worker_logger.info(f"  - Connection overhead: {connection_total_time:.4f}s ({connection_total_time*100/db_operation_total_time:.1f}% of DB time)")
        worker_logger.info(f"  - For loop overhead: {loop_time - quantize_total_time - db_operation_total_time:.4f}s")
        worker_logger.info(f"  - Chunk processing stats: Avg={avg_chunk_time:.4f}s, Min={min_chunk_time:.4f}s, Max={max_chunk_time:.4f}s")
        worker_logger.info(f"  - DB operation stats: Avg={avg_db_time:.4f}s, Max={max_db_time:.4f}s")
        worker_logger.info(f"  - Vector count stats: Avg={avg_vectors:.1f}, Max={max_vectors}")
        
        logger.debug(f"{len(stored_ids)} vector embeddings added successfully!")
        return len(stored_ids) > 0, stored_ids

    async def query_similar(
        self,
        query_embedding: Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
        k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[DocumentChunk]:
        """Find similar chunks using the max_sim function for multi-vectors."""
        # try:
        # Convert query embeddings to binary format
        binary_query_embeddings = self._binary_quantize(query_embedding)

        # Build query
        query = """
            SELECT id, document_id, chunk_number, content, chunk_metadata, 
                    max_sim(embeddings, %s) AS similarity
            FROM multi_vector_embeddings
        """

        params = [binary_query_embeddings]

        # Add document filter if needed with proper parameterization
        if doc_ids:
            # Use placeholders for each document ID
            placeholders = ', '.join(['%s'] * len(doc_ids))
            query += f" WHERE document_id IN ({placeholders})"
            # Add document IDs to params
            params.extend(doc_ids)

        # Add ordering and limit
        query += " ORDER BY similarity DESC LIMIT %s"
        params.append(k)

        # Execute query with retry logic
        with self.get_connection() as conn:
            result = conn.execute(query, params).fetchall()

        # Convert to DocumentChunks
        chunks = []
        for row in result:
            try:
                metadata = eval(row[4]) if row[4] else {}
            except (ValueError, SyntaxError):
                metadata = {}

            chunk = DocumentChunk(
                document_id=row[1],
                chunk_number=row[2],
                content=row[3],
                embedding=[],  # Don't send embeddings back
                metadata=metadata,
                score=float(row[5]),  # Use the similarity score from max_sim
            )
            chunks.append(chunk)

        return chunks

        # except Exception as e:
        #     logger.error(f"Error querying similar chunks: {str(e)}")
        #     raise e
        #     return []

    async def get_chunks_by_id(
        self,
        chunk_identifiers: List[Tuple[str, int]],
    ) -> List[DocumentChunk]:
        """
        Retrieve specific chunks by document ID and chunk number in a single database query.
        
        Args:
            chunk_identifiers: List of (document_id, chunk_number) tuples
            
        Returns:
            List of DocumentChunk objects
        """
        # try:
        if not chunk_identifiers:
            return []
            
        # Construct the WHERE clause with OR conditions
        conditions = []
        for doc_id, chunk_num in chunk_identifiers:
            conditions.append(f"(document_id = '{doc_id}' AND chunk_number = {chunk_num})")
        
        where_clause = " OR ".join(conditions)
        
        # Build and execute query
        query = f"""
            SELECT document_id, chunk_number, content, chunk_metadata
            FROM multi_vector_embeddings
            WHERE {where_clause}
        """
        
        logger.debug(f"Batch retrieving {len(chunk_identifiers)} chunks from multi-vector store")
        
        with self.get_connection() as conn:
            result = conn.execute(query).fetchall()
        
        # Convert to DocumentChunks
        chunks = []
        for row in result:
            try:
                metadata = eval(row[3]) if row[3] else {}
            except (ValueError, SyntaxError):
                metadata = {}
                
            chunk = DocumentChunk(
                document_id=row[0],
                chunk_number=row[1],
                content=row[2],
                embedding=[],  # Don't send embeddings back
                metadata=metadata,
                score=0.0,  # No relevance score for direct retrieval
            )
            chunks.append(chunk)
            
        logger.debug(f"Found {len(chunks)} chunks in batch retrieval from multi-vector store")
        return chunks
    
    async def delete_chunks_by_document_id(self, document_id: str) -> bool:
        """
        Delete all chunks associated with a document.
        
        Args:
            document_id: ID of the document whose chunks should be deleted
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        try:
            # Delete all chunks for the specified document with retry logic
            query = f"DELETE FROM multi_vector_embeddings WHERE document_id = '{document_id}'"
            with self.get_connection() as conn:
                conn.execute(query)
            
            logger.info(f"Deleted all chunks for document {document_id} from multi-vector store")
            return True
                
        except Exception as e:
            logger.error(f"Error deleting chunks for document {document_id} from multi-vector store: {str(e)}")
            return False
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")
