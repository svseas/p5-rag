import asyncio
import json
import logging
import time  # Add time import for profiling
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import arq
import jwt
import tomli
from fastapi import Depends, FastAPI, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from fastapi.responses import StreamingResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from core.agent import MorphikAgent
from core.app_factory import lifespan
from core.auth_utils import verify_token
from core.config import get_settings
from core.database.postgres_database import PostgresDatabase
from core.dependencies import get_redis_pool
from core.limits_utils import check_and_increment_limits
from core.logging_config import setup_logging
from core.middleware.profiling import ProfilingMiddleware
from core.models.auth import AuthContext, EntityType
from core.models.chat import ChatMessage
from core.models.completion import ChunkSource, CompletionResponse
from core.models.documents import ChunkResult, Document, DocumentResult, GroupedChunkResponse
from core.models.folders import Folder, FolderCreate, FolderSummary
from core.models.prompts import validate_prompt_overrides_with_http_exception
from core.models.request import (
    AgentQueryRequest,
    CompletionQueryRequest,
    GenerateUriRequest,
    IngestTextRequest,
    ListDocumentsRequest,
    RetrieveRequest,
    SetFolderRuleRequest,
)
from core.models.responses import (
    ChatTitleResponse,
    DocumentAddToFolderResponse,
    DocumentDeleteResponse,
    DocumentDownloadUrlResponse,
    FolderDeleteResponse,
    FolderRuleResponse,
    HealthCheckResponse,
    ModelsResponse,
)
from core.models.workflows import Workflow
from core.routes.cache import router as cache_router
from core.routes.document import router as document_router
from core.routes.graph import router as graph_router
from core.routes.ingest import router as ingest_router
from core.routes.logs import router as logs_router  # noqa: E402 – import after FastAPI app
from core.routes.model_config import router as model_config_router
from core.routes.models import router as models_router
from core.routes.workflow import router as workflow_router
from core.services.telemetry import TelemetryService
from core.services_init import document_service, workflow_service

# Set up logging configuration for Docker environment
setup_logging()

# Initialize FastAPI app
logger = logging.getLogger(__name__)


# Performance tracking class
class PerformanceTracker:
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = time.time()
        self.phases = {}
        self.current_phase = None
        self.phase_start = None

    def start_phase(self, phase_name: str):
        # End current phase if one is running
        if self.current_phase and self.phase_start:
            self.phases[self.current_phase] = time.time() - self.phase_start

        # Start new phase
        self.current_phase = phase_name
        self.phase_start = time.time()

    def end_phase(self):
        if self.current_phase and self.phase_start:
            self.phases[self.current_phase] = time.time() - self.phase_start
            self.current_phase = None
            self.phase_start = None

    def add_suboperation(self, name: str, duration: float):
        """Add a sub-operation timing"""
        self.phases[name] = duration

    def log_summary(self, additional_info: str = ""):
        total_time = time.time() - self.start_time

        # End current phase if still running
        if self.current_phase and self.phase_start:
            self.phases[self.current_phase] = time.time() - self.phase_start

        logger.info(f"=== {self.operation_name} Performance Summary ===")
        logger.info(f"Total time: {total_time:.2f}s")

        # Sort phases by duration (longest first)
        for phase, duration in sorted(self.phases.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            logger.info(f"  - {phase}: {duration:.2f}s ({percentage:.1f}%)")

        if additional_info:
            logger.info(additional_info)
        logger.info("=" * (len(self.operation_name) + 31))


# ---------------------------------------------------------------------------
# Application instance & core initialisation (moved lifespan, rest unchanged)
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

# --------------------------------------------------------
# Optional per-request profiler (ENABLE_PROFILING=1)
# --------------------------------------------------------

app.add_middleware(ProfilingMiddleware)

# Add CORS middleware (same behaviour as before refactor)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise telemetry service
telemetry = TelemetryService()

# OpenTelemetry instrumentation – exclude noisy spans/headers
FastAPIInstrumentor.instrument_app(
    app,
    excluded_urls="health,health/.*",
    exclude_spans=["send", "receive"],
    http_capture_headers_server_request=None,
    http_capture_headers_server_response=None,
    tracer_provider=None,
)

# Global settings object
settings = get_settings()

# ---------------------------------------------------------------------------
# Session cookie behaviour differs between cloud / self-hosted
# ---------------------------------------------------------------------------

if settings.MODE == "cloud":
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.SESSION_SECRET_KEY,
        same_site="none",
        https_only=True,
    )
else:
    app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET_KEY)


# Simple health check endpoint
@app.get("/ping", response_model=HealthCheckResponse)
async def ping_health():
    """Simple health check endpoint that returns 200 OK."""
    return {"status": "ok", "message": "Server is running"}


@app.get("/models", response_model=ModelsResponse)
async def get_available_models(auth: AuthContext = Depends(verify_token)):
    """
    Get list of available models from configuration.

    Returns models grouped by type (chat, embedding, etc.) with their metadata.
    """
    try:
        # Load the morphik.toml file to get registered models
        with open("morphik.toml", "rb") as f:
            config = tomli.load(f)

        registered_models = config.get("registered_models", {})

        # Group models by their purpose
        chat_models = []
        embedding_models = []

        for model_key, model_config in registered_models.items():
            model_info = {
                "id": model_key,
                "model": model_config.get("model_name", model_key),
                "provider": _extract_provider(model_config.get("model_name", "")),
                "config": model_config,
            }

            # Categorize models based on their names or configuration
            if "embedding" in model_key.lower():
                embedding_models.append(model_info)
            else:
                chat_models.append(model_info)

        # Also add the default configured models
        default_models = {
            "completion": config.get("completion", {}).get("model"),
            "agent": config.get("agent", {}).get("model"),
            "embedding": config.get("embedding", {}).get("model"),
        }

        return {
            "chat_models": chat_models,
            "embedding_models": embedding_models,
            "default_models": default_models,
            "providers": ["openai", "anthropic", "google", "azure", "ollama", "custom"],
        }
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail="Failed to load available models")


def _extract_provider(model_name: str) -> str:
    """Extract provider from model name."""
    if model_name.startswith("gpt"):
        return "openai"
    elif model_name.startswith("claude"):
        return "anthropic"
    elif model_name.startswith("gemini"):
        return "google"
    elif model_name.startswith("ollama"):
        return "ollama"
    elif "azure" in model_name:
        return "azure"
    else:
        return "custom"


# ---------------------------------------------------------------------------
# Core singletons (database, vector store, storage, parser, models …)
# ---------------------------------------------------------------------------


# Store on app.state for later access
app.state.document_service = document_service
logger.info("Document service initialized and stored on app.state")

# Register ingest router
app.include_router(ingest_router)

# Register document router
app.include_router(document_router)

# Register workflow router (step-2)
app.include_router(workflow_router)

# Register model config router
app.include_router(model_config_router)

# Register models router
app.include_router(models_router)

# Register logs router
app.include_router(logs_router)

# Register cache router
app.include_router(cache_router)

# Register graph router
app.include_router(graph_router)

# Single MorphikAgent instance (tool definitions cached)
morphik_agent = MorphikAgent(document_service=document_service)


# Helper function to normalize folder_name parameter
def normalize_folder_name(folder_name: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
    """Convert string 'null' to None for folder_name parameter."""
    if folder_name is None:
        return None
    if isinstance(folder_name, str):
        return None if folder_name.lower() == "null" else folder_name
    if isinstance(folder_name, list):
        return [None if f.lower() == "null" else f for f in folder_name]
    return folder_name


# Enterprise-only routes (optional)
try:
    from ee.routers import init_app as _init_ee_app  # type: ignore  # noqa: E402

    _init_ee_app(app)  # noqa: SLF001 – runtime extension
except ModuleNotFoundError as exc:
    logger.debug("Enterprise package not found – running in community mode.")
    logger.error("ModuleNotFoundError: %s", exc, exc_info=True)
except ImportError as exc:
    logger.error("Failed to import init_app from ee.routers: %s", exc, exc_info=True)
except Exception as exc:  # noqa: BLE001
    logger.error("An unexpected error occurred during EE app initialization: %s", exc, exc_info=True)


@app.post("/retrieve/chunks", response_model=List[ChunkResult])
@telemetry.track(operation_type="retrieve_chunks", metadata_resolver=telemetry.retrieve_chunks_metadata)
async def retrieve_chunks(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """
    Retrieve relevant chunks.

    Args:
        request: RetrieveRequest containing:
            - query: Search query text
            - filters: Optional metadata filters
            - k: Number of results (default: 4)
            - min_score: Minimum similarity threshold (default: 0.0)
            - use_reranking: Whether to use reranking
            - use_colpali: Whether to use ColPali-style embedding model
            - folder_name: Optional folder to scope the search to
            - end_user_id: Optional end-user ID to scope the search to
        auth: Authentication context

    Returns:
        List[ChunkResult]: List of relevant chunks
    """
    # Initialize performance tracker
    perf = PerformanceTracker(f"Retrieve Chunks: '{request.query[:50]}...'")

    try:
        # Main retrieval operation
        perf.start_phase("document_service_retrieve_chunks")
        results = await document_service.retrieve_chunks(
            request.query,
            auth,
            request.filters,
            request.k,
            request.min_score,
            request.use_reranking,
            request.use_colpali,
            request.folder_name,
            request.end_user_id,
            perf,  # Pass performance tracker
            request.padding,  # Pass padding parameter
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)} chunks")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/retrieve/chunks/grouped", response_model=GroupedChunkResponse)
@telemetry.track(operation_type="retrieve_chunks_grouped", metadata_resolver=telemetry.retrieve_chunks_metadata)
async def retrieve_chunks_grouped(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """
    Retrieve relevant chunks with grouped response format.

    Returns both flat results (for backward compatibility) and grouped results (for UI).
    When padding > 0, groups chunks by main matches and their padding chunks.

    Args:
        request: RetrieveRequest containing query, filters, padding, etc.
        auth: Authentication context

    Returns:
        GroupedChunkResponse: Contains both flat chunks and grouped chunks
    """
    # Initialize performance tracker
    perf = PerformanceTracker(f"Retrieve Chunks Grouped: '{request.query[:50]}...'")

    try:
        # Main retrieval operation
        perf.start_phase("document_service_retrieve_chunks_grouped")
        result = await document_service.retrieve_chunks_grouped(
            request.query,
            auth,
            request.filters,
            request.k,
            request.min_score,
            request.use_reranking,
            request.use_colpali,
            request.folder_name,
            request.end_user_id,
            perf,  # Pass performance tracker
            request.padding,  # Pass padding parameter
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(result.chunks)} total chunks in {len(result.groups)} groups")

        return result
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/retrieve/docs", response_model=List[DocumentResult])
@telemetry.track(operation_type="retrieve_docs", metadata_resolver=telemetry.retrieve_docs_metadata)
async def retrieve_documents(request: RetrieveRequest, auth: AuthContext = Depends(verify_token)):
    """
    Retrieve relevant documents.

    Args:
        request: RetrieveRequest containing:
            - query: Search query text
            - filters: Optional metadata filters
            - k: Number of results (default: 4)
            - min_score: Minimum similarity threshold (default: 0.0)
            - use_reranking: Whether to use reranking
            - use_colpali: Whether to use ColPali-style embedding model
            - folder_name: Optional folder to scope the search to
            - end_user_id: Optional end-user ID to scope the search to
        auth: Authentication context

    Returns:
        List[DocumentResult]: List of relevant documents
    """
    # Initialize performance tracker
    perf = PerformanceTracker(f"Retrieve Docs: '{request.query[:50]}...'")

    try:
        # Main retrieval operation
        perf.start_phase("document_service_retrieve_docs")
        results = await document_service.retrieve_docs(
            request.query,
            auth,
            request.filters,
            request.k,
            request.min_score,
            request.use_reranking,
            request.use_colpali,
            request.folder_name,
            request.end_user_id,
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)} documents")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/batch/documents", response_model=List[Document])
@telemetry.track(operation_type="batch_get_documents", metadata_resolver=telemetry.batch_documents_metadata)
async def batch_get_documents(batch_request: Dict[str, Any], auth: AuthContext = Depends(verify_token)):
    """
    Retrieve multiple documents by their IDs in a single batch operation.

    Args:
        batch_request: Dictionary containing:
            - document_ids: List of document IDs to retrieve
            - folder_name: Optional folder to scope the operation to
            - end_user_id: Optional end-user ID to scope the operation to
        auth: Authentication context

    Returns:
        List[Document]: List of documents matching the IDs
    """
    # Initialize performance tracker
    perf = PerformanceTracker("Batch Get Documents")

    try:
        # Extract document_ids from request
        perf.start_phase("request_extraction")
        document_ids = batch_request.get("document_ids", [])
        folder_name = batch_request.get("folder_name")
        end_user_id = batch_request.get("end_user_id")

        if not document_ids:
            perf.log_summary("No document IDs provided")
            return []

        # Create system filters for folder and user scoping
        perf.start_phase("filter_creation")
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id
        # Note: Don't add auth.app_id here - it's already handled in document retrieval

        # Main batch retrieval operation
        perf.start_phase("batch_retrieve_documents")
        results = await document_service.batch_retrieve_documents(document_ids, auth, folder_name, end_user_id)

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)}/{len(document_ids)} documents")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/batch/chunks", response_model=List[ChunkResult])
@telemetry.track(operation_type="batch_get_chunks", metadata_resolver=telemetry.batch_chunks_metadata)
async def batch_get_chunks(batch_request: Dict[str, Any], auth: AuthContext = Depends(verify_token)):
    """
    Retrieve specific chunks by their document ID and chunk number in a single batch operation.

    Args:
        request: Dictionary containing:
            - sources: List of ChunkSource objects (with document_id and chunk_number)
            - folder_name: Optional folder to scope the operation to
            - end_user_id: Optional end-user ID to scope the operation to
            - use_colpali: Whether to use ColPali-style embedding
        auth: Authentication context

    Returns:
        List[ChunkResult]: List of chunk results
    """
    # Initialize performance tracker
    perf = PerformanceTracker("Batch Get Chunks")

    try:
        # Extract sources from request
        perf.start_phase("request_extraction")
        sources = batch_request.get("sources", [])
        folder_name = batch_request.get("folder_name")
        end_user_id = batch_request.get("end_user_id")
        use_colpali = batch_request.get("use_colpali")

        if not sources:
            perf.log_summary("No sources provided")
            return []

        # Convert sources to ChunkSource objects if needed
        perf.start_phase("source_conversion")
        chunk_sources = []
        for source in sources:
            if isinstance(source, dict):
                chunk_sources.append(ChunkSource(**source))
            else:
                chunk_sources.append(source)

        # Create system filters for folder and user scoping
        perf.start_phase("filter_creation")
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id
        # Note: Don't add auth.app_id here - it's already handled in document retrieval

        # Main batch retrieval operation
        perf.start_phase("batch_retrieve_chunks")
        results = await document_service.batch_retrieve_chunks(
            chunk_sources, auth, folder_name, end_user_id, use_colpali
        )

        # Log consolidated performance summary
        perf.log_summary(f"Retrieved {len(results)}/{len(sources)} chunks")

        return results
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/query", response_model=CompletionResponse)
async def query_completion(
    request: CompletionQueryRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """
    Generate completion using relevant chunks as context.

    When graph_name is provided, the query will leverage the knowledge graph
    to enhance retrieval by finding relevant entities and their connected documents.

    Args:
        request: CompletionQueryRequest containing:
            - query: Query text
            - filters: Optional metadata filters
            - k: Number of chunks to use as context (default: 4)
            - min_score: Minimum similarity threshold (default: 0.0)
            - max_tokens: Maximum tokens in completion
            - temperature: Model temperature
            - use_reranking: Whether to use reranking
            - use_colpali: Whether to use ColPali-style embedding model
            - graph_name: Optional name of the graph to use for knowledge graph-enhanced retrieval
            - hop_depth: Number of relationship hops to traverse in the graph (1-3)
            - include_paths: Whether to include relationship paths in the response
            - prompt_overrides: Optional customizations for entity extraction, resolution, and query prompts
            - folder_name: Optional folder to scope the operation to
            - end_user_id: Optional end-user ID to scope the operation to
            - schema: Optional schema for structured output
            - chat_id: Optional chat conversation identifier for maintaining history
        auth: Authentication context

    Returns:
        CompletionResponse: Generated text completion or structured output
    """
    # Initialize performance tracker
    perf = PerformanceTracker(f"Query: '{request.query[:50]}...'")

    # Prepare telemetry metadata
    meta = telemetry.query_metadata(None, request=request)  # type: ignore[arg-type]
    token_est = len(request.query.split()) if isinstance(request.query, str) else 0

    try:
        # Validate prompt overrides before proceeding
        perf.start_phase("prompt_validation")
        if request.prompt_overrides:
            validate_prompt_overrides_with_http_exception(request.prompt_overrides, operation_type="query")

        # Chat history retrieval
        perf.start_phase("chat_history_retrieval")
        history_key = None
        history: List[Dict[str, Any]] = []
        if request.chat_id:
            history_key = f"chat:{request.chat_id}"
            stored = await redis.get(history_key)
            if stored:
                try:
                    history = json.loads(stored)
                except Exception:
                    history = []
            else:
                db_hist = await document_service.db.get_chat_history(request.chat_id, auth.user_id, auth.app_id)
                if db_hist:
                    history = db_hist

            history.append(
                {
                    "role": "user",
                    "content": request.query,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

        # Check query limits if in cloud mode
        perf.start_phase("limits_check")
        if settings.MODE == "cloud" and auth.user_id:
            # Check limits before proceeding
            await check_and_increment_limits(auth, "query", 1)

        # Main query processing
        perf.start_phase("document_service_query")
        result = await document_service.query(
            request.query,
            auth,
            request.filters,
            request.k,
            request.min_score,
            request.max_tokens,
            request.temperature,
            request.use_reranking,
            request.use_colpali,
            request.graph_name,
            request.hop_depth,
            request.include_paths,
            request.prompt_overrides,
            request.folder_name,
            request.end_user_id,
            request.schema,
            history,
            perf,  # Pass performance tracker
            request.stream_response,
            request.llm_config,
            request.padding,  # Pass padding parameter
        )

        # Handle streaming vs non-streaming responses
        if request.stream_response:
            # For streaming responses, unpack the tuple
            response_stream, sources = result

            async def generate_stream():
                full_content = ""
                first_token_time = None

                async for chunk in response_stream:
                    # Track time to first token
                    if first_token_time is None:
                        first_token_time = time.time()
                        completion_start_to_first_token = first_token_time - perf.start_time
                        perf.add_suboperation("completion_start_to_first_token", completion_start_to_first_token)
                        logger.info(f"Completion start to first token: {completion_start_to_first_token:.2f}s")

                    full_content += chunk
                    yield f"data: {json.dumps({'type': 'assistant', 'content': chunk})}\n\n"

                # Convert sources to the format expected by frontend
                sources_info = [
                    {"document_id": source.document_id, "chunk_number": source.chunk_number, "score": source.score}
                    for source in sources
                ]

                # Send completion signal with sources
                yield f"data: {json.dumps({'type': 'done', 'sources': sources_info})}\n\n"

                # Handle chat history after streaming is complete
                if history_key:
                    history.append(
                        {
                            "role": "assistant",
                            "content": full_content,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                    await redis.set(history_key, json.dumps(history))
                    await document_service.db.upsert_chat_history(
                        request.chat_id,
                        auth.user_id,
                        auth.app_id,
                        history,
                    )

                # Log consolidated performance summary for streaming
                streaming_time = time.time() - first_token_time if first_token_time else 0
                perf.add_suboperation("streaming_duration", streaming_time)
                perf.log_summary(f"Generated streaming completion with {len(sources)} sources")

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }

            # Wrap original generator with telemetry span so formatting and history logic are preserved
            async def wrapped():
                async with telemetry.track_operation(
                    operation_type="query",
                    user_id=auth.entity_id,
                    app_id=auth.app_id,
                    tokens_used=token_est,
                    metadata=meta,
                ):
                    async for item in generate_stream():
                        yield item

            return StreamingResponse(wrapped(), media_type="text/event-stream", headers=headers)
        else:
            # For non-streaming responses, we record telemetry around result construction
            async with telemetry.track_operation(
                operation_type="query",
                user_id=auth.entity_id,
                app_id=auth.app_id,
                tokens_used=token_est,
                metadata=meta,
            ):
                response = result

            # Chat history storage for non-streaming responses
            perf.start_phase("chat_history_storage")
            if history_key:
                history.append(
                    {
                        "role": "assistant",
                        "content": response.completion,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
                await redis.set(history_key, json.dumps(history))
                await document_service.db.upsert_chat_history(
                    request.chat_id,
                    auth.user_id,
                    auth.app_id,
                    history,
                )

            # Log consolidated performance summary
            perf.log_summary(f"Generated completion with {len(response.sources) if response.sources else 0} sources")

            return response
    except ValueError as e:
        validate_prompt_overrides_with_http_exception(operation_type="query", error=e)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get("/chat/{chat_id}", response_model=List[ChatMessage])
async def get_chat_history(
    chat_id: str,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """Retrieve the message history for a chat conversation.

    Args:
        chat_id: Identifier of the conversation whose history should be loaded.
        auth: Authentication context used to verify access to the conversation.
        redis: Redis connection where chat messages are stored.

    Returns:
        A list of :class:`ChatMessage` objects or an empty list if no history
        exists.
    """
    history_key = f"chat:{chat_id}"
    stored = await redis.get(history_key)
    if not stored:
        db_hist = await document_service.db.get_chat_history(chat_id, auth.user_id, auth.app_id)
        if not db_hist:
            return []
        return [ChatMessage(**m) for m in db_hist]
    try:
        data = json.loads(stored)
        return [ChatMessage(**m) for m in data]
    except Exception:
        return []


@app.get("/models/available")
async def get_available_models_for_selection(auth: AuthContext = Depends(verify_token)):
    """Get list of available models for UI selection.

    Returns a list of models that can be used for queries. Each model includes:
    - id: Model identifier to use in llm_config
    - name: Display name for the model
    - provider: The LLM provider (e.g., openai, anthropic, ollama)
    - description: Optional description of the model
    """
    # For now, return some common models that work with LiteLLM
    # In the future, this could be configurable or dynamically determined
    models = [
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "provider": "openai",
            "description": "OpenAI's most capable model with vision support",
        },
        {
            "id": "gpt-4o-mini",
            "name": "GPT-4o Mini",
            "provider": "openai",
            "description": "Faster, more affordable GPT-4o variant",
        },
        {
            "id": "claude-3-5-sonnet-20241022",
            "name": "Claude 3.5 Sonnet",
            "provider": "anthropic",
            "description": "Anthropic's most intelligent model",
        },
        {
            "id": "claude-3-5-haiku-20241022",
            "name": "Claude 3.5 Haiku",
            "provider": "anthropic",
            "description": "Fast and affordable Claude model",
        },
        {
            "id": "gemini/gemini-1.5-pro",
            "name": "Gemini 1.5 Pro",
            "provider": "google",
            "description": "Google's advanced model with long context",
        },
        {
            "id": "gemini/gemini-1.5-flash",
            "name": "Gemini 1.5 Flash",
            "provider": "google",
            "description": "Fast and efficient Gemini model",
        },
        {
            "id": "deepseek/deepseek-chat",
            "name": "DeepSeek Chat",
            "provider": "deepseek",
            "description": "DeepSeek's conversational AI model",
        },
        {
            "id": "groq/llama-3.3-70b-versatile",
            "name": "Llama 3.3 70B",
            "provider": "groq",
            "description": "Fast inference with Groq",
        },
        {
            "id": "groq/llama-3.1-8b-instant",
            "name": "Llama 3.1 8B",
            "provider": "groq",
            "description": "Ultra-fast small model on Groq",
        },
    ]

    # Check if there's a configured model in settings to add to the list
    if hasattr(settings, "COMPLETION_MODEL") and hasattr(settings, "REGISTERED_MODELS"):
        configured_model = settings.COMPLETION_MODEL
        if configured_model in settings.REGISTERED_MODELS:
            config = settings.REGISTERED_MODELS[configured_model]
            model_name = config.get("model_name", configured_model)
            # Add the configured model if it's not already in the list
            if not any(m["id"] == model_name for m in models):
                models.insert(
                    0,
                    {
                        "id": model_name,
                        "name": f"{configured_model} (Configured)",
                        "provider": "configured",
                        "description": "Currently configured model in morphik.toml",
                    },
                )

    return {"models": models}


@app.post("/agent", response_model=Dict[str, Any])
@telemetry.track(operation_type="agent_query")
async def agent_query(
    request: AgentQueryRequest,
    auth: AuthContext = Depends(verify_token),
    redis: arq.ArqRedis = Depends(get_redis_pool),
):
    """Execute an agent-style query using the :class:`MorphikAgent`.

    Args:
        request: The query payload containing the natural language question and optional chat_id.
        auth: Authentication context used to enforce limits and access control.
        redis: Redis connection for chat history storage.

    Returns:
        A dictionary with the agent's full response.
    """
    # Chat history retrieval
    history_key = None
    history: List[Dict[str, Any]] = []
    if request.chat_id:
        history_key = f"chat:{request.chat_id}"
        stored = await redis.get(history_key)
        if stored:
            try:
                history = json.loads(stored)
            except Exception:
                history = []
        else:
            db_hist = await document_service.db.get_chat_history(request.chat_id, auth.user_id, auth.app_id)
            if db_hist:
                history = db_hist

        history.append(
            {
                "role": "user",
                "content": request.query,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    # Check free-tier agent call limits in cloud mode
    if settings.MODE == "cloud" and auth.user_id:
        await check_and_increment_limits(auth, "agent", 1)

    # Use the shared MorphikAgent instance; per-run state is now isolated internally
    response = await morphik_agent.run(request.query, auth, history, request.display_mode)

    # Chat history storage
    if history_key:
        # Store the full agent response with structured data
        agent_message = {
            "role": "assistant",
            "content": response.get("response", ""),
            "timestamp": datetime.now(UTC).isoformat(),
            # Store agent-specific structured data
            "agent_data": {
                "display_objects": response.get("display_objects", []),
                "tool_history": response.get("tool_history", []),
                "sources": response.get("sources", []),
            },
        }
        history.append(agent_message)
        await redis.set(history_key, json.dumps(history))
        await document_service.db.upsert_chat_history(
            request.chat_id,
            auth.user_id,
            auth.app_id,
            history,
        )

    # Return the complete response dictionary
    return response


@app.post("/documents", response_model=List[Document])
async def list_documents(
    request: ListDocumentsRequest,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = Query(None),
):
    """
    List accessible documents.

    Args:
        request: Request body containing filters and pagination
        auth: Authentication context
        folder_name: Optional folder to scope the operation to
        end_user_id: Optional end-user ID to scope the operation to

    Returns:
        List[Document]: List of accessible documents
    """
    # Create system filters for folder and user scoping
    system_filters = {}

    # Normalize folder_name parameter (convert string "null" to None)
    if folder_name is not None:
        normalized_folder_name = normalize_folder_name(folder_name)
        system_filters["folder_name"] = normalized_folder_name
    if end_user_id:
        system_filters["end_user_id"] = end_user_id
    # Note: auth.app_id is already handled in _build_access_filter_optimized

    return await document_service.db.get_documents(
        auth, request.skip, request.limit, filters=request.document_filters, system_filters=system_filters
    )


@app.get("/documents/{document_id}", response_model=Document)
async def get_document(document_id: str, auth: AuthContext = Depends(verify_token)):
    """Retrieve a single document by its external identifier.

    Args:
        document_id: External ID of the document to fetch.
        auth: Authentication context used to verify access rights.

    Returns:
        The :class:`Document` metadata if found.
    """
    try:
        doc = await document_service.db.get_document(document_id, auth)
        logger.debug(f"Found document: {doc}")
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException as e:
        logger.error(f"Error getting document: {e}")
        raise e


@app.get("/documents/{document_id}/status", response_model=Dict[str, Any])
async def get_document_status(document_id: str, auth: AuthContext = Depends(verify_token)):
    """
    Get the processing status of a document.

    Args:
        document_id: ID of the document to check
        auth: Authentication context

    Returns:
        Dict containing status information for the document
    """
    try:
        doc = await document_service.db.get_document(document_id, auth)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Extract status information
        status = doc.system_metadata.get("status", "unknown")

        response = {
            "document_id": doc.external_id,
            "status": status,
            "filename": doc.filename,
            "created_at": doc.system_metadata.get("created_at"),
            "updated_at": doc.system_metadata.get("updated_at"),
        }

        # Add error information if failed
        if status == "failed":
            response["error"] = doc.system_metadata.get("error", "Unknown error")

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting document status: {str(e)}")


@app.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
@telemetry.track(operation_type="delete_document", metadata_resolver=telemetry.document_delete_metadata)
async def delete_document(document_id: str, auth: AuthContext = Depends(verify_token)):
    """
    Delete a document and all associated data.

    This endpoint deletes a document and all its associated data, including:
    - Document metadata
    - Document content in storage
    - Document chunks and embeddings in vector store

    Args:
        document_id: ID of the document to delete
        auth: Authentication context (must have write access to the document)

    Returns:
        Deletion status
    """
    try:
        success = await document_service.delete_document(document_id, auth)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or delete failed")
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.get("/documents/filename/{filename}", response_model=Document)
async def get_document_by_filename(
    filename: str,
    auth: AuthContext = Depends(verify_token),
    folder_name: Optional[Union[str, List[str]]] = Query(None),
    end_user_id: Optional[str] = None,
):
    """
    Get document by filename.

    Args:
        filename: Filename of the document to retrieve
        auth: Authentication context
        folder_name: Optional folder to scope the operation to
        end_user_id: Optional end-user ID to scope the operation to

    Returns:
        Document: Document metadata if found and accessible
    """
    try:
        # Create system filters for folder and user scoping
        system_filters = {}
        if folder_name is not None:
            normalized_folder_name = normalize_folder_name(folder_name)
            system_filters["folder_name"] = normalized_folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        doc = await document_service.db.get_document_by_filename(filename, auth, system_filters)
        logger.debug(f"Found document by filename: {doc}")
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document with filename '{filename}' not found")
        return doc
    except HTTPException as e:
        logger.error(f"Error getting document by filename: {e}")
        raise e


@app.get("/documents/{document_id}/download_url", response_model=DocumentDownloadUrlResponse)
async def get_document_download_url(
    document_id: str,
    auth: AuthContext = Depends(verify_token),
    expires_in: int = Query(3600, description="URL expiration time in seconds"),
):
    """
    Get a download URL for a specific document.

    Args:
        document_id: External ID of the document
        auth: Authentication context
        expires_in: URL expiration time in seconds (default: 1 hour)

    Returns:
        Dictionary containing the download URL and metadata
    """
    try:
        # Get the document
        doc = await document_service.db.get_document(document_id, auth)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        # Check if document has storage info
        if not doc.storage_info or not doc.storage_info.get("bucket") or not doc.storage_info.get("key"):
            raise HTTPException(status_code=404, detail="Document file not found in storage")

        # Generate download URL
        download_url = await document_service.storage.get_download_url(
            doc.storage_info["bucket"], doc.storage_info["key"], expires_in=expires_in
        )

        return {
            "document_id": doc.external_id,
            "filename": doc.filename,
            "content_type": doc.content_type,
            "download_url": download_url,
            "expires_in": expires_in,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting download URL for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting download URL: {str(e)}")


@app.get("/documents/{document_id}/file", response_model=None)
async def download_document_file(document_id: str, auth: AuthContext = Depends(verify_token)):
    """
    Download the actual file content for a document.
    This endpoint is used for local storage when file:// URLs cannot be accessed by browsers.

    Args:
        document_id: External ID of the document
        auth: Authentication context

    Returns:
        StreamingResponse with the file content
    """
    try:
        logger.info(f"Attempting to download file for document ID: {document_id}")
        logger.info(f"Auth context: entity_id={auth.entity_id}, app_id={auth.app_id}")

        # Get the document
        doc = await document_service.db.get_document(document_id, auth)
        logger.info(f"Document lookup result: {doc is not None}")

        if not doc:
            logger.warning(f"Document not found in database: {document_id}")
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

        logger.info(f"Found document: {doc.filename}, content_type: {doc.content_type}")
        logger.info(f"Storage info: {doc.storage_info}")

        # Check if document has storage info
        if not doc.storage_info or not doc.storage_info.get("bucket") or not doc.storage_info.get("key"):
            logger.warning(f"Document has no storage info: {document_id}")
            raise HTTPException(status_code=404, detail="Document file not found in storage")

        # Download file content from storage
        logger.info(f"Downloading from bucket: {doc.storage_info['bucket']}, key: {doc.storage_info['key']}")
        file_content = await document_service.storage.download_file(doc.storage_info["bucket"], doc.storage_info["key"])

        logger.info(f"Successfully downloaded {len(file_content)} bytes")

        # Create streaming response

        from fastapi.responses import StreamingResponse

        def generate():
            yield file_content

        return StreamingResponse(
            generate(),
            media_type=doc.content_type or "application/octet-stream",
            headers={
                "Content-Disposition": f"inline; filename=\"{doc.filename or 'document'}\"",
                "Content-Length": str(len(file_content)),
            },
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found in storage for document {document_id}: {e}")
        raise HTTPException(status_code=404, detail=f"File not found in storage: {str(e)}")
    except Exception as e:
        logger.error(f"Error downloading document file {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@app.post("/documents/{document_id}/update_text", response_model=Document)
@telemetry.track(operation_type="update_document_text", metadata_resolver=telemetry.document_update_text_metadata)
async def update_document_text(
    document_id: str,
    request: IngestTextRequest,
    update_strategy: str = "add",
    auth: AuthContext = Depends(verify_token),
):
    """
    Update a document with new text content using the specified strategy.

    Args:
        document_id: ID of the document to update
        request: Text content and metadata for the update
        update_strategy: Strategy for updating the document (default: 'add')
        auth: Authentication context

    Returns:
        Document: Updated document metadata
    """
    try:
        doc = await document_service.update_document(
            document_id=document_id,
            auth=auth,
            content=request.content,
            file=None,
            filename=request.filename,
            metadata=request.metadata,
            rules=request.rules,
            update_strategy=update_strategy,
            use_colpali=request.use_colpali,
        )

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found or update failed")

        return doc
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/documents/{document_id}/update_file", response_model=Document)
@telemetry.track(operation_type="update_document_file", metadata_resolver=telemetry.document_update_file_metadata)
async def update_document_file(
    document_id: str,
    file: UploadFile,
    metadata: str = Form("{}"),
    rules: str = Form("[]"),
    update_strategy: str = Form("add"),
    use_colpali: Optional[bool] = Form(None),
    auth: AuthContext = Depends(verify_token),
):
    """
    Update a document with content from a file using the specified strategy.

    Args:
        document_id: ID of the document to update
        file: File to add to the document
        metadata: JSON string of metadata to merge with existing metadata
        rules: JSON string of rules to apply to the content
        update_strategy: Strategy for updating the document (default: 'add')
        use_colpali: Whether to use multi-vector embedding
        auth: Authentication context

    Returns:
        Document: Updated document metadata
    """
    try:
        metadata_dict = json.loads(metadata)
        rules_list = json.loads(rules)

        doc = await document_service.update_document(
            document_id=document_id,
            auth=auth,
            content=None,
            file=file,
            filename=file.filename,
            metadata=metadata_dict,
            rules=rules_list,
            update_strategy=update_strategy,
            use_colpali=use_colpali,
        )

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found or update failed")

        return doc
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/documents/{document_id}/update_metadata", response_model=Document)
@telemetry.track(
    operation_type="update_document_metadata",
    metadata_resolver=telemetry.document_update_metadata_resolver,
)
async def update_document_metadata(
    document_id: str, metadata_updates: Dict[str, Any], auth: AuthContext = Depends(verify_token)
):
    """
    Update only a document's metadata.

    Args:
        document_id: ID of the document to update
        metadata_updates: New metadata to merge with existing metadata
        auth: Authentication context

    Returns:
        Document: Updated document metadata
    """
    try:
        doc = await document_service.update_document(
            document_id=document_id,
            auth=auth,
            content=None,
            file=None,
            filename=None,
            metadata=metadata_updates,
            rules=[],
            update_strategy="add",
            use_colpali=None,
        )

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found or update failed")

        return doc
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


# Usage tracking endpoints
@app.get("/usage/stats")
@telemetry.track(operation_type="get_usage_stats", metadata_resolver=telemetry.usage_stats_metadata)
async def get_usage_stats(auth: AuthContext = Depends(verify_token)) -> Dict[str, int]:
    """Get usage statistics for the authenticated user.

    Args:
        auth: Authentication context identifying the caller.

    Returns:
        A mapping of operation types to token usage counts.
    """
    if not auth.permissions or "admin" not in auth.permissions:
        return telemetry.get_user_usage(auth.entity_id)
    return telemetry.get_user_usage(auth.entity_id)


@app.get("/usage/recent")
@telemetry.track(operation_type="get_recent_usage", metadata_resolver=telemetry.recent_usage_metadata)
async def get_recent_usage(
    auth: AuthContext = Depends(verify_token),
    operation_type: Optional[str] = None,
    since: Optional[datetime] = None,
    status: Optional[str] = None,
) -> List[Dict]:
    """Retrieve recent telemetry records for the user or application.

    Args:
        auth: Authentication context; admin users receive global records.
        operation_type: Optional operation type to filter by.
        since: Only return records newer than this timestamp.
        status: Optional status filter (e.g. ``success`` or ``error``).

    Returns:
        A list of usage entries sorted by timestamp, each represented as a
        dictionary.
    """
    if not auth.permissions or "admin" not in auth.permissions:
        records = telemetry.get_recent_usage(
            user_id=auth.entity_id, operation_type=operation_type, since=since, status=status
        )
    else:
        records = telemetry.get_recent_usage(operation_type=operation_type, since=since, status=status)

    return [
        {
            "timestamp": record.timestamp,
            "operation_type": record.operation_type,
            "tokens_used": record.tokens_used,
            "user_id": record.user_id,
            "duration_ms": record.duration_ms,
            "status": record.status,
            "metadata": record.metadata,
        }
        for record in records
    ]


@app.post("/folders", response_model=Folder)
@telemetry.track(operation_type="create_folder", metadata_resolver=telemetry.create_folder_metadata)
async def create_folder(
    folder_create: FolderCreate,
    auth: AuthContext = Depends(verify_token),
) -> Folder:
    """
    Create a new folder.

    Args:
        folder_create: Folder creation request containing name and optional description
        auth: Authentication context

    Returns:
        Folder: Created folder
    """
    try:
        # Create a folder object with explicit ID
        import uuid

        folder_id = str(uuid.uuid4())
        logger.info(f"Creating folder with ID: {folder_id}, auth.user_id: {auth.user_id}")

        # Set up access control with user_id
        folder = Folder(
            id=folder_id, name=folder_create.name, description=folder_create.description, app_id=auth.app_id
        )

        # Store in database
        success = await document_service.db.create_folder(folder, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to create folder")

        return folder
    except Exception as e:
        logger.error(f"Error creating folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/folders", response_model=List[Folder])
@telemetry.track(operation_type="list_folders", metadata_resolver=telemetry.list_folders_metadata)
async def list_folders(
    auth: AuthContext = Depends(verify_token),
) -> List[Folder]:
    """
    List all folders the user has access to.

    Args:
        auth: Authentication context

    Returns:
        List[Folder]: List of folders
    """
    try:
        folders = await document_service.db.list_folders(auth)
        return folders
    except Exception as e:
        logger.error(f"Error listing folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/folders/summary", response_model=List[FolderSummary])
@telemetry.track(operation_type="list_folders_summary")
async def list_folder_summaries(auth: AuthContext = Depends(verify_token)) -> List[FolderSummary]:
    """Return compact folder list (id, name, doc_count, updated_at)."""

    try:
        summaries = await document_service.db.list_folders_summary(auth)
        return summaries  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.error("Error listing folder summaries: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/folders/{folder_id}", response_model=Folder)
@telemetry.track(operation_type="get_folder", metadata_resolver=telemetry.get_folder_metadata)
async def get_folder(
    folder_id: str,
    auth: AuthContext = Depends(verify_token),
) -> Folder:
    """
    Get a folder by ID.

    Args:
        folder_id: ID of the folder
        auth: Authentication context

    Returns:
        Folder: Folder if found and accessible
    """
    try:
        folder = await document_service.db.get_folder(folder_id, auth)

        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

        return folder
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/folders/{folder_name}", response_model=FolderDeleteResponse)
@telemetry.track(operation_type="delete_folder", metadata_resolver=telemetry.delete_folder_metadata)
async def delete_folder(
    folder_name: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Delete a folder and all associated documents.

    Args:
        folder_name: Name of the folder to delete
        auth: Authentication context (must have write access to the folder)

    Returns:
        Deletion status
    """
    try:
        folder = await document_service.db.get_folder_by_name(folder_name, auth)
        folder_id = folder.id
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")

        document_ids = folder.document_ids
        tasks = [remove_document_from_folder(folder_id, document_id, auth) for document_id in document_ids]
        results = await asyncio.gather(*tasks)
        stati = [res.get("status", False) for res in results]
        if not all(stati):
            failed = [doc for doc, stat in zip(document_ids, stati) if not stat]
            msg = "Failed to remove the following documents from folder: " + ", ".join(failed)
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        # folder is empty now
        delete_tasks = [document_service.db.delete_document(document_id, auth) for document_id in document_ids]
        stati = await asyncio.gather(*delete_tasks)
        if not all(stati):
            failed = [doc for doc, stat in zip(document_ids, stati) if not stat]
            msg = "Failed to delete the following documents: " + ", ".join(failed)
            logger.error(msg)
            raise HTTPException(status_code=500, detail=msg)

        db: PostgresDatabase = document_service.db
        # just remove the folder too now.
        status = await db.delete_folder(folder_id, auth)
        if not status:
            logger.error(f"Failed to delete folder {folder_id}")
            raise HTTPException(status_code=500, detail=f"Failed to delete folder {folder_id}")
        return {"status": "success", "message": f"Folder {folder_id} deleted successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/folders/{folder_id}/documents/{document_id}", response_model=DocumentAddToFolderResponse)
@telemetry.track(operation_type="add_document_to_folder", metadata_resolver=telemetry.add_document_to_folder_metadata)
async def add_document_to_folder(
    folder_id: str,
    document_id: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Add a document to a folder.

    Args:
        folder_id: ID of the folder
        document_id: ID of the document
        auth: Authentication context

    Returns:
        Success status
    """
    try:
        success = await document_service.db.add_document_to_folder(folder_id, document_id, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add document to folder")

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error adding document to folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/folders/{folder_id}/documents/{document_id}", response_model=DocumentDeleteResponse)
@telemetry.track(
    operation_type="remove_document_from_folder", metadata_resolver=telemetry.remove_document_from_folder_metadata
)
async def remove_document_from_folder(
    folder_id: str,
    document_id: str,
    auth: AuthContext = Depends(verify_token),
):
    """
    Remove a document from a folder.

    Args:
        folder_id: ID of the folder
        document_id: ID of the document
        auth: Authentication context

    Returns:
        Success status
    """
    try:
        success = await document_service.db.remove_document_from_folder(folder_id, document_id, auth)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove document from folder")

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error removing document from folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/local/generate_uri", include_in_schema=True)
async def generate_local_uri(
    name: str = Form("admin"),
    expiry_days: int = Form(30),
) -> Dict[str, str]:
    """Generate a development URI for running Morphik locally.

    Args:
        name: Developer name to embed in the token payload.
        expiry_days: Number of days the generated token should remain valid.

    Returns:
        A dictionary containing the ``uri`` that can be used to connect to the
        local instance.
    """
    try:
        # Clean name
        name = name.replace(" ", "_").lower()

        # Create payload
        payload = {
            "type": "developer",
            "entity_id": name,
            "permissions": ["read", "write", "admin"],
            "exp": datetime.now(UTC) + timedelta(days=expiry_days),
        }

        # Generate token
        token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

        # Read config for host/port
        with open("morphik.toml", "rb") as f:
            config = tomli.load(f)
        base_url = f"{config['api']['host']}:{config['api']['port']}".replace("localhost", "127.0.0.1")

        # Generate URI
        uri = f"morphik://{name}:{token}@{base_url}"
        return {"uri": uri}
    except Exception as e:
        logger.error(f"Error generating local URI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cloud/generate_uri", include_in_schema=True)
async def generate_cloud_uri(
    request: GenerateUriRequest,
    authorization: str = Header(None),
) -> Dict[str, str]:
    """Generate an authenticated URI for a cloud-hosted Morphik application.

    Args:
        request: Parameters for URI generation including ``app_id`` and ``name``.
        authorization: Bearer token of the user requesting the URI.

    Returns:
        A dictionary with the generated ``uri`` and associated ``app_id``.
    """
    try:
        app_id = request.app_id
        name = request.name
        user_id = request.user_id
        expiry_days = request.expiry_days

        logger.debug(f"Generating cloud URI for app_id={app_id}, name={name}, user_id={user_id}")

        # Verify authorization header before proceeding
        if not authorization:
            logger.warning("Missing authorization header")
            raise HTTPException(
                status_code=401,
                detail="Missing authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify the token is valid
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")

        token = authorization[7:]  # Remove "Bearer "

        try:
            # Decode the token to ensure it's valid
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])

            # Only allow users to create apps for themselves (or admin)
            token_user_id = payload.get("user_id")
            logger.debug(f"Token user ID: {token_user_id}")
            logger.debug(f"User ID: {user_id}")
            if not (token_user_id == user_id or "admin" in payload.get("permissions", [])):
                raise HTTPException(
                    status_code=403,
                    detail="You can only create apps for your own account unless you have admin permissions",
                )
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=str(e))

        # Import UserService here to avoid circular imports
        from core.services.user_service import UserService

        user_service = UserService()

        # Initialize user service if needed
        await user_service.initialize()

        # Clean name
        name = name.replace(" ", "_").lower()

        # Check if the user is within app limit and generate URI
        uri = await user_service.generate_cloud_uri(user_id, app_id, name, expiry_days)

        if not uri:
            logger.debug("Application limit reached for this account tier with user_id: %s", user_id)
            raise HTTPException(status_code=403, detail="Application limit reached for this account tier")

        return {"uri": uri, "app_id": app_id}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error generating cloud URI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/folders/{folder_id}/set_rule", response_model=FolderRuleResponse)
@telemetry.track(operation_type="set_folder_rule", metadata_resolver=telemetry.set_folder_rule_metadata)
async def set_folder_rule(
    folder_id: str,
    request: SetFolderRuleRequest,
    auth: AuthContext = Depends(verify_token),
    apply_to_existing: bool = True,
):
    """
    Set extraction rules for a folder.

    Args:
        folder_id: ID of the folder to set rules for
        request: SetFolderRuleRequest containing metadata extraction rules
        auth: Authentication context
        apply_to_existing: Whether to apply rules to existing documents in the folder

    Returns:
        Success status with processing results
    """
    # Import text here to ensure it's available in this function's scope

    try:
        # Log detailed information about the rules
        logger.debug(f"Setting rules for folder {folder_id}")
        logger.debug(f"Number of rules: {len(request.rules)}")

        for i, rule in enumerate(request.rules):
            logger.debug(f"\nRule {i + 1}:")
            logger.debug(f"Type: {rule.type}")
            logger.debug("Schema:")
            for field_name, field_config in rule.schema.items():
                logger.debug(f"  Field: {field_name}")
                logger.debug(f"    Type: {field_config.get('type', 'unknown')}")
                logger.debug(f"    Description: {field_config.get('description', 'No description')}")
                if "schema" in field_config:
                    logger.debug("    Has JSON schema: Yes")
                    logger.debug(f"    Schema: {field_config['schema']}")

        # Get the folder
        folder = await document_service.db.get_folder(folder_id, auth)
        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

        # Note: Write access will be checked by the database operations

        # Update folder with rules
        # Convert rules to dicts for JSON serialization
        rules_dicts = [rule.model_dump() for rule in request.rules]

        # Update the folder in the database
        async with document_service.db.async_session() as session:
            # Execute update query
            await session.execute(
                text(
                    """
                    UPDATE folders
                    SET rules = :rules
                    WHERE id = :folder_id
                    """
                ),
                {"folder_id": folder_id, "rules": json.dumps(rules_dicts)},
            )
            await session.commit()

        logger.info(f"Successfully updated folder {folder_id} with {len(request.rules)} rules")

        # Get updated folder
        updated_folder = await document_service.db.get_folder(folder_id, auth)

        # If apply_to_existing is True, apply these rules to all existing documents in the folder
        processing_results = {"processed": 0, "errors": []}

        if apply_to_existing and folder.document_ids:
            logger.info(f"Applying rules to {len(folder.document_ids)} existing documents in folder")

            # Import rules processor

            # Get all documents in the folder
            documents = await document_service.db.get_documents_by_id(folder.document_ids, auth)

            # Process each document
            for doc in documents:
                try:
                    # Get document content
                    logger.info(f"Processing document {doc.external_id}")

                    # For each document, apply the rules from the folder
                    doc_content = None

                    # Get content from system_metadata if available
                    if doc.system_metadata and "content" in doc.system_metadata:
                        doc_content = doc.system_metadata["content"]
                        logger.info(f"Retrieved content from system_metadata for document {doc.external_id}")

                    # If we still have no content, log error and continue
                    if not doc_content:
                        error_msg = f"No content found in system_metadata for document {doc.external_id}"
                        logger.error(error_msg)
                        processing_results["errors"].append({"document_id": doc.external_id, "error": error_msg})
                        continue

                    # Process document with rules
                    try:
                        # Convert request rules to actual rule models and apply them
                        from core.models.rules import MetadataExtractionRule

                        for rule_request in request.rules:
                            if rule_request.type == "metadata_extraction":
                                # Create the actual rule model
                                rule = MetadataExtractionRule(type=rule_request.type, schema=rule_request.schema)

                                # Apply the rule with retries
                                max_retries = 3
                                base_delay = 1  # seconds
                                extracted_metadata = None
                                last_error = None

                                for retry_count in range(max_retries):
                                    try:
                                        if retry_count > 0:
                                            # Exponential backoff
                                            delay = base_delay * (2 ** (retry_count - 1))
                                            logger.info(f"Retry {retry_count}/{max_retries} after {delay}s delay")
                                            await asyncio.sleep(delay)

                                        extracted_metadata, _ = await rule.apply(doc_content, {})
                                        logger.info(
                                            f"Successfully extracted metadata on attempt {retry_count + 1}: "
                                            f"{extracted_metadata}"
                                        )
                                        break  # Success, exit retry loop

                                    except Exception as rule_apply_error:
                                        last_error = rule_apply_error
                                        logger.warning(
                                            f"Metadata extraction attempt {retry_count + 1} failed: "
                                            f"{rule_apply_error}"
                                        )
                                        if retry_count == max_retries - 1:  # Last attempt
                                            logger.error(f"All {max_retries} metadata extraction attempts failed")
                                            processing_results["errors"].append(
                                                {
                                                    "document_id": doc.external_id,
                                                    "error": f"Failed to extract metadata after {max_retries} "
                                                    f"attempts: {str(last_error)}",
                                                }
                                            )
                                            continue  # Skip to next document

                                # Update document metadata if extraction succeeded
                                if extracted_metadata:
                                    # Merge new metadata with existing
                                    doc.metadata.update(extracted_metadata)

                                    # Create an updates dict that only updates metadata
                                    # We need to create system_metadata with all preserved fields
                                    # Note: In the database, metadata is stored as 'doc_metadata', not 'metadata'
                                    updates = {
                                        "doc_metadata": doc.metadata,  # Use doc_metadata for the database
                                        "system_metadata": {},  # Will be merged with existing in update_document
                                    }

                                    # Explicitly preserve the content field in system_metadata
                                    if "content" in doc.system_metadata:
                                        updates["system_metadata"]["content"] = doc.system_metadata["content"]

                                    # Log the updates we're making
                                    logger.info(
                                        f"Updating document {doc.external_id} with metadata: {extracted_metadata}"
                                    )
                                    logger.info(f"Full metadata being updated: {doc.metadata}")
                                    logger.info(f"Update object being sent to database: {updates}")
                                    logger.info(
                                        f"Preserving content in system_metadata: {'content' in doc.system_metadata}"
                                    )

                                    # Update document in database
                                    app_db = document_service.db
                                    success = await app_db.update_document(doc.external_id, updates, auth)

                                    if success:
                                        logger.info(f"Updated metadata for document {doc.external_id}")
                                        processing_results["processed"] += 1
                                    else:
                                        logger.error(f"Failed to update metadata for document {doc.external_id}")
                                        processing_results["errors"].append(
                                            {
                                                "document_id": doc.external_id,
                                                "error": "Failed to update document metadata",
                                            }
                                        )
                    except Exception as rule_error:
                        logger.error(f"Error processing rules for document {doc.external_id}: {rule_error}")
                        processing_results["errors"].append(
                            {
                                "document_id": doc.external_id,
                                "error": f"Error processing rules: {str(rule_error)}",
                            }
                        )

                except Exception as doc_error:
                    logger.error(f"Error processing document {doc.external_id}: {doc_error}")
                    processing_results["errors"].append({"document_id": doc.external_id, "error": str(doc_error)})

            return {
                "status": "success",
                "message": "Rules set successfully",
                "folder_id": folder_id,
                "rules": updated_folder.rules,
                "processing_results": processing_results,
            }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error setting folder rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Folder-Workflow Association Endpoints
# ---------------------------------------------------------------------------


@app.post("/folders/{folder_id}/workflows/{workflow_id}")
@telemetry.track(operation_type="associate_workflow_to_folder")
async def associate_workflow_to_folder(
    folder_id: str,
    workflow_id: str,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Any]:
    """Associate a workflow with a folder for automatic execution on document ingestion."""
    try:
        # Verify the workflow exists and is accessible
        workflow = await workflow_service.get_workflow(workflow_id, auth)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found or not accessible")

        # Use the database method which handles access control properly
        success = await document_service.db.associate_workflow_to_folder(folder_id, workflow_id, auth)

        if not success:
            # Check if folder exists by trying to get it
            folder = await document_service.db.get_folder(folder_id, auth)
            if not folder:
                raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")
            else:
                raise HTTPException(status_code=403, detail="You don't have write access to this folder")

        return {"success": True, "message": f"Successfully associated workflow {workflow_id} with folder {folder_id}"}
    except Exception as e:
        logger.error(f"Error associating workflow to folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/folders/{folder_id}/workflows/{workflow_id}")
@telemetry.track(operation_type="disassociate_workflow_from_folder")
async def disassociate_workflow_from_folder(
    folder_id: str,
    workflow_id: str,
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Any]:
    """Remove a workflow association from a folder."""
    try:
        # Use the database method which handles access control properly
        success = await document_service.db.disassociate_workflow_from_folder(folder_id, workflow_id, auth)

        if not success:
            # Check if folder exists by trying to get it
            folder = await document_service.db.get_folder(folder_id, auth)
            if not folder:
                raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")
            else:
                raise HTTPException(status_code=403, detail="You don't have write access to this folder")

        return {"success": True, "message": f"Successfully removed workflow {workflow_id} from folder {folder_id}"}
    except Exception as e:
        logger.error(f"Error disassociating workflow from folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/folders/{folder_id}/workflows", response_model=List[Workflow])
@telemetry.track(operation_type="list_folder_workflows")
async def list_folder_workflows(
    folder_id: str,
    auth: AuthContext = Depends(verify_token),
) -> List[Workflow]:
    """List all workflows associated with a folder."""
    try:
        # Get the folder
        folder = await document_service.db.get_folder(folder_id, auth)
        if not folder:
            raise HTTPException(status_code=404, detail=f"Folder {folder_id} not found")

        # Get all workflows
        workflows = []
        for workflow_id in folder.workflow_ids:
            workflow = await workflow_service.get_workflow(workflow_id, auth)
            if workflow:
                workflows.append(workflow)

        return workflows

    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))


# ---------------------------------------------------------------------------
# Cloud – delete application (control-plane only)
# ---------------------------------------------------------------------------


@app.delete("/cloud/apps")
async def delete_cloud_app(
    app_name: str = Query(..., description="Name of the application to delete"),
    auth: AuthContext = Depends(verify_token),
) -> Dict[str, Any]:
    """Delete all resources associated with a given cloud application.

    Args:
        app_name: Name of the application whose data should be removed.
        auth: Authentication context of the requesting user.

    Returns:
        A summary describing how many documents and folders were removed.
    """

    user_id = auth.user_id or auth.entity_id
    logger.info(f"Deleting app {app_name} for user {user_id}")

    from sqlalchemy import delete as sa_delete
    from sqlalchemy import select

    from core.models.apps import AppModel
    from core.services.user_service import UserService

    # 1) Resolve app_id from apps table ----------------------------------
    async with document_service.db.async_session() as session:
        stmt = select(AppModel).where(AppModel.user_id == user_id, AppModel.name == app_name)
        res = await session.execute(stmt)
        app_row = res.scalar_one_or_none()

    if app_row is None:
        raise HTTPException(status_code=404, detail="Application not found")

    app_id = app_row.app_id

    # ------------------------------------------------------------------
    # Create an AuthContext scoped to *this* application so that the
    # underlying access-control filters in the database layer allow us to
    # see and delete resources that belong to the app – even if the JWT
    # used to call this endpoint was scoped to a *different* app.
    # ------------------------------------------------------------------

    if auth.entity_type == EntityType.DEVELOPER:
        app_auth = AuthContext(
            entity_type=auth.entity_type,
            entity_id=auth.entity_id,
            app_id=app_id,
            permissions=auth.permissions or {"read", "write", "admin"},
            user_id=auth.user_id,
        )
    else:
        app_auth = auth

    # 2) Delete all documents for this app ------------------------------
    # ------------------------------------------------------------------
    # Fetch ALL documents for *this* app using the app-scoped auth.
    # ------------------------------------------------------------------
    doc_ids = await document_service.db.find_authorized_and_filtered_documents(app_auth)

    deleted = 0
    for doc_id in doc_ids:
        try:
            await document_service.delete_document(doc_id, app_auth)
            deleted += 1
        except Exception as exc:
            logger.warning("Failed to delete document %s for app %s: %s", doc_id, app_id, exc)

    # 3) Delete folders associated with this app -----------------------
    # ------------------------------------------------------------------
    # Fetch ALL folders for *this* app using the same app-scoped auth.
    # ------------------------------------------------------------------
    folder_ids_deleted = 0
    folders = await document_service.db.list_folders(app_auth)

    for folder in folders:
        try:
            await document_service.db.delete_folder(folder.id, app_auth)
            folder_ids_deleted += 1
        except Exception as f_exc:  # noqa: BLE001
            logger.warning("Failed to delete folder %s for app %s: %s", folder.id, app_id, f_exc)

    # 4) Remove apps table entry ---------------------------------------
    async with document_service.db.async_session() as session:
        await session.execute(sa_delete(AppModel).where(AppModel.app_id == app_id))
        await session.commit()

    # 5) Update user_limits --------------------------------------------
    user_service = UserService()
    await user_service.initialize()
    await user_service.unregister_app(user_id, app_id)

    return {
        "app_name": app_name,
        "status": "deleted",
        "documents_deleted": deleted,
        "folders_deleted": folder_ids_deleted,
    }


@app.get("/chats", response_model=List[Dict[str, Any]])
async def list_chat_conversations(
    auth: AuthContext = Depends(verify_token),
    limit: int = Query(100, ge=1, le=500),
):
    """List chat conversations available to the current user.

    Args:
        auth: Authentication context containing user and app identifiers.
        limit: Maximum number of conversations to return (1-500)

    Returns:
        A list of dictionaries describing each conversation, ordered by most
        recent activity.
    """
    try:
        convos = await document_service.db.list_chat_conversations(
            user_id=auth.user_id,
            app_id=auth.app_id,
            limit=limit,
        )
        return convos
    except Exception as exc:  # noqa: BLE001
        logger.error("Error listing chat conversations: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list chat conversations")


@app.patch("/chats/{chat_id}/title", response_model=ChatTitleResponse)
async def update_chat_title(
    chat_id: str,
    title: str = Query(..., description="New title for the chat"),
    auth: AuthContext = Depends(verify_token),
):
    """Update the title of a chat conversation.

    Args:
        chat_id: ID of the chat conversation to update
        title: New title for the chat
        auth: Authentication context

    Returns:
        Success status
    """
    try:
        success = await document_service.db.update_chat_title(
            conversation_id=chat_id,
            title=title,
            user_id=auth.user_id,
            app_id=auth.app_id,
        )
        if success:
            return {"success": True, "message": "Chat title updated successfully"}
        else:
            raise HTTPException(status_code=404, detail="Chat not found or access denied")
    except Exception as exc:  # noqa: BLE001
        logger.error("Error updating chat title: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update chat title")
