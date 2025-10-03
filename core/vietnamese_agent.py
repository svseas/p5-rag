"""Vietnamese Contract Agent using PydanticAI with English instructions."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from core.agents.vietnamese_query_analyzer import create_vietnamese_query_analyzer
from core.models.auth import AuthContext
from core.services.document_service import DocumentService
from core.tools.document_tools import ToolError

logger = logging.getLogger(__name__)


# Pydantic models for structured extraction (Stage 2)
class EquipmentItem(BaseModel):
    """Equipment/item with consistent core fields + flexible additional fields."""
    name: str = Field(description="Equipment/item name (from table row)")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    quantity: Optional[float] = Field(None, description="Quantity")
    unit_price: Optional[float] = Field(None, description="Unit price in VND")
    total_price: Optional[float] = Field(None, description="Total price in VND")
    additional_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Other fields that vary by contract (origin, specs, etc.)"
    )


class ExtractedData(BaseModel):
    """Structured data extracted from contract chunks."""
    equipment_items: List[EquipmentItem] = Field(
        default_factory=list,
        description="Equipment/items from tables with prices"
    )
    contract_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contract metadata (numbers, dates, parties, total values)"
    )
    relevant_context: str = Field(
        default="",
        description="Any relevant text that helps answer the query"
    )


@dataclass
class MorphikDeps:
    """Dependencies for the Vietnamese contract agent."""
    document_service: DocumentService
    auth: AuthContext
    query_analyzer: Any = None  # VietnameseQueryAnalyzer for intelligent query mapping
    retrieved_chunks: list = None  # Store chunks for source attribution

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.retrieved_chunks is None:
            self.retrieved_chunks = []


# THREE-STAGE PYTHON-ORCHESTRATED ARCHITECTURE (following pdf-qa-system pattern):
# Stage 1: Retrieval - Use PydanticAI agent with retrieve_chunks tool
# Stage 2: Extraction - PydanticAI agent with result_type=ExtractedData (structured output)
# Stage 3: Generation - Direct httpx call to Gemma 3 12B for Vietnamese answer
#
# Key insight from pdf-qa-system: Don't rely on LLM to orchestrate tools.
# Python code calls each step directly in sequence. All stages use Gemma 3 12B via vLLM.

# Custom HTTP client with timeout for vLLM (10 minutes)
vllm_http_client = httpx.AsyncClient(timeout=600.0)

# Stage 1: Retrieval agent (only for chunk retrieval tool)
retrieval_model = OpenAIChatModel(
    model_name='/models/gemma-3-12b-it',
    provider=OpenAIProvider(
        base_url='http://vllm:8080/v1',
        api_key='dummy',
        http_client=vllm_http_client
    ),
)

retrieval_agent = Agent(
    retrieval_model,
    deps_type=MorphikDeps,
    instructions="""You are a document retrieval assistant.

Your only job is to call the retrieve_chunks tool with the user's query.
Use folder_name='folder-contracts' for Vietnamese contract queries."""
)

# Stage 2: Extraction agent using vLLM with Gemma 3 12B IT (structured output!)
extraction_model = OpenAIChatModel(
    model_name='/models/gemma-3-12b-it',
    provider=OpenAIProvider(
        base_url='http://vllm:8080/v1',
        api_key='dummy',  # vLLM doesn't need API key
        http_client=vllm_http_client  # Use shared HTTP client with timeout
    ),
)

extraction_agent = Agent(
    extraction_model,
    output_type=ExtractedData,
    system_prompt="""Extract equipment/items from Vietnamese contract tables into structured format.

CRITICAL RULES:
- Extract ALL equipment/items from tables with their prices
- Preserve exact numbers - DO NOT round or modify
- Look for tables with columns: DANH MỤC HÀNG HÓA, ĐƠN GIÁ, THÀNH TIỀN
- Put name, unit, quantity, unit_price, total_price in the standard fields
- Put other info (specs, origin, etc.) in additional_fields
- Extract contract metadata (numbers, dates) into contract_info

NO explanations - just extract the structured data."""
)


# Stage 3: Generation model configuration (direct API call, not PydanticAI)
# Using vLLM with Gemma 3 12B IT for Vietnamese answer generation
# Single unified model for all stages (retrieval, extraction, generation)
GEMMA3_GENERATION_CONFIG = {
    "model": "/models/gemma-3-12b-it",
    "base_url": "http://vllm:8080/v1",
}



@retrieval_agent.tool
async def retrieve_chunks(
    ctx: RunContext[MorphikDeps],
    query: str,
    k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    min_relevance: float = 0.7,
    folder_name: Optional[str] = None,
    end_user_id: Optional[str] = None,
) -> str:
    """
    Retrieves the most relevant text and image chunks from the knowledge base based on semantic similarity to the query.

    For Vietnamese contract queries (hợp đồng), use folder_name='folder-contracts' to find contract documents.

    The query parameter should be the user's original Vietnamese question.
    The system uses vector search with the ORIGINAL query (semantic similarity handles variations).
    Intent analysis provides instructions to guide how you answer, not what to search for.
    """
    try:
        # Use query analyzer to get intent-based instructions (NOT query rewriting)
        instruction_context = None
        analysis_info = ""

        if ctx.deps.query_analyzer:
            try:
                analysis = await ctx.deps.query_analyzer.analyze(query)
                # Get instruction context to guide the answer (NOT the search query)
                instruction_context = analysis.instruction_context
                # Override parameters from analyzer if not explicitly provided
                if folder_name is None:
                    folder_name = analysis.folder_name
                if k == 5:  # Default value
                    k = analysis.k
                    # Increase k for comparison queries (need more chunks from multiple contracts)
                    if analysis.intent.value == "contract_comparison":
                        k = 15
                if min_relevance == 0.7:  # Default value
                    min_relevance = analysis.min_relevance

                analysis_info = f" (Intent: {analysis.intent.value})"
                logger.info(f"Query analysis: {analysis.reasoning}")
            except Exception as e:
                logger.warning(f"Query analysis failed: {e}")

        # Use ORIGINAL query for vector search (semantic similarity handles variations)
        chunks = await ctx.deps.document_service.retrieve_chunks(
            query=query,  # Original query, not rewritten
            auth=ctx.deps.auth,
            filters=filters,
            k=k,
            min_score=min_relevance,
            use_colpali=True,
            use_reranking=True,  # Enable re-ranking for better relevance
            folder_name=folder_name,
            end_user_id=end_user_id,
        )

        if not chunks:
            return f"Không tìm thấy tài liệu nào phù hợp với truy vấn: {query}{analysis_info}"

        # Store chunks for source attribution
        ctx.deps.retrieved_chunks.extend(chunks)

        # Debug: Log chunk numbers and check for amplifier
        chunk_numbers = []
        has_amplifier = False
        for chunk in chunks:
            chunk_num = getattr(chunk, 'chunk_number', 'unknown')
            chunk_numbers.append(chunk_num)
            if "Bộ khuếch đại" in chunk.content or "УМ-100" in chunk.content or "2.346" in chunk.content:
                has_amplifier = True
                logger.info(f"✅ Found amplifier in chunk {chunk_num}: score={chunk.score:.3f}")

        logger.info(f"Retrieved chunks: {chunk_numbers}")
        if has_amplifier:
            logger.info("✅ Amplifier data IS in retrieved chunks")
        else:
            logger.warning("❌ Amplifier data NOT in retrieved chunks")

        result = f"Tìm thấy {len(chunks)} đoạn văn liên quan{analysis_info}:\n\n"

        for i, chunk in enumerate(chunks, 1):
            result += f"**Đoạn {i}** (Tài liệu: {chunk.filename or 'Không tên'}, Điểm: {chunk.score:.3f}):\n"
            result += f"{chunk.content}\n\n"

        # Add instruction context to guide the agent's answer
        if instruction_context:
            result += f"\n**Instructions for answering:**\n{instruction_context}\n"

        return result

    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return f"Lỗi khi tìm kiếm tài liệu: {str(e)}"


async def extract_relevant_data(query: str, chunks: list) -> str:
    """Stage 2: Extract structured data using PydanticAI agent with result_type.

    This function is called directly by Python code (not as a tool).
    Following pdf-qa-system pattern: Python orchestrates, not LLM.
    Uses Pydantic model to enforce structured output (no reasoning text).

    Args:
        query: User's Vietnamese query
        chunks: Retrieved document chunks

    Returns:
        Formatted string representation of extracted data for Vi-Qwen2-RAG
    """
    if not chunks:
        return "No chunks retrieved to extract from."

    # Format chunks for extraction
    chunks_text = format_chunks_for_viqwen(chunks)

    try:
        # Build user prompt with chunks context
        user_prompt = f"""Query: {query}

Document chunks to extract from:

{chunks_text}

Extract all equipment/items from tables in these chunks, including their prices and specifications."""

        # Use PydanticAI extraction agent with structured output (no manual JSON parsing!)
        # vLLM's guided decoding + hermes tool parser enforces the ExtractedData schema
        result = await extraction_agent.run(user_prompt)
        extracted_data = result.output  # Already validated ExtractedData object

        logger.info(f"Stage 2: Extracted {len(extracted_data.equipment_items)} equipment items")

        # Format ExtractedData into clean text for Vi-Qwen2-RAG
        formatted_parts = []

        # Add equipment table if present
        if extracted_data.equipment_items:
            formatted_parts.append("## Equipment/Items:\n")
            formatted_parts.append("| Name | Unit | Quantity | Unit Price (VND) | Total Price (VND) | Additional Info |")
            formatted_parts.append("|------|------|----------|------------------|-------------------|-----------------|")

            for item in extracted_data.equipment_items:
                additional = ", ".join(f"{k}: {v}" for k, v in item.additional_fields.items()) if item.additional_fields else "-"
                formatted_parts.append(
                    f"| {item.name} | {item.unit or '-'} | {item.quantity or '-'} | "
                    f"{item.unit_price or '-'} | {item.total_price or '-'} | {additional} |"
                )

        # Add contract info if present
        if extracted_data.contract_info:
            formatted_parts.append("\n## Contract Information:")
            for key, value in extracted_data.contract_info.items():
                formatted_parts.append(f"- {key}: {value}")

        # Add relevant context if present
        if extracted_data.relevant_context:
            formatted_parts.append(f"\n## Context:\n{extracted_data.relevant_context}")

        formatted_output = "\n".join(formatted_parts)
        logger.info(f"Stage 2 complete: Formatted {len(formatted_output)} chars")
        return formatted_output

    except Exception as e:
        logger.error(f"Error extracting data with extraction_agent: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error extracting data: {str(e)}"


# Helper functions for three-stage approach

def format_chunks_for_viqwen(chunks: list, query_analyzer: Any = None) -> str:
    """Format retrieved chunks into context for Vi-Qwen2-RAG generation.

    Args:
        chunks: List of retrieved document chunks
        query_analyzer: Optional query analyzer for intent-specific instructions

    Returns:
        Formatted context string for the generation model
    """
    if not chunks:
        return ""

    context_parts = []
    context_parts.append(f"Retrieved {len(chunks)} relevant document chunks:\n")

    for i, chunk in enumerate(chunks, 1):
        doc_name = chunk.filename or "Unknown Document"
        score = f"{chunk.score:.3f}" if hasattr(chunk, 'score') else "N/A"
        context_parts.append(f"\n--- Chunk {i} ---")
        context_parts.append(f"Document: {doc_name} (Relevance: {score})")
        context_parts.append(f"Content:\n{chunk.content}\n")

    return "\n".join(context_parts)


async def call_viqwen2_rag(query: str, context: str, intent_instructions: str = "") -> str:
    """Call Vi-Qwen2-7B-RAG directly via Ollama API for answer generation.

    Args:
        query: User's Vietnamese query
        context: Formatted chunks context
        intent_instructions: Optional intent-specific instructions

    Returns:
        Generated Vietnamese answer
    """
    import httpx

    # Build prompt for Vi-Qwen2-RAG
    system_prompt = """Bạn là trợ lý AI chuyên phân tích hợp đồng tiếng Việt.

NHIỆM VỤ:
- Trả lời câu hỏi dựa HOÀN TOÀN trên nội dung được cung cấp
- Các đoạn văn có thể chứa bảng Markdown với cột: TT, DANH MỤC HÀNG HÓA, QUY CÁCH, ĐVT, SL, ĐƠN GIÁ, THÀNH TIỀN
- Nếu có bảng liên quan: trích xuất CHÍNH XÁC tên mục từ cột "DANH MỤC HÀNG HÓA" và số tiền từ cột "ĐƠN GIÁ" hoặc "THÀNH TIỀN"
- Trả lời bằng tiếng Việt

QUY TẮC QUAN TRỌNG:
✗ NGHIÊM CẤM bịa đặt thông tin không có trong chunks
✗ NGHIÊM CẤM sử dụng kiến thức bên ngoài
✗ NGHIÊM CẤM bỏ qua dữ liệu trong bảng khi có liên quan đến câu hỏi
✓ CHỈ trích xuất dữ liệu từ chunks được cung cấp
✓ Nếu không tìm thấy thông tin: nói rõ "Không tìm thấy thông tin về..."
✓ Trích dẫn tên tài liệu và giá trị cụ thể"""

    if intent_instructions:
        system_prompt += f"\n\nHƯỚNG DẪN CỤ THỂ:\n{intent_instructions}"

    user_prompt = f"""Dựa trên các đoạn văn sau:

{context}

Câu hỏi: {query}

Trả lời bằng tiếng Việt, chỉ dựa trên thông tin trong các đoạn văn trên."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GEMMA3_GENERATION_CONFIG['base_url']}/chat/completions",
                json={
                    "model": GEMMA3_GENERATION_CONFIG['model'],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 4000,  # Increased for detailed answers with table data
                },
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Error calling Vi-Qwen2-RAG: {e}")
        return f"Lỗi khi tạo câu trả lời: {str(e)}"


@retrieval_agent.tool
async def retrieve_document(
    ctx: RunContext[MorphikDeps],
    document_id: str,
    format: str = "text",
    end_user_id: Optional[str] = None,
) -> str:
    """
    Retrieves the complete content or metadata of a specific document identified by its unique ID.
    """
    try:
        from core.tools.document_tools import retrieve_document as _retrieve_document

        result = await _retrieve_document(
            document_id=document_id,
            format=format,
            end_user_id=end_user_id,
            document_service=ctx.deps.document_service,
            auth=ctx.deps.auth,
        )

        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}")
        return f"Lỗi khi lấy tài liệu {document_id}: {str(e)}"


@retrieval_agent.tool
async def document_analyzer(
    ctx: RunContext[MorphikDeps],
    document_id: str,
    analysis_type: str = "full",
) -> str:
    """
    Analyzes documents to extract structured information including entities, relationships, key facts, and sentiment.
    """
    try:
        from core.tools.analysis_tools import document_analyzer as _document_analyzer

        result = await _document_analyzer(
            document_id=document_id,
            analysis_type=analysis_type,
            document_service=ctx.deps.document_service,
            auth=ctx.deps.auth,
        )

        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    except Exception as e:
        logger.error(f"Error analyzing document {document_id}: {e}")
        return f"Lỗi khi phân tích tài liệu {document_id}: {str(e)}"


@retrieval_agent.tool
async def list_documents(
    ctx: RunContext[MorphikDeps],
    filters: Optional[Dict[str, Any]] = None,
    skip: int = 0,
    limit: int = 100,
    folder_name: Optional[str] = None,
    end_user_id: Optional[str] = None,
) -> str:
    """
    Lists accessible documents, showing their IDs and filenames.
    For Vietnamese contracts (hợp đồng), use folder_name 'folder-contracts' to list contract documents.
    """
    try:
        from core.tools.document_tools import list_documents as _list_documents

        result = await _list_documents(
            filters=filters,
            skip=skip,
            limit=limit,
            folder_name=folder_name,
            end_user_id=end_user_id,
            document_service=ctx.deps.document_service,
            auth=ctx.deps.auth,
        )

        if isinstance(result, (list, dict)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return f"Lỗi khi liệt kê tài liệu: {str(e)}"


@retrieval_agent.tool
async def knowledge_graph_query(
    ctx: RunContext[MorphikDeps],
    query_type: str,
    start_nodes: List[str],
    max_depth: int = 3,
    graph_name: Optional[str] = None,
    end_user_id: Optional[str] = None,
) -> str:
    """
    Queries the knowledge graph to explore entities, relationships, and connections.
    """
    try:
        from core.tools.graph_tools import knowledge_graph_query as _knowledge_graph_query

        result = await _knowledge_graph_query(
            query_type=query_type,
            start_nodes=start_nodes,
            max_depth=max_depth,
            graph_name=graph_name,
            end_user_id=end_user_id,
            document_service=ctx.deps.document_service,
            auth=ctx.deps.auth,
        )

        if isinstance(result, (list, dict)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    except Exception as e:
        logger.error(f"Error querying knowledge graph: {e}")
        return f"Lỗi khi truy vấn đồ thị tri thức: {str(e)}"


async def run_vietnamese_agent(
    query: str,
    document_service: DocumentService,
    auth: AuthContext,
    conversation_history: list = None,
    display_mode: str = "formatted",
) -> dict:
    """
    Run the Vietnamese contract agent with the given query.

    Returns dict compatible with existing morphik API:
    {
        "response": str,
        "tool_history": list,
        "display_objects": list,
        "sources": list,
    }

    Args:
        query: User query in Vietnamese or English
        document_service: DocumentService instance
        auth: Authentication context
        conversation_history: Previous conversation messages (not used yet)
        display_mode: Display mode for the response (not used yet)

    Returns:
        Dictionary with response, tool_history, display_objects, and sources
    """
    try:
        # Initialize query analyzer
        query_analyzer = None
        intent_instructions = ""
        try:
            query_analyzer = create_vietnamese_query_analyzer(enable_semantic_analysis=True)
            logger.info("Query analyzer initialized successfully")
            # Get intent-specific instructions if available
            analysis = await query_analyzer.analyze(query)
            intent_instructions = analysis.instruction_context
            logger.info(f"Query intent: {analysis.intent.value}")
        except Exception as e:
            logger.warning(f"Failed to initialize query analyzer: {e}")

        deps = MorphikDeps(
            document_service=document_service,
            auth=auth,
            query_analyzer=query_analyzer
        )

        # PYTHON-ORCHESTRATED THREE-STAGE PIPELINE (following pdf-qa-system pattern)
        # Python code calls each step directly in sequence, not relying on LLM tool calling

        # STAGE 1: Retrieval - Direct call to document service (no agent needed)
        logger.info("Stage 1: Retrieving relevant chunks...")

        # Get retrieval parameters from query analysis
        folder_name = "folder-contracts"
        k = 20  # Increased for better coverage
        min_relevance = 0.5  # Lowered threshold to get more results

        if query_analyzer:
            try:
                analysis = await query_analyzer.analyze(query)
                folder_name = analysis.folder_name
                logger.info(f"Query analysis: {analysis.reasoning}")
            except Exception as e:
                logger.warning(f"Query analysis failed: {e}")

        # Direct semantic search - no LLM tool calling needed!
        chunks = await document_service.retrieve_chunks(
            query=query,
            auth=auth,
            filters=None,
            k=k,
            min_score=min_relevance,
            use_colpali=True,
            use_reranking=True,
            folder_name=folder_name,
            end_user_id=None,
        )

        deps.retrieved_chunks = chunks
        logger.info(f"Stage 1 complete. Retrieved {len(chunks)} chunks")

        if not deps.retrieved_chunks:
            response_text = "Không tìm thấy tài liệu nào phù hợp với câu hỏi của bạn."
        else:
            # STAGE 2: Extraction - Direct Python call to extract clean data
            logger.info("Stage 2: Extracting relevant data from chunks...")
            clean_data = await extract_relevant_data(query, deps.retrieved_chunks)
            logger.info(f"Stage 2 complete. Extracted {len(clean_data)} chars of clean data")
            logger.info(f"Clean data preview (first 500 chars): {clean_data[:500]}")

            # STAGE 3: Generation - Direct Python call to Gemma 3 12B for Vietnamese answer
            if clean_data and "Error" not in clean_data:
                logger.info("Stage 3: Generating Vietnamese answer with Gemma 3 12B...")
                response_text = await call_viqwen2_rag(query, clean_data, intent_instructions)
                logger.info(f"Stage 3 complete. Generated {len(response_text)} chars")
            else:
                response_text = "Không thể trích xuất dữ liệu từ tài liệu."

        # Extract sources from retrieved chunks
        sources = []
        if deps.retrieved_chunks:
            for chunk in deps.retrieved_chunks:
                sources.append({
                    "sourceId": f"{chunk.filename}_{chunk.chunk_index if hasattr(chunk, 'chunk_index') else 'unknown'}",
                    "documentName": chunk.filename or "Unknown Document",
                    "documentId": chunk.document_id if hasattr(chunk, 'document_id') else "unknown",
                    "content": chunk.content,
                    "score": chunk.score if hasattr(chunk, 'score') else None,
                })

        # Format as morphik API expects
        return {
            "response": response_text,
            "tool_history": [],  # TODO: Extract from result if needed
            "display_objects": [
                {
                    "type": "text",
                    "content": response_text,
                    "source": "viqwen2-rag"
                }
            ],
            "sources": sources,
        }

    except Exception as e:
        logger.error(f"Error running Vietnamese agent: {e}")
        error_msg = f"Lỗi khi chạy trợ lý: {str(e)}"
        return {
            "response": error_msg,
            "tool_history": [],
            "display_objects": [
                {
                    "type": "text",
                    "content": error_msg,
                    "source": "error"
                }
            ],
            "sources": [],
        }