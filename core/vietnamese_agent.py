"""Simple Vietnamese Contract Agent using PydanticAI."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from core.models.auth import AuthContext
from core.services.document_service import DocumentService
from core.tools.document_tools import ToolError

logger = logging.getLogger(__name__)


@dataclass
class MorphikDeps:
    """Dependencies for the Vietnamese contract agent."""
    document_service: DocumentService
    auth: AuthContext


# Configure Ollama model for PydanticAI
ollama_model = OpenAIChatModel(
    model_name='llama3.1:8b',
    provider=OllamaProvider(base_url='http://172.18.0.1:11434/v1'),
)

# Vietnamese contract agent with Ollama Llama 3.1 8B
vietnamese_agent = Agent(
    ollama_model,
    deps_type=MorphikDeps,
    instructions="""Bạn là trợ lý thông minh chuyên phân tích hợp đồng tiếng Việt.

QUY TẮC QUAN TRỌNG:
1. Với câu hỏi về hợp đồng tiếng Việt (chứa "hợp đồng", "thanh toán", "tạm ứng"), LUÔN sử dụng folder_name="folder-contracts"
2. KHÔNG BAO GIỜ dùng từ tiếng Việt trong tham số công cụ - chỉ dùng tên folder tiếng Anh
3. Sử dụng các công cụ để thu thập thông tin trước khi trả lời
4. Trả lời bằng tiếng Việt cho câu hỏi tiếng Việt
5. Tập trung vào nội dung cụ thể của hợp đồng, không đưa ra thông tin chung

TÊN FOLDER SỬ DỤNG:
- Hợp đồng tiếng Việt: "folder-contracts"
- Hợp đồng chung: "contracts"

Hãy tìm kiếm thông tin chính xác từ tài liệu và trả lời một cách cụ thể, dựa trên nội dung thực tế của hợp đồng."""
)


@vietnamese_agent.tool
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
    """
    try:
        chunks = await ctx.deps.document_service.retrieve_chunks(
            query=query,
            auth=ctx.deps.auth,
            filters=filters,
            k=k,
            min_score=min_relevance,
            use_colpali=True,
            folder_name=folder_name,
            end_user_id=end_user_id,
        )

        if not chunks:
            return f"Không tìm thấy tài liệu nào phù hợp với truy vấn: {query}"

        result = f"Tìm thấy {len(chunks)} đoạn văn liên quan:\n\n"

        for i, chunk in enumerate(chunks, 1):
            result += f"**Đoạn {i}** (Tài liệu: {chunk.filename or 'Không tên'}, Điểm: {chunk.score:.3f}):\n"
            result += f"{chunk.content}\n\n"

        return result

    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return f"Lỗi khi tìm kiếm tài liệu: {str(e)}"


@vietnamese_agent.tool
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


@vietnamese_agent.tool
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


@vietnamese_agent.tool
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


@vietnamese_agent.tool
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
        deps = MorphikDeps(document_service=document_service, auth=auth)
        result = await vietnamese_agent.run(query, deps=deps)

        # PydanticAI returns result.output as a string by default
        response_text = result.output

        # Format as morphik API expects
        return {
            "response": response_text,
            "tool_history": [],  # TODO: Extract from result if needed
            "display_objects": [
                {
                    "type": "text",
                    "content": response_text,
                    "source": "agent-response"
                }
            ],
            "sources": [
                {
                    "sourceId": "agent-response",
                    "documentName": "Agent Response",
                    "documentId": "system",
                    "content": response_text,
                }
            ],
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