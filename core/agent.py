import json
import logging
import os
import re

import httpx
from dotenv import load_dotenv
from litellm import acompletion
from litellm.exceptions import ContextWindowExceededError

from core.config import get_settings
from core.models.auth import AuthContext
from core.tools.tools import (
    document_analyzer,
    execute_code,
    knowledge_graph_query,
    list_documents,
    list_graphs,
    retrieve_chunks,
    retrieve_document,
    save_to_memory,
)
from core.utils.agent_helpers import crop_images_in_display_objects, extract_display_object

logger = logging.getLogger(__name__)


def _truncate_for_log(obj, limit=100):
    s = str(obj)
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


# Load environment variables
load_dotenv(override=True)


class MorphikAgent:
    """
    Morphik agent for orchestrating tools via LiteLLM function calling.
    """

    def __init__(
        self,
        document_service,
        model: str = None,
    ):
        self.document_service = document_service
        # Load settings
        self.settings = get_settings()
        self.model = model or self.settings.AGENT_MODEL
        # Load tool definitions (function schemas)
        desc_path = os.path.join(os.path.dirname(__file__), "tools", "descriptions.json")
        with open(desc_path, "r") as f:
            self.tools_json = json.load(f)

        self.tool_definitions = []
        graph_mode = self.settings.GRAPH_MODE if hasattr(self.settings, "GRAPH_MODE") else "local"

        for tool in self.tools_json:
            # Filter tools based on graph mode
            if graph_mode == "api" and tool["name"] == "knowledge_graph_query":
                # Skip complex local-graph query tool when using remote API graphs
                continue
            if graph_mode != "api" and tool["name"] == "graph_api_retrieve":
                # Skip API-specific retrieval tool when using local graphs
                continue
            self.tool_definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
            )

        content_guidelines = (
            "for text objects, this is markdown content; for image objects, this is a description for the "
        )
        content_guidelines += (
            "image, describing the exact part you want to extract from the source chunk. This description will be "
        )
        content_guidelines += "used to create a bounding box around the image and extract the image from the source chunk. Be as precise as possible. "
        content_guidelines += "Use labels, diagram numbers, etc. where possible to be more precise. Please ensure that when you choose an image display "
        content_guidelines += "object, the corresponding source is also an image."

        example_response = """
```json
[
  {
    "type": "text",
    "content": "## Introduction to the Topic\nHere is some detailed information...",
    "source": "doc123-chunk1"
  },
  {
    "type": "text",
    "content": "This analysis shows that...",
    "source": "doc456-chunk2"
  }
]
```
"""
        # Build bullet list based on graph mode
        bullet_parts = [
            "- retrieve_chunks: retrieve relevant text and image chunks from the knowledge base",
            "- retrieve_document: get full document content or metadata",
            "- document_analyzer: analyze documents for entities, facts, summary, sentiment, or full analysis",
            "- execute_code: run Python code in a safe sandbox",
        ]

        if graph_mode == "api":
            bullet_parts.append("- graph_api_retrieve: retrieve answers from a remote Morphik knowledge graph")
        else:
            bullet_parts.append(
                "- knowledge_graph_query: query the knowledge graph for entities, paths, subgraphs, or list entities"
            )

        bullet_parts.extend(
            [
                "- list_graphs: list available knowledge graphs",
                "- save_to_memory: save important information to persistent memory",
                "- list_documents: list documents accessible to you",
            ]
        )

        bullet_lines = "\n".join(bullet_parts)

        # Store bullet_lines for use in template formatting
        self.bullet_lines = bullet_lines

        # System prompt template (will be formatted with query and bullet_lines at runtime)
        self.system_prompt_template = """
You are Morphik, an intelligent research assistant. Your role is to answer the following query: {query}

**ALWAYS RESPOND IN VIETNAMESE FOR VIETNAMESE QUERIES.**

**CRITICAL GROUNDING RULE**:
- ONLY answer based on information retrieved from tools (retrieve_chunks, retrieve_document, etc.)
- DO NOT use your own knowledge or make assumptions
- If the retrieved information doesn't contain the answer, say "Không tìm thấy thông tin" (Information not found)
- NEVER hallucinate or invent information not present in the tool results
- When you receive retrieved information, use ONLY that information to answer the query

You can use the following tools to help answer user queries:
{bullet_lines}

TOOL SELECTION GUIDE:
- **PRIMARY TOOL**: Use retrieve_chunks for almost ALL queries including: prices, equipment, specifications, factual data, lists, "what/which/how much/liệt kê/thiết bị/giá" queries
- Use knowledge_graph_query ONLY for: complex relationships, networks, "who works with whom", multi-entity connections
- ONLY call list_graphs before knowledge_graph_query IF you need to query graphs (not for document/chunk retrieval)

IMPORTANT RULES:
1. For Vietnamese contract queries (containing "hợp đồng", "thanh toán", "tạm ứng"), use folder_name="folder-contracts"
2. Always use English folder names in tool parameters
3. Use function calls to gather information before responding

FOLDER NAMES TO USE:
- Vietnamese contracts: "folder-contracts"
- General contracts: "contracts"

When you have gathered information using tools, provide a final response as a JSON array of display objects:

```json
[
  {{
    "type": "text",
    "content": "Your answer in Vietnamese with markdown formatting, STRICTLY based on the retrieved information",
    "source": "source-id-from-chunks"
  }}
]
```

Always cite sources and provide accurate information STRICTLY from the retrieved chunks.
""".strip()

    async def _execute_tool(self, name: str, args: dict, auth: AuthContext, source_map: dict):
        """Dispatch tool calls, injecting document_service and auth."""
        match name:
            case "retrieve_chunks":
                # Remove document_id if it was incorrectly included
                # (model sometimes confuses retrieve_chunks with retrieve_document)
                filtered_args = {k: v for k, v in args.items() if k != "document_id"}
                content, found_sources = await retrieve_chunks(
                    document_service=self.document_service, auth=auth, **filtered_args
                )
                source_map.update(found_sources)
                return content
            case "retrieve_document":
                result = await retrieve_document(document_service=self.document_service, auth=auth, **args)
                # Add document as a source if it's a successful retrieval
                if isinstance(result, str) and not result.startswith("Document") and not result.startswith("Error"):
                    doc_id = args.get("document_id", "unknown")
                    source_id = f"doc{doc_id}-full"
                    source_map[source_id] = {
                        "document_id": doc_id,
                        "document_name": f"Full Document {doc_id}",
                        "chunk_number": "full",
                    }
                return result
            case "document_analyzer":
                result = await document_analyzer(document_service=self.document_service, auth=auth, **args)
                # Track document being analyzed as a source
                if args.get("document_id"):
                    doc_id = args.get("document_id")
                    analysis_type = args.get("analysis_type", "analysis")
                    source_id = f"doc{doc_id}-{analysis_type}"
                    source_map[source_id] = {
                        "document_id": doc_id,
                        "document_name": f"Document {doc_id} ({analysis_type})",
                        "analysis_type": analysis_type,
                    }
                return result
            case "execute_code":
                res = await execute_code(**args)
                return res["content"]
            case "knowledge_graph_query":
                if self.settings.GRAPH_MODE == "api":
                    from core.tools.graph_tools_api import graph_api_retrieve

                    return await graph_api_retrieve(document_service=self.document_service, auth=auth, **args)
                return await knowledge_graph_query(document_service=self.document_service, auth=auth, **args)
            case "graph_api_retrieve":
                from core.tools.graph_tools_api import graph_api_retrieve

                return await graph_api_retrieve(document_service=self.document_service, auth=auth, **args)
            case "list_graphs":
                return await list_graphs(document_service=self.document_service, auth=auth, **args)
            case "save_to_memory":
                return await save_to_memory(document_service=self.document_service, auth=auth, **args)
            case "list_documents":
                return await list_documents(document_service=self.document_service, auth=auth, **args)
            case _:
                raise ValueError(f"Unknown tool: {name}")

    async def _run_ollama_direct(
        self,
        messages: list,
        model_config: dict,
        auth: AuthContext,
        source_map: dict,
        tool_history: list,
        display_mode: str = "formatted",
    ) -> dict:
        """Direct Ollama API integration bypassing LiteLLM for reliable tool calling"""

        # Extract Ollama configuration
        api_base = model_config.get("api_base")
        model_name_full = model_config.get("model_name")

        # Parse base model name: "qwen3:32b" from "ollama_chat/qwen3:32b"
        match = re.search(r"[^/]+$", model_name_full)
        base_model_name = match.group(0) if match else model_name_full

        logger.info(f"Using direct Ollama API: {api_base}, model: {base_model_name}")

        async with httpx.AsyncClient(timeout=600.0) as client:
            while True:
                # Prepare Ollama request
                ollama_request = {
                    "model": base_model_name,
                    "messages": messages,
                    "tools": self.tool_definitions,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Force deterministic grounding to retrieved data
                        "num_ctx": 16384,  # 16k context window - balance between quality and speed
                    },
                }

                logger.info(f"Ollama direct: Sending request with {len(messages)} messages")
                # Log each message for debugging
                for i, msg in enumerate(messages):
                    role = msg.get('role', 'unknown')
                    content = str(msg.get('content', ''))
                    content_preview = content[:300]
                    logger.info(f"Message {i}: role={role}, length={len(content)}, preview: {content_preview}")
                    if i == 3:  # Log full message 3 (retrieved info)
                        logger.info(f"FULL Message 3 content: {content[:2000]}")

                # Call Ollama native /api/chat
                response = await client.post(f"{api_base}/api/chat", json=ollama_request)
                response.raise_for_status()
                result = response.json()

                message = result["message"]
                logger.info(f"Ollama response: {_truncate_for_log(message)}")

                # Check for tool calls
                tool_calls = message.get("tool_calls")
                if not tool_calls:
                    # Final response - parse display objects and return
                    logger.info("Ollama: No tool calls detected, returning final response")

                    # Parse display objects (same logic as LiteLLM path)
                    display_objects = []
                    content = message.get("content", "")

                    try:
                        from core.utils.agent_helpers import parse_json

                        content_to_parse = content.strip()

                        # Check if wrapped in markdown code blocks
                        if content_to_parse.startswith("```json") and content_to_parse.endswith("```"):
                            content_to_parse = parse_json(content_to_parse)
                        elif content_to_parse.startswith("```") and content_to_parse.endswith("```"):
                            lines = content_to_parse.split("\n")
                            if len(lines) > 2:
                                content_to_parse = "\n".join(lines[1:-1])

                        # Try to parse as JSON
                        try:
                            parsed_content = json.loads(content_to_parse)
                            if isinstance(parsed_content, list):
                                for item in parsed_content:
                                    if isinstance(item, dict) and "type" in item and "content" in item:
                                        display_obj = extract_display_object(item, source_map)
                                        if not display_obj.get("invalid"):
                                            display_objects.append(display_obj)
                        except json.JSONDecodeError:
                            pass  # Not JSON, use raw content

                    except Exception as e:
                        logger.warning(f"Error parsing display objects: {e}")

                    # Build sources list
                    sources = []
                    seen_source_ids = set()
                    for obj in display_objects:
                        if "source" in obj and obj["source"] not in seen_source_ids:
                            seen_source_ids.add(obj["source"])

                    for source_id, source_info in source_map.items():
                        if source_id not in seen_source_ids:
                            sources.append(
                                {
                                    "sourceId": source_id,
                                    "documentName": source_info.get("document_name", "Unknown Document"),
                                    "documentId": source_info.get("document_id", "unknown"),
                                }
                            )

                    if display_mode == "formatted":
                        display_objects = crop_images_in_display_objects(display_objects)

                    # Generate response text
                    response_text = content
                    if display_objects:
                        text_contents = []
                        for obj in display_objects:
                            if obj.get("type") == "text" and obj.get("content"):
                                text_contents.append(obj["content"])
                        if text_contents:
                            response_text = "\n\n".join(text_contents)

                    return {
                        "response": response_text,
                        "tool_history": tool_history,
                        "display_objects": display_objects,
                        "sources": sources,
                    }

                # Execute tools
                logger.info(f"Ollama: {len(tool_calls)} tool calls detected")

                # Add assistant message to conversation
                messages.append(
                    {"role": "assistant", "content": message.get("content", ""), "tool_calls": tool_calls}
                )

                # Execute each tool
                for tool_call in tool_calls:
                    func = tool_call["function"]
                    name = func["name"]
                    args = func.get("arguments", {})

                    logger.info(f"Ollama: Executing tool {name} with args: {_truncate_for_log(args)}")

                    # Use existing tool execution logic
                    tool_result = await self._execute_tool(name, args, auth, source_map)

                    logger.info(f"Ollama: Tool {name} result: {_truncate_for_log(tool_result)}")

                    # Add to history
                    tool_history.append({"tool_name": name, "tool_args": args, "tool_result": tool_result})

                    # Add tool result to conversation (Ollama format - simple content, no tool_call_id)
                    # Extract text from structured content for better model comprehension
                    if isinstance(tool_result, list):
                        # Extract text from list of dicts with 'type' and 'text' fields
                        text_parts = []
                        for item in tool_result:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                        tool_content = '\n\n'.join(text_parts) if text_parts else json.dumps(tool_result)
                    elif isinstance(tool_result, str):
                        tool_content = tool_result
                    else:
                        tool_content = json.dumps(tool_result)

                    logger.info(f"Ollama: Tool content sent to model (first 500 chars): {tool_content[:500]}")

                    # Use "user" role instead of "tool" for better grounding
                    # Extract the original query from messages[1] to repeat it explicitly
                    original_query = messages[1].get("content", "the query") if len(messages) > 1 else "the query"
                    user_message = f"RETRIEVED INFORMATION:\n\n{tool_content}\n\nNow answer this query: '{original_query}' using ONLY the retrieved information above. Do not use your own knowledge."
                    logger.info(f"Ollama: Appending USER role message (first 100 chars): {user_message[:100]}")
                    messages.append({"role": "user", "content": user_message})

                logger.info("Ollama: All tools executed, continuing conversation...")

    async def run(
        self, query: str, auth: AuthContext, conversation_history: list = None, display_mode: str = "formatted"
    ) -> str:
        """Synchronously run the agent and return the final answer."""
        # Per-run state to avoid cross-request leakage
        source_map: dict = {}

        # Format system prompt with the actual query and bullet_lines
        system_prompt = self.system_prompt_template.format(query=query, bullet_lines=self.bullet_lines)

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history[:-1]:  # Exclude the last message (current user query)
                # Properly handle all message types including tool messages and assistant messages with tool calls
                if isinstance(msg, dict):
                    # Copy the entire message to preserve all fields (tool_call_id, name, tool_calls, etc.)
                    messages.append(msg)
                else:
                    # Fallback for simple message objects
                    messages.append({"role": msg["role"], "content": msg["content"]})

        # Add the current user query
        messages.append({"role": "user", "content": query})

        tool_history = []  # Initialize tool history list
        # Get the full model name from the registered models config
        settings = get_settings()
        if self.model not in settings.REGISTERED_MODELS:
            raise ValueError(f"Model '{self.model}' not found in registered_models configuration")

        model_config = settings.REGISTERED_MODELS[self.model]
        model_name = model_config.get("model_name")

        # Prepare model parameters
        model_params = {
            "model": model_name,
            "messages": messages,
            "tools": self.tool_definitions,
            "tool_choice": "auto",
        }

        # Add any other parameters from model config
        for key, value in model_config.items():
            if key != "model_name":
                model_params[key] = value

        # Check if we're using Ollama model - use direct Ollama client for better function calling
        if "ollama" in model_name.lower():
            return await self._run_ollama_direct(messages, model_config, auth, source_map, tool_history, display_mode)

        while True:
            logger.info(f"Sending completion request with {len(messages)} messages")
            try:
                resp = await acompletion(**model_params)
            except ContextWindowExceededError as e:
                logger.info("Context window exceeded, truncating messages")
                # Save messages to JSON for debugging or analysis
                debug_dir = os.path.join(os.getcwd(), "debug_logs")
                os.makedirs(debug_dir, exist_ok=True)

                # Create a unique filename with timestamp
                import datetime

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(debug_dir, f"agent_messages_{timestamp}.json")

                # Save the current messages to the file
                with open(log_file, "w") as f:
                    json.dump(messages, f, indent=2)

                logger.info(f"Saved messages to {log_file}")
                raise e
            logger.info(f"Received response: {resp}")

            msg = resp.choices[0].message
            # If no tool call, return final content
            if not getattr(msg, "tool_calls", None):
                logger.info("No tool calls detected, returning final content")

                # Parse the response as display objects if possible
                display_objects = []
                default_text = ""

                try:
                    # Check if the response is JSON formatted
                    import re

                    from core.utils.agent_helpers import parse_json

                    # First try to parse the entire response as JSON
                    content_to_parse = msg.content.strip()

                    # Check if it's wrapped in markdown code blocks
                    if content_to_parse.startswith("```json") and content_to_parse.endswith("```"):
                        content_to_parse = parse_json(content_to_parse)
                    elif content_to_parse.startswith("```") and content_to_parse.endswith("```"):
                        # Extract content from any code block
                        lines = content_to_parse.split("\n")
                        if len(lines) > 2:
                            content_to_parse = "\n".join(lines[1:-1])

                    # Try to parse as complete JSON first
                    try:
                        parsed_content = json.loads(content_to_parse)

                        # Handle both array and object formats
                        if isinstance(parsed_content, list):
                            for item in parsed_content:
                                if isinstance(item, dict) and "type" in item and "content" in item:
                                    display_obj = extract_display_object(item, source_map)
                                    if not display_obj.get("invalid"):
                                        display_objects.append(display_obj)
                        elif (
                            isinstance(parsed_content, dict)
                            and "type" in parsed_content
                            and "content" in parsed_content
                        ):
                            display_obj = extract_display_object(parsed_content, source_map)
                            if not display_obj.get("invalid"):
                                display_objects.append(display_obj)

                    except json.JSONDecodeError:
                        # If complete parsing fails, try to extract JSON arrays or objects
                        json_array_pattern = r'\[\s*\{[^}]*"type"\s*:[^}]*"content"\s*:[^}]*\}[^]]*\]'
                        json_object_pattern = r'\{\s*"type"\s*:[^}]*"content"\s*:[^}]*\}'

                        # Try array pattern first
                        array_match = re.search(json_array_pattern, content_to_parse, re.DOTALL)
                        if array_match:
                            try:
                                parsed_content = json.loads(array_match.group(0))
                                if isinstance(parsed_content, list):
                                    for item in parsed_content:
                                        if isinstance(item, dict) and "type" in item and "content" in item:
                                            display_obj = extract_display_object(item, source_map)
                                            if not display_obj.get("invalid"):
                                                display_objects.append(display_obj)
                            except json.JSONDecodeError:
                                pass

                        # Try object pattern if array didn't work
                        if not display_objects:
                            object_matches = re.findall(json_object_pattern, content_to_parse, re.DOTALL)
                            for match in object_matches:
                                try:
                                    parsed_content = json.loads(match)
                                    if (
                                        isinstance(parsed_content, dict)
                                        and "type" in parsed_content
                                        and "content" in parsed_content
                                    ):
                                        display_obj = extract_display_object(parsed_content, source_map)
                                        if not display_obj.get("invalid"):
                                            display_objects.append(display_obj)
                                except json.JSONDecodeError:
                                    continue

                    # If no display objects were created, treat the entire content as text
                    if not display_objects:
                        default_text = msg.content

                except Exception as e:
                    logger.warning(f"Failed to parse response as JSON: {e}")
                    default_text = msg.content

                # If no structured display objects were found, create a default text object
                if not display_objects and default_text:
                    display_objects.append({"type": "text", "content": default_text, "source": "agent-response"})

                # Create sources from the collected source IDs in display objects
                sources = []
                seen_source_ids = set()

                for obj in display_objects:
                    source_id = obj.get("source")
                    if source_id and source_id != "agent-response" and source_id not in seen_source_ids:
                        seen_source_ids.add(source_id)
                        # Extract document info from source ID if available
                        if "-" in source_id:
                            parts = source_id.split("-", 1)
                            doc_id = parts[0].replace("doc", "")
                            sources.append(
                                {
                                    "sourceId": source_id,
                                    "documentName": f"Document {doc_id}",
                                    "documentId": doc_id,
                                    "content": source_map.get(source_id, {"content": ""}).get("content", ""),
                                }
                            )
                        else:
                            sources.append(
                                {
                                    "sourceId": source_id,
                                    "documentName": "Referenced Source",
                                    "documentId": "unknown",
                                    "content": source_map.get(source_id, {"content": ""}).get("content", ""),
                                }
                            )

                # Add agent response source if not already included
                if "agent-response" not in seen_source_ids:
                    sources.append(
                        {
                            "sourceId": "agent-response",
                            "documentName": "Agent Response",
                            "documentId": "system",
                            "content": msg.content,
                        }
                    )

                # Add sources from document chunks used during the session
                for source_id, source_info in source_map.items():
                    if source_id not in seen_source_ids:
                        sources.append(
                            {
                                "sourceId": source_id,
                                "documentName": source_info.get("document_name", "Unknown Document"),
                                "documentId": source_info.get("document_id", "unknown"),
                            }
                        )

                # Return final content, tool history, display objects and sources
                if display_mode == "formatted":
                    display_objects = crop_images_in_display_objects(display_objects)

                # Generate a user-friendly response text from display objects
                response_text = ""
                if display_objects:
                    # Extract text content from display objects for a clean response
                    text_contents = []
                    for obj in display_objects:
                        if obj.get("type") == "text" and obj.get("content"):
                            text_contents.append(obj["content"])

                    if text_contents:
                        # Join text contents with proper spacing
                        response_text = "\n\n".join(text_contents)
                    else:
                        # If no text objects, provide a generic response
                        response_text = "I've found relevant information in the documents. Please see the display objects above for details."
                else:
                    # Fallback to original content if no display objects
                    response_text = msg.content

                return {
                    "response": response_text,
                    "tool_history": tool_history,
                    "display_objects": display_objects,
                    "sources": sources,
                }

            # Process ALL tool calls in the assistant message
            logger.info(f"Tool calls detected: {len(msg.tool_calls)} calls")

            # Add the assistant message with tool calls to conversation
            messages.append(msg.to_dict(exclude_none=True))

            # Execute each tool call and add responses
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)
                logger.info(f"Tool call detected: {name} with args: {_truncate_for_log(args)}")

                logger.info(f"Executing tool: {name}")
                result = await self._execute_tool(name, args, auth, source_map)
                logger.info(f"Tool execution result: {_truncate_for_log(result)}")

                # Add tool call and result to history
                tool_history.append({"tool_name": name, "tool_args": args, "tool_result": result})

                # Append raw tool output (string or structured data)
                content = [{"type": "text", "text": result}] if isinstance(result, str) else result
                messages.append({"role": "tool", "name": name, "content": content, "tool_call_id": call.id})

            logger.info("Added all tool results to conversation, continuing...")

    def stream(self, query: str):
        """
        (Streaming stub) In future, this will:
          - yield f"[ToolCall] {tool_name}({args})" when a tool is invoked
          - yield f"[ToolResult] {tool_name} -> {result}" after execution
        For now, streaming is disabled; use run() to get the complete answer.
        """
        raise NotImplementedError("Streaming not supported yet; please use run()")
