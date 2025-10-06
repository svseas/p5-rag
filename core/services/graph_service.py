import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel

from core.completion.base_completion import BaseCompletionModel
from core.config import get_settings
from core.database.base_database import BaseDatabase
from core.embedding.base_embedding_model import BaseEmbeddingModel
from core.models.auth import AuthContext
from core.models.completion import ChunkSource, CompletionRequest, CompletionResponse
from core.models.documents import ChunkResult, Document
from core.models.graph import Entity, Graph, Relationship
from core.models.prompts import EntityExtractionPromptOverride, GraphPromptOverrides, QueryPromptOverrides
from core.services.entity_resolution import EntityResolver

logger = logging.getLogger(__name__)


class EntityExtraction(BaseModel):
    """Model for entity extraction results"""

    label: str
    type: str
    properties: Dict[str, Any] = {}


class RelationshipExtraction(BaseModel):
    """Model for relationship extraction results"""

    source: str
    target: str
    relationship: str


class ExtractionResult(BaseModel):
    """Model for structured extraction from LLM"""

    entities: List[EntityExtraction] = []
    relationships: List[RelationshipExtraction] = []


class GraphService:
    """Service for managing knowledge graphs and graph-based operations"""

    def __init__(
        self,
        db: BaseDatabase,
        embedding_model: BaseEmbeddingModel,
        completion_model: BaseCompletionModel,
    ):
        self.db = db
        self.embedding_model = embedding_model
        self.completion_model = completion_model
        self.entity_resolver = EntityResolver()

    async def update_graph(
        self,
        name: str,
        auth: AuthContext,
        document_service,  # Passed in to avoid circular import
        additional_filters: Optional[Dict[str, Any]] = None,
        additional_documents: Optional[List[str]] = None,
        prompt_overrides: Optional[GraphPromptOverrides] = None,
        system_filters: Optional[Dict[str, Any]] = None,
        is_initial_build: bool = False,
    ) -> Graph:
        """Update an existing graph with new documents.

        This function processes additional documents matching the original or new filters,
        extracts entities and relationships, and updates the graph with new information.

        Args:
            name: Name of the graph to update
            auth: Authentication context
            document_service: DocumentService instance for retrieving documents and chunks
            additional_filters: Optional additional metadata filters to determine which new documents to include
            additional_documents: Optional list of specific additional document IDs to include
            prompt_overrides: Optional GraphPromptOverrides with customizations for prompts
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)
            to determine which documents to include
            is_initial_build: Whether this is the initial build of the graph

        Returns:
            Graph: The updated graph
        """
        # Initialize system_filters if None
        if system_filters is None:
            system_filters = {}

        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        # Get the existing graph with system filters for proper user_id scoping
        existing_graph = await self.db.get_graph(name, auth, system_filters=system_filters)
        if not existing_graph:
            raise ValueError(f"Graph '{name}' not found")

        # Check if the graph is currently being processed by another operation
        if existing_graph.system_metadata.get("status") == "processing" and not is_initial_build:
            raise ValueError(
                f"Graph '{name}' is currently being processed and cannot be updated yet. "
                f"Please wait for the creation process to complete."
            )

        # Ensure app_id scoping: update app_id if this is a developer-scoped token
        if auth.app_id and existing_graph.app_id != auth.app_id:
            existing_graph.app_id = auth.app_id

        # Track explicitly added documents to ensure they're included in the final graph
        # even if they don't have new entities or relationships
        explicit_doc_ids = set(additional_documents or [])

        # Find new documents to process
        document_ids = await self._get_new_document_ids(
            auth, existing_graph, additional_filters, additional_documents, system_filters
        )

        if not document_ids and not explicit_doc_ids:
            # No new documents to add
            existing_graph.system_metadata["status"] = "completed"
            await self.db.update_graph(existing_graph)
            return existing_graph

        # Create a set for all document IDs that should be included in the updated graph
        # Includes existing document IDs, explicitly added document IDs, and documents found via filters
        all_doc_ids = set(existing_graph.document_ids).union(document_ids).union(explicit_doc_ids)
        logger.info(f"Total document IDs to include in updated graph: {len(all_doc_ids)}")

        # Batch retrieve all document IDs (both regular and explicit) in a single call
        all_ids_to_retrieve = list(document_ids)

        # Add explicit document IDs if not already included
        if explicit_doc_ids and additional_documents:
            # Add any missing IDs to the list
            for doc_id in additional_documents:
                if doc_id not in document_ids:
                    all_ids_to_retrieve.append(doc_id)

        # Batch retrieve all documents in a single call
        document_objects = await document_service.batch_retrieve_documents(
            all_ids_to_retrieve,
            auth,
            system_filters.get("folder_name", None),
            system_filters.get("end_user_id", None),
        )

        # Process explicit documents if needed
        if explicit_doc_ids and additional_documents:
            # Extract authorized explicit IDs from the retrieved documents
            authorized_explicit_ids = {
                doc.external_id for doc in document_objects if doc.external_id in explicit_doc_ids
            }
            logger.info(
                f"Authorized explicit document IDs: {len(authorized_explicit_ids)} out of {len(explicit_doc_ids)}"
            )

            # Update document_ids and all_doc_ids
            document_ids.update(authorized_explicit_ids)
            all_doc_ids.update(authorized_explicit_ids)

        # If we have additional filters, make sure we include the document IDs from filter matches
        # even if they don't have new entities or relationships
        if additional_filters:
            filtered_docs = await document_service.batch_retrieve_documents(
                [doc_id for doc_id in all_doc_ids if doc_id not in {d.external_id for d in document_objects}],
                auth,
                system_filters.get("folder_name", None),
                system_filters.get("end_user_id", None),
            )
            logger.info(f"Additional filtered documents to include: {len(filtered_docs)}")
            document_objects.extend(filtered_docs)

        if not document_objects:
            # No authorized new documents
            return existing_graph

        # Validation is now handled by type annotations

        # Extract entities and relationships from new documents
        new_entities_dict, new_relationships = await self._process_documents_for_entities(
            document_objects, auth, document_service, prompt_overrides
        )

        # Track document IDs that need to be included even without entities/relationships
        additional_doc_ids = {doc.external_id for doc in document_objects}

        # Merge new entities and relationships with existing ones
        existing_graph = self._merge_graph_data(
            existing_graph,
            new_entities_dict,
            new_relationships,
            all_doc_ids,
            additional_filters,
            additional_doc_ids,
        )

        # NEW: mark graph as completed after processing
        existing_graph.system_metadata["status"] = "completed"

        # Store the updated graph in the database
        if not await self.db.update_graph(existing_graph):
            raise Exception("Failed to update graph")

        return existing_graph

    async def _get_new_document_ids(
        self,
        auth: AuthContext,
        existing_graph: Graph,
        additional_filters: Optional[Dict[str, Any]] = None,
        additional_documents: Optional[List[str]] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> Set[str]:
        """Get IDs of new documents to add to the graph."""
        # Initialize system_filters if None
        if system_filters is None:
            system_filters = {}
        # Initialize with explicitly specified documents, ensuring it's a set
        document_ids = set(additional_documents or [])

        logger.info(f"[GRAPH DEBUG] _get_new_document_ids called with:")
        logger.info(f"[GRAPH DEBUG]   additional_filters={additional_filters}")
        logger.info(f"[GRAPH DEBUG]   additional_documents={additional_documents}")
        logger.info(f"[GRAPH DEBUG]   system_filters={system_filters}")
        logger.info(f"[GRAPH DEBUG]   existing_graph.filters={existing_graph.filters}")

        # Process documents matching additional filters
        if additional_filters or system_filters:
            logger.info(f"[GRAPH DEBUG] Querying documents with filters={additional_filters}, system_filters={system_filters}")
            filtered_docs = await self.db.get_documents(auth, filters=additional_filters, system_filters=system_filters)
            filter_doc_ids = {doc.external_id for doc in filtered_docs}
            logger.info(f"[GRAPH DEBUG] Query returned {len(filtered_docs)} documents: {filter_doc_ids}")
            logger.info(f"Found {len(filter_doc_ids)} documents matching additional filters and system filters")
            document_ids.update(filter_doc_ids)

        # Process documents matching the original filters
        if existing_graph.filters:
            # Original filters shouldn't include system filters, as we're applying them separately
            filtered_docs = await self.db.get_documents(
                auth, filters=existing_graph.filters, system_filters=system_filters
            )
            orig_filter_doc_ids = {doc.external_id for doc in filtered_docs}
            logger.info(f"Found {len(orig_filter_doc_ids)} documents matching original filters and system filters")
            document_ids.update(orig_filter_doc_ids)

        # Get only the document IDs that are not already in the graph
        new_doc_ids = document_ids - set(existing_graph.document_ids)
        logger.info(f"Found {len(new_doc_ids)} new documents to add to graph '{existing_graph.name}'")
        return new_doc_ids

    def _merge_graph_data(
        self,
        existing_graph: Graph,
        new_entities_dict: Dict[str, Entity],
        new_relationships: List[Relationship],
        document_ids: Set[str],
        additional_filters: Optional[Dict[str, Any]] = None,
        additional_doc_ids: Optional[Set[str]] = None,
    ) -> Graph:
        """Merge new entities and relationships with existing graph data."""
        # Create a mapping of existing entities by label for merging
        existing_entities_dict = {entity.label: entity for entity in existing_graph.entities}

        # Merge entities
        merged_entities = self._merge_entities(existing_entities_dict, new_entities_dict)

        # Create a mapping of entity labels to IDs for new relationships
        entity_id_map = {entity.label: entity.id for entity in merged_entities.values()}

        # Merge relationships
        merged_relationships = self._merge_relationships(
            existing_graph.relationships, new_relationships, new_entities_dict, entity_id_map
        )

        # Update the graph
        existing_graph.entities = list(merged_entities.values())
        existing_graph.relationships = merged_relationships

        # Ensure we include all necessary document IDs:
        # 1. All document IDs from document_ids parameter
        # 2. All document IDs that have authorized documents (from additional_doc_ids)
        final_doc_ids = document_ids.copy()
        if additional_doc_ids:
            final_doc_ids.update(additional_doc_ids)

        logger.info(f"Final document count in graph: {len(final_doc_ids)}")
        existing_graph.document_ids = list(final_doc_ids)
        existing_graph.updated_at = datetime.now(timezone.utc)

        # Update filters if additional filters were provided
        if additional_filters and existing_graph.filters:
            # Smarter filter merging
            self._smart_merge_filters(existing_graph.filters, additional_filters)

        return existing_graph

    def _smart_merge_filters(self, existing_filters: Dict[str, Any], additional_filters: Dict[str, Any]):
        """Merge filters with more intelligence to handle different data types and filter values."""
        for key, value in additional_filters.items():
            # If the key doesn't exist in existing filters, just add it
            if key not in existing_filters:
                existing_filters[key] = value
                continue

            existing_value = existing_filters[key]

            # Handle list values - merge them
            if isinstance(existing_value, list) and isinstance(value, list):
                # Union the lists without duplicates
                existing_filters[key] = list(set(existing_value + value))
            # Handle dict values - recursively merge them
            elif isinstance(existing_value, dict) and isinstance(value, dict):
                # Recursive merge for nested dictionaries
                self._smart_merge_filters(existing_value, value)
            # Default to overwriting with the new value
            else:
                existing_filters[key] = value

    def _merge_entities(
        self, existing_entities: Dict[str, Entity], new_entities: Dict[str, Entity]
    ) -> Dict[str, Entity]:
        """Merge new entities with existing entities."""
        merged_entities = existing_entities.copy()

        for label, new_entity in new_entities.items():
            if label in merged_entities:
                # Entity exists, merge chunk sources and document IDs
                existing_entity = merged_entities[label]

                # Merge document IDs
                for doc_id in new_entity.document_ids:
                    if doc_id not in existing_entity.document_ids:
                        existing_entity.document_ids.append(doc_id)

                # Merge chunk sources
                for doc_id, chunk_numbers in new_entity.chunk_sources.items():
                    if doc_id not in existing_entity.chunk_sources:
                        existing_entity.chunk_sources[doc_id] = chunk_numbers
                    else:
                        for chunk_num in chunk_numbers:
                            if chunk_num not in existing_entity.chunk_sources[doc_id]:
                                existing_entity.chunk_sources[doc_id].append(chunk_num)
            else:
                # Add new entity
                merged_entities[label] = new_entity

        return merged_entities

    def _merge_relationships(
        self,
        existing_relationships: List[Relationship],
        new_relationships: List[Relationship],
        new_entities_dict: Dict[str, Entity],
        entity_id_map: Dict[str, str],
    ) -> List[Relationship]:
        """Merge new relationships with existing ones."""
        merged_relationships = list(existing_relationships)

        # Create reverse mappings for entity IDs to labels for efficient lookup
        entity_id_to_label = {entity.id: label for label, entity in new_entities_dict.items()}

        for rel in new_relationships:
            # Look up entity labels using the reverse mapping
            source_label = entity_id_to_label.get(rel.source_id)
            target_label = entity_id_to_label.get(rel.target_id)

            if source_label in entity_id_map and target_label in entity_id_map:
                # Update relationship to use existing entity IDs
                rel.source_id = entity_id_map[source_label]
                rel.target_id = entity_id_map[target_label]

                # Check if this relationship already exists
                is_duplicate = False
                for existing_rel in existing_relationships:
                    if (
                        existing_rel.source_id == rel.source_id
                        and existing_rel.target_id == rel.target_id
                        and existing_rel.type == rel.type
                    ):

                        # Found a duplicate, merge the chunk sources
                        is_duplicate = True
                        self._merge_relationship_sources(existing_rel, rel)
                        break

                if not is_duplicate:
                    merged_relationships.append(rel)

        return merged_relationships

    def _merge_relationship_sources(self, existing_rel: Relationship, new_rel: Relationship) -> None:
        """Merge chunk sources and document IDs from new relationship into existing one."""
        # Merge chunk sources
        for doc_id, chunk_numbers in new_rel.chunk_sources.items():
            if doc_id not in existing_rel.chunk_sources:
                existing_rel.chunk_sources[doc_id] = chunk_numbers
            else:
                for chunk_num in chunk_numbers:
                    if chunk_num not in existing_rel.chunk_sources[doc_id]:
                        existing_rel.chunk_sources[doc_id].append(chunk_num)

        # Merge document IDs
        for doc_id in new_rel.document_ids:
            if doc_id not in existing_rel.document_ids:
                existing_rel.document_ids.append(doc_id)

    async def create_graph(
        self,
        name: str,
        auth: AuthContext,
        document_service,  # Passed in to avoid circular import
        filters: Optional[Dict[str, Any]] = None,
        documents: Optional[List[str]] = None,
        prompt_overrides: Optional[GraphPromptOverrides] = None,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> Graph:
        """Create a graph from documents.

        This function processes documents matching filters or specific document IDs,
        extracts entities and relationships from document chunks, and saves them as a graph.

        Args:
            name: Name of the graph to create
            auth: Authentication context
            document_service: DocumentService instance for retrieving documents and chunks
            filters: Optional metadata filters to determine which documents to include
            documents: Optional list of specific document IDs to include
            prompt_overrides: Optional GraphPromptOverrides with customizations for prompts
            system_filters: Optional system metadata filters (e.g. folder_name, end_user_id)
            to determine which documents to include

        Returns:
            Graph: The created graph
        """
        # Initialize system_filters if None
        if system_filters is None:
            system_filters = {}

        if "write" not in auth.permissions:
            raise PermissionError("User does not have write permission")

        # Find documents to process based on filters and/or specific document IDs
        document_ids = set(documents or [])

        # If filters were provided, get matching documents
        if filters or system_filters:
            logger.info(f"[GRAPH DEBUG] create_graph: Querying documents with filters={filters}, system_filters={system_filters}")
            filtered_docs = await self.db.get_documents(auth, filters=filters, system_filters=system_filters)
            logger.info(f"[GRAPH DEBUG] create_graph: Query returned {len(filtered_docs)} documents")
            document_ids.update(doc.external_id for doc in filtered_docs)

        logger.info(f"[GRAPH DEBUG] create_graph: Total document_ids to process: {document_ids}")

        if not document_ids:
            raise ValueError("No documents found matching criteria")

        # Convert system_filters for document retrieval
        folder_name = system_filters.get("folder_name") if system_filters else None
        end_user_id = system_filters.get("end_user_id") if system_filters else None

        # Batch retrieve documents for authorization check
        document_objects = await document_service.batch_retrieve_documents(
            list(document_ids), auth, folder_name, end_user_id
        )

        # Log for debugging
        logger.info(f"[GRAPH DEBUG] Graph creation with folder_name={folder_name}, end_user_id={end_user_id}")
        logger.info(f"[GRAPH DEBUG] Documents retrieved: {len(document_objects)} out of {len(document_ids)} requested")
        logger.info(f"Graph creation with folder_name={folder_name}, end_user_id={end_user_id}")
        logger.info(f"Documents retrieved: {len(document_objects)} out of {len(document_ids)} requested")
        if not document_objects:
            raise ValueError("No authorized documents found matching criteria")

        # Validation is now handled by type annotations

        # Create a new graph
        graph = Graph(
            name=name,
            document_ids=[doc.external_id for doc in document_objects],
            filters=filters,
            folder_name=system_filters.get("folder_name") if system_filters else None,
            end_user_id=system_filters.get("end_user_id") if system_filters else None,
            app_id=auth.app_id,
        )

        # Extract entities and relationships
        entities, relationships = await self._process_documents_for_entities(
            document_objects, auth, document_service, prompt_overrides
        )

        # Add entities and relationships to the graph
        graph.entities = list(entities.values())
        graph.relationships = relationships

        # NEW: Mark completion status
        graph.system_metadata["status"] = "completed"

        # Store the graph in the database
        if not await self.db.store_graph(graph, auth):
            raise Exception("Failed to store graph")

        return graph

    async def _process_documents_for_entities(
        self,
        documents: List[Document],
        auth: AuthContext,
        document_service,
        prompt_overrides: Optional[GraphPromptOverrides] = None,
    ) -> Tuple[Dict[str, Entity], List[Relationship]]:
        """Process documents to extract entities and relationships.

        Args:
            documents: List of documents to process
            auth: Authentication context
            document_service: DocumentService instance for retrieving chunks
            prompt_overrides: Optional dictionary with customizations for prompts
                {
                    "entity_resolution": {
                        "prompt_template": "Custom template...",
                        "examples": [{"canonical": "...", "variants": [...]}]
                    }
                }

        Returns:
            Tuple of (entities_dict, relationships_list)
        """
        # Dictionary to collect entities by label (to avoid duplicates)
        entities = {}
        # List to collect all relationships
        relationships = []
        # List to collect all extracted entities for resolution
        all_entities = []
        # Track all initial entities with their labels to fix relationship mapping
        initial_entities = []

        # Collect all chunk sources from documents.
        chunk_sources = [
            ChunkSource(document_id=doc.external_id, chunk_number=i)
            for doc in documents
            for i, _ in enumerate(doc.chunk_ids)
        ]

        logger.info(f"[GRAPH DEBUG] Processing {len(documents)} documents for entity extraction")
        logger.info(f"[GRAPH DEBUG] Document IDs: {[doc.external_id for doc in documents]}")
        logger.info(f"[GRAPH DEBUG] Total chunk sources to retrieve: {len(chunk_sources)}")

        # Batch retrieve chunks
        chunks = await document_service.batch_retrieve_chunks(chunk_sources, auth)
        logger.info(f"[GRAPH DEBUG] Retrieved {len(chunks)} chunks for processing")
        logger.info(f"Retrieved {len(chunks)} chunks for processing")

        # Process each chunk individually
        for chunk in chunks:
            try:
                # Get entity_extraction override if provided
                extraction_overrides = None
                if prompt_overrides:
                    # Get entity_extraction from the model
                    extraction_overrides = prompt_overrides.entity_extraction

                # Extract entities and relationships from the chunk
                chunk_entities, chunk_relationships = await self.extract_entities_from_text(
                    chunk.content, chunk.document_id, chunk.chunk_number, extraction_overrides
                )

                # Store all initially extracted entities to track their IDs
                initial_entities.extend(chunk_entities)

                # Add entities to the collection, avoiding duplicates based on exact label match
                for entity in chunk_entities:
                    if entity.label not in entities:
                        # For new entities, initialize chunk_sources with the current chunk
                        entities[entity.label] = entity
                        all_entities.append(entity)
                    else:
                        # If entity already exists, add this chunk source if not already present
                        existing_entity = entities[entity.label]

                        # Add to chunk_sources dictionary
                        if chunk.document_id not in existing_entity.chunk_sources:
                            existing_entity.chunk_sources[chunk.document_id] = [chunk.chunk_number]
                        elif chunk.chunk_number not in existing_entity.chunk_sources[chunk.document_id]:
                            existing_entity.chunk_sources[chunk.document_id].append(chunk.chunk_number)

                # Add the current chunk source to each relationship
                for relationship in chunk_relationships:
                    # Add to chunk_sources dictionary
                    if chunk.document_id not in relationship.chunk_sources:
                        relationship.chunk_sources[chunk.document_id] = [chunk.chunk_number]
                    elif chunk.chunk_number not in relationship.chunk_sources[chunk.document_id]:
                        relationship.chunk_sources[chunk.document_id].append(chunk.chunk_number)

                # Add relationships to the collection
                relationships.extend(chunk_relationships)

            except ValueError as e:
                # Handle specific extraction errors we've wrapped
                logger.warning(f"Skipping chunk {chunk.chunk_number} in document {chunk.document_id}: {e}")
                continue
            except Exception as e:
                # For other errors, log and re-raise to abort graph creation
                logger.error(f"Fatal error processing chunk {chunk.chunk_number} in document {chunk.document_id}: {e}")
                raise

        # Build a mapping from entity ID to label for ALL initially extracted entities
        original_entity_id_to_label = {entity.id: entity.label for entity in initial_entities}

        # Check if entity resolution is enabled in settings
        settings = get_settings()

        # Resolve entities to handle variations like "Trump" vs "Donald J Trump"
        if settings.ENABLE_ENTITY_RESOLUTION:
            logger.info("Resolving %d entities using LLM...", len(all_entities))

            # Extract entity_resolution part if this is a structured override
            resolution_overrides = None
            if prompt_overrides:
                if hasattr(prompt_overrides, "entity_resolution"):
                    # Get from Pydantic model
                    resolution_overrides = prompt_overrides.entity_resolution
                elif isinstance(prompt_overrides, dict) and "entity_resolution" in prompt_overrides:
                    # Get from dict
                    resolution_overrides = prompt_overrides["entity_resolution"]
                else:
                    # Otherwise pass as-is
                    resolution_overrides = prompt_overrides

            resolved_entities, entity_mapping = await self.entity_resolver.resolve_entities(
                all_entities, resolution_overrides
            )
            logger.info("Entity resolution completed successfully")
        else:
            logger.info("Entity resolution is disabled in settings.")
            # Return identity mapping (each entity maps to itself)
            entity_mapping = {entity.label: entity.label for entity in all_entities}
            resolved_entities = all_entities

        if entity_mapping:
            logger.info("Entity resolution complete. Found %d mappings.", len(entity_mapping))
            # Create a new entities dictionary with resolved entities
            resolved_entities_dict = {}
            # Build new entities dictionary with canonical labels
            for entity in resolved_entities:
                resolved_entities_dict[entity.label] = entity
            # Update relationships to use canonical entity labels
            updated_relationships = []

            # Remap relationships using original entity ID to label mapping
            remapped_count = 0
            skipped_count = 0

            for relationship in relationships:
                # Use original_entity_id_to_label to get the labels for relationship endpoints
                original_source_label = original_entity_id_to_label.get(relationship.source_id)
                original_target_label = original_entity_id_to_label.get(relationship.target_id)

                if not original_source_label or not original_target_label:
                    logger.warning(
                        f"Skipping relationship with type '{relationship.type}' - could not find original entity labels"
                    )
                    skipped_count += 1
                    continue

                # Find canonical labels using the mapping from the resolver
                source_canonical = entity_mapping.get(original_source_label, original_source_label)
                target_canonical = entity_mapping.get(original_target_label, original_target_label)

                # Find the final unique Entity objects using the canonical labels
                canonical_source = resolved_entities_dict.get(source_canonical)
                canonical_target = resolved_entities_dict.get(target_canonical)

                if canonical_source and canonical_target:
                    # Successfully found the final entities, update the relationship's IDs
                    relationship.source_id = canonical_source.id
                    relationship.target_id = canonical_target.id
                    updated_relationships.append(relationship)
                    remapped_count += 1
                else:
                    # Could not map to final entities, log and skip
                    logger.warning(
                        f"Skipping relationship between '{original_source_label}' and '{original_target_label}' - "
                        f"canonical entities not found after resolution"
                    )
                    skipped_count += 1

            logger.info(f"Remapped {remapped_count} relationships, skipped {skipped_count} relationships")

            # Deduplicate relationships (same source, target, type)
            final_relationships_map = {}
            for rel in updated_relationships:
                key = (rel.source_id, rel.target_id, rel.type)
                if key not in final_relationships_map:
                    final_relationships_map[key] = rel
                else:
                    # Merge sources into the existing relationship
                    existing_rel = final_relationships_map[key]
                    self._merge_relationship_sources(existing_rel, rel)

            final_relationships = list(final_relationships_map.values())
            logger.info(f"Deduplicated to {len(final_relationships)} unique relationships")

            return resolved_entities_dict, final_relationships

        # If no entity resolution occurred, return original entities and relationships
        return entities, relationships

    async def extract_entities_from_text(
        self,
        content: str,
        doc_id: str,
        chunk_number: int,
        prompt_overrides: Optional[EntityExtractionPromptOverride] = None,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text content using the LLM.

        Args:
            content: Text content to process
            doc_id: Document ID
            chunk_number: Chunk number within the document

        Returns:
            Tuple of (entities, relationships)
        """
        settings = get_settings()

        # Limit text length to avoid token limits
        content_limited = content[: min(len(content), 5000)]

        # We'll use the Pydantic model directly when calling litellm
        # No need to generate JSON schema separately

        # Get entity extraction overrides if available
        extraction_overrides = {}

        # Convert prompt_overrides to dict for processing
        if prompt_overrides:
            # If it's already an EntityExtractionPromptOverride, convert to dict
            extraction_overrides = prompt_overrides.model_dump(exclude_none=True)

        # Check for custom prompt template
        custom_prompt = extraction_overrides.get("prompt_template")
        custom_examples = extraction_overrides.get("examples")

        # Prepare examples if provided
        examples_str = ""
        if custom_examples:
            # Ensure proper serialization for both dict and Pydantic model examples
            if isinstance(custom_examples, list) and custom_examples and hasattr(custom_examples[0], "model_dump"):
                # List of Pydantic model objects
                serialized_examples = [example.model_dump() for example in custom_examples]
            else:
                # List of dictionaries
                serialized_examples = custom_examples

            examples_json = {"entities": serialized_examples}
            examples_str = (
                f"\nHere are some examples of the kind of entities to extract:\n```json\n"
                f"{json.dumps(examples_json, indent=2)}\n```\n"
            )

        # Modify the system message to handle properties as a string that will be parsed later
        system_message = {
            "role": "system",
            "content": (
                "You are an entity extraction and relationship extraction assistant. Extract entities and "
                "their relationships from text precisely and thoroughly, extract as many entities and "
                "relationships as possible. "
                "For entities, include entity label and type (some examples: PERSON, ORGANIZATION, LOCATION, "
                "CONCEPT, etc.). If the user has given examples, use those, these are just suggestions"
                "For relationships, use a simple format with source, target, and relationship fields. "
                "Be very through, there are many relationships that are not obvious"
                "IMPORTANT: The source and target fields must be simple strings representing "
                "entity labels. For example: "
                "if you extract entities 'Entity A' and 'Entity B', a relationship would have source: 'Entity A', "
                "target: 'Entity B', relationship: 'relates to'. "
                "Respond directly in json format, without any additional text or explanations. "
            ),
        }

        # Use custom prompt if provided, otherwise use default
        if custom_prompt:
            user_message = {
                "role": "user",
                "content": custom_prompt.format(content=content_limited, examples=examples_str),
            }
        else:
            user_message = {
                "role": "user",
                "content": (
                    "Extract named entities and their relationships from the following text. "
                    "For entities, include entity label and type (PERSON, ORGANIZATION, LOCATION, CONCEPT, etc.). "
                    "For relationships, specify the source entity, target entity, and the relationship between them. "
                    "The source and target must be simple strings matching the entity labels, not objects. "
                    f"{examples_str}"
                    'Sample relationship format: {"source": "Entity A", "target": "Entity B", '
                    '"relationship": "works for"}\n\n'
                    "Return your response as valid JSON:\n\n" + content_limited
                ),
            }

        # Get the model configuration from registered_models
        model_config = settings.REGISTERED_MODELS.get(settings.GRAPH_MODEL, {})
        if not model_config:
            raise ValueError(f"Model '{settings.GRAPH_MODEL}' not found in registered_models configuration")

        # Prepare the completion request parameters
        model_params = {
            "model": model_config.get("model_name"),
            "messages": [system_message, user_message],
            "response_format": ExtractionResult,
        }

        # Add all model-specific parameters from the config
        for key, value in model_config.items():
            if key != "model_name":  # Skip as we've already handled it
                model_params[key] = value
        import instructor
        import litellm

        # Use instructor with litellm to get structured responses
        client = instructor.from_litellm(litellm.acompletion, mode=instructor.Mode.JSON)
        try:
            # Use LiteLLM with instructor for structured completion
            logger.info(f"[GRAPH DEBUG] Extracting entities from chunk {chunk_number} in doc {doc_id}")
            logger.info(f"[GRAPH DEBUG] Using model: {model_config.get('model_name')}")
            logger.info(f"[GRAPH DEBUG] Content preview (first 200 chars): {content_limited[:200]}")
            logger.debug(f"Calling LiteLLM with instructor and params: {model_params}")
            # Extract the model and messages from model_params
            model = model_params.pop("model")
            messages = model_params.pop("messages")
            # Use instructor's chat.completions.create with response_model
            response = await client.chat.completions.create(
                model=model, messages=messages, response_model=ExtractionResult, **model_params
            )

            try:

                logger.info(f"[GRAPH DEBUG] Extraction completed successfully")
                logger.info(f"Extraction result type: {type(response)}")
                extraction_result = response  # The response is already our Pydantic model

                # Make sure the extraction_result has the expected properties
                if not hasattr(extraction_result, "entities"):
                    extraction_result.entities = []
                if not hasattr(extraction_result, "relationships"):
                    extraction_result.relationships = []

            except AttributeError as e:
                logger.error(f"Invalid response format from LiteLLM: {e}")
                logger.debug(f"Raw response structure: {response.choices[0]}")
                return [], []

        except Exception as e:
            logger.error(f"Error during entity extraction with LiteLLM: {str(e)}")
            # Enable this for more verbose debugging
            # litellm.set_verbose = True
            return [], []

        # Process extraction results
        entities, relationships = self._process_extraction_results(extraction_result, doc_id, chunk_number)
        logger.info(
            f"Extracted {len(entities)} entities and {len(relationships)} relationships from document "
            f"{doc_id}, chunk {chunk_number}"
        )
        return entities, relationships

    def _process_extraction_results(
        self, extraction_result: ExtractionResult, doc_id: str, chunk_number: int
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Process extraction results into entity and relationship objects."""
        # Initialize chunk_sources with the current chunk - reused across entities
        chunk_sources = {doc_id: [chunk_number]}

        # Convert extracted data to entity objects using list comprehension
        entities = [
            Entity(
                label=entity.label,
                type=entity.type,
                properties=entity.properties,
                chunk_sources=chunk_sources.copy(),  # Need to copy to avoid shared reference
                document_ids=[doc_id],
            )
            for entity in extraction_result.entities
        ]

        # Create a mapping of entity labels to IDs
        entity_mapping = {entity.label: entity.id for entity in entities}

        # Convert to relationship objects using list comprehension with filtering
        relationships = [
            Relationship(
                source_id=entity_mapping[rel.source],
                target_id=entity_mapping[rel.target],
                type=rel.relationship,
                chunk_sources=chunk_sources.copy(),  # Need to copy to avoid shared reference
                document_ids=[doc_id],
            )
            for rel in extraction_result.relationships
            if rel.source in entity_mapping and rel.target in entity_mapping
        ]

        return entities, relationships

    async def query_with_graph(
        self,
        query: str,
        graph_name: str,
        auth: AuthContext,
        document_service,  # Passed to avoid circular import
        filters: Optional[Dict[str, Any]] = None,
        k: int = 20,
        min_score: float = 0.0,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_reranking: Optional[bool] = None,
        use_colpali: Optional[bool] = None,
        hop_depth: int = 1,
        include_paths: bool = False,
        prompt_overrides: Optional[QueryPromptOverrides] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
        perf_tracker: Optional[Any] = None,
        stream_response: Optional[bool] = False,
        llm_config: Optional[Dict[str, Any]] = None,
        padding: Optional[int] = None,
        inline_citations: Optional[bool] = True,
    ) -> CompletionResponse:
        """Generate completion using knowledge graph-enhanced retrieval.

        This method enhances retrieval by:
        1. Extracting entities from the query
        2. Finding similar entities in the graph
        3. Traversing the graph to find related entities
        4. Retrieving chunks containing these entities
        5. Combining with traditional vector search results
        6. Generating a completion with enhanced context

        Args:
            query: The query text
            graph_name: Name of the graph to use
            auth: Authentication context
            document_service: DocumentService instance for retrieving documents
            filters: Optional metadata filters
            k: Number of chunks to retrieve
            min_score: Minimum similarity score
            max_tokens: Maximum tokens for completion
            temperature: Temperature for completion
            use_reranking: Whether to use reranking
            use_colpali: Whether to use colpali embedding
            hop_depth: Number of relationship hops to traverse (1-3)
            include_paths: Whether to include relationship paths in response
            prompt_overrides: Optional QueryPromptOverrides with customizations for prompts
            folder_name: Optional folder name for scoping
            end_user_id: Optional end user ID for scoping
        """
        logger.info(f"Querying with graph: {graph_name}, hop depth: {hop_depth}")

        # Validation is now handled by type annotations

        # Build system filters for scoping
        system_filters = {}
        if folder_name:
            system_filters["folder_name"] = folder_name
        if end_user_id:
            system_filters["end_user_id"] = end_user_id

        logger.info(f"Querying graph with system_filters: {system_filters}")
        graph = await self.db.get_graph(graph_name, auth, system_filters=system_filters)
        if not graph:
            logger.warning(f"Graph '{graph_name}' not found or not accessible")
            # Fall back to standard retrieval if graph not found
            return await document_service.query(
                query=query,
                auth=auth,
                filters=filters,
                k=k,
                min_score=min_score,
                max_tokens=max_tokens,
                temperature=temperature,
                use_reranking=use_reranking,
                use_colpali=use_colpali,
                graph_name=None,
                folder_name=folder_name,
                end_user_id=end_user_id,
            )

        # Parallel approach
        # 1. Standard vector search
        vector_chunks = await document_service.retrieve_chunks(
            query, auth, filters, k, min_score, use_reranking, use_colpali, folder_name, end_user_id
        )
        logger.info(f"Vector search retrieved {len(vector_chunks)} chunks")

        # 2. Graph-based retrieval
        # First extract entities from the query
        query_entities = await self._extract_entities_from_query(query, prompt_overrides)
        logger.info(
            f"Extracted {len(query_entities)} entities from query: {', '.join(e.label for e in query_entities)}"
        )

        # If no entities extracted, fallback to embedding similarity
        if not query_entities:
            # Find similar entities using embedding similarity
            top_entities = await self._find_similar_entities(query, graph.entities, k)
        else:
            # Use entity resolution to handle variants of the same entity
            settings = get_settings()

            # First, create combined list of query entities and graph entities for resolution
            combined_entities = query_entities + graph.entities

            # Resolve entities to identify variants if enabled
            if settings.ENABLE_ENTITY_RESOLUTION:
                logger.info(f"Resolving {len(combined_entities)} entities from query and graph...")
                # Get the entity_resolution override if provided
                resolution_overrides = None
                if prompt_overrides:
                    # Get just the entity_resolution part
                    resolution_overrides = prompt_overrides.entity_resolution

                resolved_entities, entity_mapping = await self.entity_resolver.resolve_entities(
                    combined_entities, prompt_overrides=resolution_overrides
                )
            else:
                logger.info("Entity resolution is disabled in settings.")
                # Return identity mapping (each entity maps to itself)
                entity_mapping = {entity.label: entity.label for entity in combined_entities}

            # Create a mapping of resolved entity labels to graph entities
            entity_map = {}
            for entity in graph.entities:
                # Get canonical form for this entity
                canonical_label = entity_mapping.get(entity.label, entity.label)
                entity_map[canonical_label.lower()] = entity

            matched_entities = []
            # Match extracted entities with graph entities using canonical labels
            for query_entity in query_entities:
                # Get canonical form for this query entity
                canonical_query = entity_mapping.get(query_entity.label, query_entity.label)
                if canonical_query.lower() in entity_map:
                    matched_entities.append(entity_map[canonical_query.lower()])

            # If no matches, fallback to embedding similarity
            if matched_entities:
                top_entities = [(entity, 1.0) for entity in matched_entities]  # Score 1.0 for direct matches
            else:
                top_entities = await self._find_similar_entities(query, graph.entities, k)

        logger.info(f"Found {len(top_entities)} relevant entities in graph")

        # Traverse the graph to find related entities
        expanded_entities = self._expand_entities(graph, [e[0] for e in top_entities], hop_depth)
        logger.info(f"Expanded to {len(expanded_entities)} entities after traversal")

        # Get specific chunks containing these entities
        graph_chunks = await self._retrieve_entity_chunks(
            expanded_entities, auth, filters, document_service, folder_name, end_user_id
        )
        logger.info(f"Retrieved {len(graph_chunks)} chunks containing relevant entities")

        # Calculate paths if requested
        paths = []
        if include_paths:
            paths = self._find_relationship_paths(graph, [e[0] for e in top_entities], hop_depth)
            logger.info(f"Found {len(paths)} relationship paths")

        # Combine vector and graph results
        combined_chunks = self._combine_chunk_results(vector_chunks, graph_chunks, k)

        # Generate completion with enhanced context
        completion_response = await self._generate_completion(
            query,
            combined_chunks,
            document_service,
            max_tokens,
            temperature,
            include_paths,
            paths,
            auth,
            graph_name,
            prompt_overrides,
            folder_name=folder_name,
            end_user_id=end_user_id,
        )

        return completion_response

    async def _extract_entities_from_query(
        self, query: str, prompt_overrides: Optional[QueryPromptOverrides] = None
    ) -> List[Entity]:
        """Extract entities from the query text using the LLM."""
        try:
            # Get entity_extraction override if provided
            extraction_overrides = None
            if prompt_overrides:
                # Get the entity_extraction part
                extraction_overrides = prompt_overrides.entity_extraction

            # Extract entities from the query using the same extraction function
            # but with a simplified prompt specific for queries
            entities, _ = await self.extract_entities_from_text(
                content=query,
                doc_id="query",  # Use "query" as doc_id
                chunk_number=0,  # Use 0 as chunk_number
                prompt_overrides=extraction_overrides,
            )
            return entities
        except Exception as e:
            # If extraction fails, log and return empty list to fall back to embedding similarity
            logger.warning(f"Failed to extract entities from query: {e}")
            return []

    async def _find_similar_entities(self, query: str, entities: List[Entity], k: int) -> List[Tuple[Entity, float]]:
        """Find entities similar to the query based on embedding similarity."""
        if not entities:
            return []

        # Get embedding for query
        query_embedding = await self.embedding_model.embed_for_query(query)

        # Create entity text representations and get embeddings for all entities
        entity_texts = [
            f"{entity.label} {entity.type} " + " ".join(f"{key}: {value}" for key, value in entity.properties.items())
            for entity in entities
        ]

        # Get embeddings for all entity texts
        entity_embeddings = await self._batch_get_embeddings(entity_texts)

        # Calculate similarities and pair with entities
        entity_similarities = [
            (entity, self._calculate_cosine_similarity(query_embedding, embedding))
            for entity, embedding in zip(entities, entity_embeddings)
        ]

        # Sort by similarity and take top k
        entity_similarities.sort(key=lambda x: x[1], reverse=True)
        return entity_similarities[: min(k, len(entity_similarities))]

    async def _batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts efficiently."""
        # This could be implemented with proper batch embedding if the embedding model supports it
        # For now, we'll just map over the texts and get embeddings one by one
        return [await self.embedding_model.embed_for_query(text) for text in texts]

    def _expand_entities(self, graph: Graph, seed_entities: List[Entity], hop_depth: int) -> List[Entity]:
        """Expand entities by traversing relationships."""
        if hop_depth <= 1:
            return seed_entities

        # Create a set of entity IDs we've seen
        seen_entity_ids = {entity.id for entity in seed_entities}
        all_entities = list(seed_entities)

        # Create a map for fast entity lookup
        entity_map = {entity.id: entity for entity in graph.entities}

        # For each hop
        for _ in range(hop_depth - 1):
            new_entities = []

            # For each entity we've found so far
            for entity in all_entities:
                # Find connected entities through relationships
                connected_ids = self._get_connected_entity_ids(graph.relationships, entity.id, seen_entity_ids)

                # Add new connected entities
                for entity_id in connected_ids:
                    if target_entity := entity_map.get(entity_id):
                        new_entities.append(target_entity)
                        seen_entity_ids.add(entity_id)

            # Add new entities to our list
            all_entities.extend(new_entities)

            # Stop if no new entities found
            if not new_entities:
                break

        return all_entities

    def _get_connected_entity_ids(
        self, relationships: List[Relationship], entity_id: str, seen_ids: Set[str]
    ) -> Set[str]:
        """Get IDs of entities connected to the given entity that haven't been seen yet."""
        connected_ids = set()

        for relationship in relationships:
            # Check outgoing relationships
            if relationship.source_id == entity_id and relationship.target_id not in seen_ids:
                connected_ids.add(relationship.target_id)

            # Check incoming relationships
            elif relationship.target_id == entity_id and relationship.source_id not in seen_ids:
                connected_ids.add(relationship.source_id)

        return connected_ids

    async def _retrieve_entity_chunks(
        self,
        entities: List[Entity],
        auth: AuthContext,
        filters: Optional[Dict[str, Any]],
        document_service,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
    ) -> List[ChunkResult]:
        """Retrieve chunks containing the specified entities."""
        # Initialize filters if None
        if filters is None:
            filters = {}
        if not entities:
            return []

        # Collect all chunk sources from entities using set comprehension
        entity_chunk_sources = {
            (doc_id, chunk_num)
            for entity in entities
            for doc_id, chunk_numbers in entity.chunk_sources.items()
            for chunk_num in chunk_numbers
        }

        # Get unique document IDs for authorization check
        doc_ids = {doc_id for doc_id, _ in entity_chunk_sources}

        # Check document authorization with system filters
        documents = await document_service.batch_retrieve_documents(list(doc_ids), auth, folder_name, end_user_id)

        # Apply filters if needed
        authorized_doc_ids = {
            doc.external_id
            for doc in documents
            if not filters or all(doc.metadata.get(k) == v for k, v in filters.items())
        }

        # Filter chunk sources to only those from authorized documents
        chunk_sources = [
            ChunkSource(document_id=doc_id, chunk_number=chunk_num)
            for doc_id, chunk_num in entity_chunk_sources
            if doc_id in authorized_doc_ids
        ]

        # Retrieve and return chunks if we have any valid sources
        return (
            await document_service.batch_retrieve_chunks(
                chunk_sources, auth, folder_name=folder_name, end_user_id=end_user_id
            )
            if chunk_sources
            else []
        )

    def _combine_chunk_results(
        self, vector_chunks: List[ChunkResult], graph_chunks: List[ChunkResult], k: int
    ) -> List[ChunkResult]:
        """Combine and deduplicate chunk results from vector search and graph search.

        Returns all merged chunks sorted by score - no truncation to k.
        The token limit in the completion step will naturally control context size.
        This ensures important vector search results aren't filtered out by graph chunks.
        """
        # Create dictionary with vector chunks first
        all_chunks = {f"{chunk.document_id}_{chunk.chunk_number}": chunk for chunk in vector_chunks}

        # Process and add graph chunks with a boost
        for chunk in graph_chunks:
            chunk_key = f"{chunk.document_id}_{chunk.chunk_number}"

            # Set default score if missing and apply boost (5%)
            chunk.score = min(1.0, (getattr(chunk, "score", 0.7) or 0.7) * 1.05)

            # Keep the higher-scored version
            if chunk_key not in all_chunks or chunk.score > (getattr(all_chunks.get(chunk_key), "score", 0) or 0):
                all_chunks[chunk_key] = chunk

        # Convert to list and sort by score - return ALL chunks (no truncation to k)
        # Let the token limit control context size instead of artificially limiting here
        return sorted(all_chunks.values(), key=lambda x: getattr(x, "score", 0), reverse=True)

    def _find_relationship_paths(self, graph: Graph, seed_entities: List[Entity], hop_depth: int) -> List[List[str]]:
        """Find meaningful paths in the graph starting from seed entities."""
        paths = []
        entity_map = {entity.id: entity for entity in graph.entities}

        # For each seed entity
        for start_entity in seed_entities:
            # Start BFS from this entity
            queue = [(start_entity.id, [start_entity.label])]
            visited = set([start_entity.id])

            while queue:
                entity_id, path = queue.pop(0)

                # If path is already at max length, record it but don't expand
                if len(path) >= hop_depth * 2:  # *2 because path includes relationship types
                    paths.append(path)
                    continue

                # Find connected relationships
                for relationship in graph.relationships:
                    # Process both outgoing and incoming relationships
                    if relationship.source_id == entity_id:
                        target_id = relationship.target_id
                        if target_id in visited:
                            continue

                        target_entity = entity_map.get(target_id)
                        if not target_entity:
                            continue

                        # Check for common chunks
                        common_chunks = self._find_common_chunks(entity_map[entity_id], target_entity, relationship)

                        # Only include relationships where entities co-occur
                        if common_chunks:
                            visited.add(target_id)
                            # Create path with relationship info
                            rel_context = f"({relationship.type}, {len(common_chunks)} shared chunks)"
                            new_path = path + [rel_context, target_entity.label]
                            queue.append((target_id, new_path))
                            paths.append(new_path)

                    elif relationship.target_id == entity_id:
                        source_id = relationship.source_id
                        if source_id in visited:
                            continue

                        source_entity = entity_map.get(source_id)
                        if not source_entity:
                            continue

                        # Check for common chunks
                        common_chunks = self._find_common_chunks(entity_map[entity_id], source_entity, relationship)

                        # Only include relationships where entities co-occur
                        if common_chunks:
                            visited.add(source_id)
                            # Create path with relationship info (note reverse direction)
                            rel_context = f"(is {relationship.type} of, {len(common_chunks)} shared chunks)"
                            new_path = path + [rel_context, source_entity.label]
                            queue.append((source_id, new_path))
                            paths.append(new_path)

        return paths

    def _find_common_chunks(self, entity1: Entity, entity2: Entity, relationship: Relationship) -> Set[Tuple[str, int]]:
        """Find chunks that contain both entities and their relationship."""
        # Get chunk locations for each element
        entity1_chunks = set()
        for doc_id, chunk_numbers in entity1.chunk_sources.items():
            for chunk_num in chunk_numbers:
                entity1_chunks.add((doc_id, chunk_num))

        entity2_chunks = set()
        for doc_id, chunk_numbers in entity2.chunk_sources.items():
            for chunk_num in chunk_numbers:
                entity2_chunks.add((doc_id, chunk_num))

        rel_chunks = set()
        for doc_id, chunk_numbers in relationship.chunk_sources.items():
            for chunk_num in chunk_numbers:
                rel_chunks.add((doc_id, chunk_num))

        # Return intersection
        return entity1_chunks.intersection(entity2_chunks).intersection(rel_chunks)

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        # Convert to numpy arrays and calculate in one go
        vec1_np, vec2_np = np.array(vec1), np.array(vec2)

        # Get magnitudes
        magnitude1, magnitude2 = np.linalg.norm(vec1_np), np.linalg.norm(vec2_np)

        # Avoid division by zero and calculate similarity
        return 0 if magnitude1 == 0 or magnitude2 == 0 else np.dot(vec1_np, vec2_np) / (magnitude1 * magnitude2)

    async def _generate_completion(
        self,
        query: str,
        chunks: List[ChunkResult],
        document_service,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        include_paths: bool = False,
        paths: Optional[List[List[str]]] = None,
        auth: Optional[AuthContext] = None,
        graph_name: Optional[str] = None,
        prompt_overrides: Optional[QueryPromptOverrides] = None,
        folder_name: Optional[Union[str, List[str]]] = None,
        end_user_id: Optional[str] = None,
    ) -> CompletionResponse:
        """Generate completion using the retrieved chunks and optional path information."""
        if not chunks:
            chunks = []  # Ensure chunks is a list even if empty

        # Create document results for context augmentation
        documents = await document_service._create_document_results(auth, chunks)

        # Create augmented chunk contents
        chunk_contents = [
            chunk.augmented_content(documents[chunk.document_id]) for chunk in chunks if chunk.document_id in documents
        ]

        # Include graph context in prompt if paths are requested
        if include_paths and paths:
            # Create a readable representation of the paths
            paths_text = "Knowledge Graph Context:\n"
            # Limit to 5 paths to avoid token limits
            for path in paths[:5]:
                paths_text += " -> ".join(path) + "\n"

            # Add to the first chunk or create a new first chunk if none
            if chunk_contents:
                chunk_contents[0] = paths_text + "\n\n" + chunk_contents[0]
            else:
                chunk_contents = [paths_text]

        # Generate completion with prompt override if provided
        custom_prompt_template = None
        if prompt_overrides and prompt_overrides.query:
            custom_prompt_template = prompt_overrides.query.prompt_template

        request = CompletionRequest(
            query=query,
            context_chunks=chunk_contents,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_template=custom_prompt_template,
            folder_name=folder_name,
            end_user_id=end_user_id,
        )

        # Get completion from model
        response = await document_service.completion_model.complete(request)

        # Add sources information
        response.sources = [
            ChunkSource(
                document_id=chunk.document_id,
                chunk_number=chunk.chunk_number,
                score=getattr(chunk, "score", 0),
            )
            for chunk in chunks
        ]

        # Include graph metadata if paths were requested
        if include_paths:
            # Initialize metadata if it doesn't exist
            if not hasattr(response, "metadata") or response.metadata is None:
                response.metadata = {}

            # Extract unique entities from paths (items that don't start with "(")
            unique_entities = set()
            if paths:
                for path in paths[:5]:
                    for item in path:
                        if not item.startswith("("):
                            unique_entities.add(item)

            # Add graph-specific metadata
            response.metadata["graph"] = {
                "name": graph_name,
                "relevant_entities": list(unique_entities),
                "paths": [" -> ".join(path) for path in paths[:5]] if paths else [],
            }

        return response

    async def get_graph_visualization_data(
        self,
        graph_name: str,
        auth: AuthContext,
        system_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get graph visualization data for local graphs.

        Args:
            graph_name: Name of the graph to visualize
            auth: Authentication context
            system_filters: Optional system filters for graph retrieval

        Returns:
            Dict containing nodes and links for visualization
        """
        # Initialize system_filters if None
        if system_filters is None:
            system_filters = {}

        graph = await self.db.get_graph(graph_name, auth, system_filters=system_filters)
        if not graph:
            logger.warning(f"Graph '{graph_name}' not found or not accessible")
            return {"nodes": [], "links": []}

        # Transform entities to nodes format
        nodes = []
        for entity in graph.entities:
            nodes.append(
                {
                    "id": entity.id,
                    "label": entity.label,
                    "type": entity.type,
                    "properties": entity.properties,
                    "color": self._get_node_color(entity.type),
                }
            )

        # Transform relationships to links format
        links = []
        entity_id_set = {entity.id for entity in graph.entities}
        for relationship in graph.relationships:
            # Only include relationships where both source and target exist
            if relationship.source_id in entity_id_set and relationship.target_id in entity_id_set:
                links.append(
                    {"source": relationship.source_id, "target": relationship.target_id, "type": relationship.type}
                )

        return {"nodes": nodes, "links": links}

    def _get_node_color(self, node_type: str) -> str:
        """Get color for a node type to match the UI color scheme."""
        color_map = {
            "person": "#4f46e5",  # Indigo
            "organization": "#06b6d4",  # Cyan
            "location": "#10b981",  # Emerald
            "date": "#f59e0b",  # Amber
            "concept": "#8b5cf6",  # Violet
            "event": "#ec4899",  # Pink
            "product": "#ef4444",  # Red
            "entity": "#4f46e5",  # Indigo (for generic entities)
            "attribute": "#f59e0b",  # Amber
            "relationship": "#ec4899",  # Pink
            "high_level_element": "#10b981",  # Emerald
            "semantic_unit": "#8b5cf6",  # Violet
        }
        return color_map.get(node_type.lower(), "#6b7280")  # Gray as default
