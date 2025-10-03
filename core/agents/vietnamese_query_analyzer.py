"""Vietnamese Contract Query Analyzer with Intent Classification.

Based on proven pdf-qa-system architecture:
1. Intent Classification: Fast detection of query intent
2. Pattern Analysis: Regex + keyword matching (no LLM)
3. Semantic Analysis: LLM provides deeper understanding (optional)
4. Routing Logic: Python deterministic routing

This pre-analysis step solves the problem where "thiết bị đắt nhất" (equipment prices)
was being mapped to total contract value queries.
"""

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

logger = logging.getLogger(__name__)


class QueryIntent(StrEnum):
    """Vietnamese contract query intent categories."""

    EQUIPMENT_PRICE_LOOKUP = "equipment_price_lookup"
    TOTAL_VALUE_LOOKUP = "total_value_lookup"
    PAYMENT_TERMS_LOOKUP = "payment_terms_lookup"
    CONTRACT_PARTIES_LOOKUP = "contract_parties_lookup"
    CONTRACT_DATES_LOOKUP = "contract_dates_lookup"
    CONTRACT_COMPARISON = "contract_comparison"
    CONTRACT_ANALYSIS = "contract_analysis"
    UNKNOWN = "unknown"


class QueryScope(StrEnum):
    """Query scope classification."""

    SINGLE_DOCUMENT = "single_document"
    MULTI_DOCUMENT = "multi_document"
    PORTFOLIO_WIDE = "portfolio_wide"
    UNKNOWN_SCOPE = "unknown_scope"


class QueryComplexity(StrEnum):
    """Query complexity classification."""

    SIMPLE_LOOKUP = "simple_lookup"
    MODERATE_ANALYSIS = "moderate_analysis"
    COMPLEX_SYNTHESIS = "complex_synthesis"


class QuerySemanticAnalysis(BaseModel):
    """LLM semantic analysis output - NO routing decisions."""

    scope_indicators: List[str] = Field(description="Semantic scope indicators found")
    complexity_signals: List[str] = Field(description="Complexity signals identified")
    semantic_complexity_score: float = Field(ge=0.0, le=1.0, description="Semantic complexity")
    detected_intent_hints: List[str] = Field(description="Intent hints from semantic analysis")
    reasoning: str = Field(description="Semantic analysis reasoning")


class QueryAnalysisResult(BaseModel):
    """Final query analysis with routing information."""

    intent: QueryIntent = Field(description="Detected query intent")
    scope: QueryScope = Field(description="Query scope")
    complexity: QueryComplexity = Field(description="Query complexity")
    complexity_score: float = Field(ge=0.0, le=1.0, description="Final complexity score")
    instruction_context: str = Field(description="Abstract instructions for the LLM on how to answer based on intent")
    folder_name: str = Field(description="Folder to search in")
    k: int = Field(description="Number of chunks to retrieve")
    min_relevance: float = Field(description="Minimum relevance score")
    reasoning: str = Field(description="Analysis reasoning")
    method: str = Field(description="How intent was detected: pattern_match, semantic_analysis, or hybrid")


@dataclass
class VietnameseQueryAnalyzer:
    """Analyzer for Vietnamese contract queries with intelligent intent detection."""

    patterns_path: Path
    enable_semantic_analysis: bool = True

    def __post_init__(self):
        """Load patterns and configure analyzer."""
        self.patterns = self._load_patterns()
        self.routing_config = self.patterns.get("routing_defaults", {})
        self.intent_instructions = self.patterns.get("intent_instructions", {})
        self.intent_patterns = self.patterns.get("intent_patterns", {})
        self.complexity_config = self.patterns.get("complexity_scoring", {})

        # Create semantic analysis agent (qwen3:8b for fast analysis)
        if self.enable_semantic_analysis:
            model_config = self.patterns.get("models", {}).get("query_analyzer", {})
            self.semantic_agent = self._create_semantic_agent(model_config)
        else:
            self.semantic_agent = None

    def _load_patterns(self) -> Dict[str, Any]:
        """Load pattern configuration from YAML."""
        with open(self.patterns_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _create_semantic_agent(self, model_config: Dict[str, Any]) -> Agent:
        """Create PydanticAI agent for semantic analysis using vLLM."""
        vllm_client = httpx.AsyncClient(timeout=600.0)  # 10 minute timeout
        model = OpenAIChatModel(
            model_name='/models/gemma-3-12b-it',
            provider=OpenAIProvider(
                base_url='http://vllm:8080/v1',
                api_key='dummy',
                http_client=vllm_client
            )
        )

        system_prompt = """You are a Vietnamese contract query analysis expert.

Analyze the semantic content of Vietnamese contract queries.

Extract:
1. Scope signals (single document vs multi-document)
2. Complexity signals (simple lookup vs complex analysis)
3. Intent hints (what is the user really asking for?)

Focus on distinguishing between:
- Individual item/equipment queries ("thiết bị", "đơn giá", "từng")
- Total/aggregate queries ("tổng", "tổng cộng", "tổng giá trị")

Provide semantic complexity score (0.0=simple lookup, 1.0=complex synthesis)."""

        agent = Agent(
            model,
            output_type=QuerySemanticAnalysis,
            system_prompt=system_prompt,
            model_settings={
                "max_tokens": model_config.get("max_tokens", 4000),
                "temperature": model_config.get("temperature", 0.1),
                "timeout": model_config.get("timeout", 10),
            }
        )

        return agent

    async def analyze(self, query: str) -> QueryAnalysisResult:
        """Analyze query using 4-step hybrid approach.

        Step 1: Pattern-based intent detection (fast, no LLM)
        Step 2: Regex + keyword pattern analysis
        Step 3: Optional semantic analysis with LLM
        Step 4: Final routing and query mapping
        """
        # Step 1: Intent detection from patterns
        intent, pattern_confidence = self._detect_intent_from_patterns(query)

        # Step 2: Pattern analysis (scope, complexity indicators)
        pattern_hints = self._analyze_query_patterns(query)

        # Step 3: Semantic analysis if enabled and pattern confidence is low
        semantic_analysis = None
        if self.semantic_agent and (pattern_confidence < 0.8 or intent == QueryIntent.UNKNOWN):
            try:
                result = await self.semantic_agent.run(query)
                semantic_analysis = result.output
                logger.info(f"Semantic analysis: {semantic_analysis.reasoning}")
            except Exception as e:
                logger.warning(f"Semantic analysis failed: {e}")
                semantic_analysis = self._fallback_semantic_analysis()
        else:
            semantic_analysis = self._fallback_semantic_analysis()

        # Step 4: Final routing and query mapping
        return self._create_analysis_result(
            query=query,
            intent=intent,
            pattern_confidence=pattern_confidence,
            pattern_hints=pattern_hints,
            semantic_analysis=semantic_analysis
        )

    def _detect_intent_from_patterns(self, query: str) -> Tuple[QueryIntent, float]:
        """Detect intent using keyword and exclusion patterns.

        This is the critical fix: properly distinguish equipment prices from total value.
        """
        query_lower = query.lower()
        intent_scores: Dict[QueryIntent, float] = {}

        for intent_key, intent_config in self.intent_patterns.items():
            try:
                intent_enum = QueryIntent(intent_key)
            except ValueError:
                logger.warning(f"Unknown intent key: {intent_key}")
                continue

            # Check keywords
            keywords = intent_config.get("keywords", [])
            exclude_keywords = intent_config.get("exclude_keywords", [])

            if not keywords:
                continue

            # Count matching keywords
            matches = sum(1 for kw in keywords if kw in query_lower)

            # Check exclusions (critical for distinguishing equipment vs total)
            has_exclusions = any(excl in query_lower for excl in exclude_keywords)

            if has_exclusions:
                # Exclude this intent if exclusion keywords are present
                intent_scores[intent_enum] = 0.0
            elif matches > 0:
                # Calculate confidence based on match ratio
                confidence = min(1.0, matches / len(keywords))
                intent_scores[intent_enum] = confidence

        # Return highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            if best_intent[1] > 0.0:
                return best_intent[0], best_intent[1]

        # Default to unknown
        return QueryIntent.UNKNOWN, 0.0

    def _analyze_query_patterns(self, query: str) -> Dict[str, Any]:
        """Analyze query using regex and keyword patterns."""
        query_lower = query.lower()

        # Scope analysis
        scope_indicators = self.patterns.get("scope_indicators", {})
        has_single_doc = any(
            pattern in query_lower
            for pattern in scope_indicators.get("single_document", [])
        )
        has_multi_doc = any(
            pattern in query_lower
            for pattern in scope_indicators.get("multi_document", [])
        )
        has_portfolio = any(
            pattern in query_lower
            for pattern in scope_indicators.get("portfolio_wide", [])
        )

        # Regex pattern matching
        regex_patterns = self.patterns.get("regex_patterns", {})
        regex_hits = {}
        for pattern_name, pattern in regex_patterns.items():
            try:
                hits = re.findall(pattern, query)
                regex_hits[pattern_name] = len(hits) > 0
            except re.error:
                regex_hits[pattern_name] = False

        return {
            "has_single_doc": has_single_doc,
            "has_multi_doc": has_multi_doc,
            "has_portfolio": has_portfolio,
            "regex_hits": regex_hits,
            "has_structured_data": any(regex_hits.values()),
            "query_length": len(query),
        }

    def _create_analysis_result(
        self,
        query: str,
        intent: QueryIntent,
        pattern_confidence: float,
        pattern_hints: Dict[str, Any],
        semantic_analysis: QuerySemanticAnalysis
    ) -> QueryAnalysisResult:
        """Create final analysis result with routing information."""

        # Determine scope
        scope = self._determine_scope(pattern_hints)

        # Determine complexity
        complexity, complexity_score = self._determine_complexity(
            intent, pattern_confidence, pattern_hints, semantic_analysis
        )

        # Get instruction context for this intent
        instruction_context = self.intent_instructions.get(
            intent.value,
            "Extract relevant information from the contract to answer the user's query."
        )

        # Get routing defaults
        folder_name = self.routing_config.get("default_folder", "folder-contracts")
        k = self.routing_config.get("default_k", 5)
        min_relevance = self.routing_config.get("min_relevance", 0.7)

        # Apply intent-specific k overrides if available
        intent_k_overrides = self.routing_config.get("intent_k_overrides", {})
        if intent.value in intent_k_overrides:
            k = intent_k_overrides[intent.value]

        # Determine analysis method
        if pattern_confidence >= 0.8:
            method = "pattern_match"
        elif semantic_analysis and semantic_analysis.reasoning != "Fallback analysis - LLM unavailable":
            method = "hybrid"
        else:
            method = "semantic_analysis"

        reasoning = (
            f"Intent: {intent.value} (confidence: {pattern_confidence:.2f}) | "
            f"Scope: {scope.value} | "
            f"Complexity: {complexity.value} ({complexity_score:.2f}) | "
            f"Method: {method}"
        )

        return QueryAnalysisResult(
            intent=intent,
            scope=scope,
            complexity=complexity,
            complexity_score=complexity_score,
            instruction_context=instruction_context,
            folder_name=folder_name,
            k=k,
            min_relevance=min_relevance,
            reasoning=reasoning,
            method=method
        )

    def _determine_scope(self, pattern_hints: Dict[str, Any]) -> QueryScope:
        """Determine query scope using pattern matching."""
        if pattern_hints.get("has_portfolio"):
            return QueryScope.PORTFOLIO_WIDE
        elif pattern_hints.get("has_multi_doc"):
            return QueryScope.MULTI_DOCUMENT
        elif pattern_hints.get("has_single_doc"):
            return QueryScope.SINGLE_DOCUMENT
        else:
            return QueryScope.UNKNOWN_SCOPE

    def _determine_complexity(
        self,
        intent: QueryIntent,
        pattern_confidence: float,
        pattern_hints: Dict[str, Any],
        semantic_analysis: QuerySemanticAnalysis
    ) -> Tuple[QueryComplexity, float]:
        """Determine complexity using intent mapping + patterns + semantics."""

        # Get base score from intent mapping
        intent_mapping = self.patterns.get("intent_to_complexity_mapping", {})
        if intent.value in intent_mapping:
            complexity_name, base_score = intent_mapping[intent.value]
        else:
            complexity_name, base_score = "moderate_analysis", 0.5

        # Get scoring weights
        weights = self.complexity_config.get("weights", {})
        thresholds = self.complexity_config.get("thresholds", {"complex": 0.7, "moderate": 0.4})

        # Pattern adjustments
        pattern_adjustment = 0.0
        if pattern_hints.get("has_structured_data"):
            pattern_adjustment += weights.get("regex_hit_boost", 0.1)

        # Semantic analysis contribution
        semantic_weight = weights.get("semantic_multiplier", 0.3)
        semantic_score = semantic_analysis.semantic_complexity_score if semantic_analysis else 0.5

        # Final score calculation
        final_score = min(1.0, max(0.0,
            base_score + pattern_adjustment + (semantic_score * semantic_weight)
        ))

        # Determine complexity category
        if final_score >= thresholds.get("complex", 0.7):
            final_complexity = QueryComplexity.COMPLEX_SYNTHESIS
        elif final_score >= thresholds.get("moderate", 0.4):
            final_complexity = QueryComplexity.MODERATE_ANALYSIS
        else:
            final_complexity = QueryComplexity.SIMPLE_LOOKUP

        return final_complexity, final_score

    def _fallback_semantic_analysis(self) -> QuerySemanticAnalysis:
        """Fallback when semantic analysis is disabled or fails."""
        return QuerySemanticAnalysis(
            scope_indicators=[],
            complexity_signals=[],
            semantic_complexity_score=0.5,
            detected_intent_hints=[],
            reasoning="Fallback analysis - LLM unavailable"
        )


# Factory function for easy instantiation
def create_vietnamese_query_analyzer(
    patterns_path: Optional[Path] = None,
    enable_semantic_analysis: bool = True
) -> VietnameseQueryAnalyzer:
    """Create a Vietnamese query analyzer instance.

    Args:
        patterns_path: Path to vietnamese_patterns.yaml file
        enable_semantic_analysis: Whether to use LLM for semantic analysis

    Returns:
        Configured VietnameseQueryAnalyzer instance
    """
    if patterns_path is None:
        # Default to core/config/vietnamese_patterns.yaml
        patterns_path = Path(__file__).parent.parent / "config" / "vietnamese_patterns.yaml"

    if not patterns_path.exists():
        raise FileNotFoundError(f"Patterns file not found: {patterns_path}")

    return VietnameseQueryAnalyzer(
        patterns_path=patterns_path,
        enable_semantic_analysis=enable_semantic_analysis
    )