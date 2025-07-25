"""RAG-specific domain models and value objects."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    """Types of RAG queries."""

    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    EXTRACTION = "extraction"


class RetrievalStrategy(str, Enum):
    """Document retrieval strategies."""

    SIMILARITY = "similarity"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class GenerationStrategy(str, Enum):
    """Text generation strategies."""

    STANDARD = "standard"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STEP_BY_STEP = "step_by_step"
    CREATIVE = "creative"


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""

    id: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None
    chunk_index: int = 0
    parent_document_id: str | None = None


@dataclass
class ProcessedQuery:
    """Represents a processed user query."""

    original_text: str
    processed_text: str
    query_type: QueryType
    intent: str | None = None
    entities: list[str] = None
    keywords: list[str] = None
    metadata: dict[str, Any] = None
    timestamp: datetime = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.entities is None:
            self.entities = []
        if self.keywords is None:
            self.keywords = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalResult:
    """Result from document retrieval."""

    chunk: DocumentChunk
    relevance_score: float
    retrieval_method: str
    rank: int


@dataclass
class RAGContext:
    """Complete context for RAG generation."""

    query: ProcessedQuery
    retrieved_chunks: list[RetrievalResult]
    generation_strategy: GenerationStrategy
    prompt_template: str | None = None
    system_prompt: str | None = None
    context_window_size: int = 4000
    max_chunks: int = 5


@dataclass
class GenerationResult:
    """Result from text generation."""

    generated_text: str
    confidence_score: float | None = None
    tokens_used: int | None = None
    model_used: str | None = None
    generation_time: float | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RAGResponse:
    """Complete RAG response."""

    query: ProcessedQuery
    answer: str
    sources: list[RetrievalResult]
    generation_result: GenerationResult
    processing_metadata: dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    @property
    def source_documents(self) -> list[DocumentChunk]:
        """Get source document chunks."""
        return [result.chunk for result in self.sources]

    @property
    def confidence_score(self) -> float | None:
        """Get overall confidence score."""
        return self.generation_result.confidence_score

    @property
    def total_sources(self) -> int:
        """Get total number of sources used."""
        return len(self.sources)
