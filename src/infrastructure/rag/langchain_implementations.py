"""LangChain implementations of RAG services."""

import time
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from langchain.schema import Document as LangChainDocument
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ...config import settings
from ...core.domain.rag import (
    DocumentChunk,
    GenerationResult,
    ProcessedQuery,
    RAGContext,
    RetrievalResult,
    RetrievalStrategy,
)
from ...core.services.rag.embedding_service import EmbeddingMetrics, EmbeddingService
from ...core.services.rag.generation_service import (
    GenerationOptimizer,
    GenerationService,
)
from ...core.services.rag.prompt_service import DefaultPromptService
from ...core.services.rag.retrieval_service import RetrievalService

logger = structlog.get_logger()


class LangChainEmbeddingService(EmbeddingService):
    """LangChain-based embedding service."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model, openai_api_key=settings.openai_api_key
        )
        self.metrics = EmbeddingMetrics()

    async def embed_query(self, query: ProcessedQuery) -> list[float]:
        """Generate embedding for a processed query."""
        start_time = time.time()
        try:
            text = query.processed_text or query.original_text
            embedding = await self.embeddings.aembed_query(text)

            processing_time = time.time() - start_time
            self.metrics.record_embedding(len(text.split()), processing_time)

            return embedding
        except Exception as e:
            logger.error("Error generating query embedding", error=str(e))
            raise

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for raw text."""
        start_time = time.time()
        try:
            embedding = await self.embeddings.aembed_query(text)

            processing_time = time.time() - start_time
            self.metrics.record_embedding(len(text.split()), processing_time)

            return embedding
        except Exception as e:
            logger.error("Error generating text embedding", error=str(e))
            raise

    async def embed_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Generate embeddings for document chunks."""
        start_time = time.time()
        try:
            texts = [chunk.content for chunk in chunks]
            embeddings = await self.embeddings.aembed_documents(texts)

            for chunk, embedding in zip(chunks, embeddings, strict=False):
                chunk.embedding = embedding

            processing_time = time.time() - start_time
            total_tokens = sum(len(text.split()) for text in texts)
            self.metrics.record_embedding(total_tokens, processing_time, len(chunks))

            return chunks
        except Exception as e:
            logger.error("Error generating chunk embeddings", error=str(e))
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        start_time = time.time()
        try:
            embeddings = await self.embeddings.aembed_documents(texts)

            processing_time = time.time() - start_time
            total_tokens = sum(len(text.split()) for text in texts)
            self.metrics.record_embedding(total_tokens, processing_time, len(texts))

            return embeddings
        except Exception as e:
            logger.error("Error generating batch embeddings", error=str(e))
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        # OpenAI text-embedding-ada-002 has 1536 dimensions
        return 1536

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model."""
        return {
            "model": settings.embedding_model,
            "provider": "openai",
            "dimension": self.get_embedding_dimension(),
            "metrics": self.metrics.get_metrics(),
        }


class LangChainRetrievalService(RetrievalService):
    """LangChain-based retrieval service."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.vector_store = None

    async def _ensure_vector_store(self):
        """Ensure vector store is initialized."""
        if self.vector_store is None:
            # Initialize with sample data
            dummy_docs = [
                LangChainDocument(
                    page_content="Artificial Intelligence (AI) is a branch of computer science that deals with creating systems capable of performing tasks that normally require human intelligence.",
                    metadata={
                        "id": "ai_definition",
                        "source": "ai_basics",
                        "category": "definition",
                    },
                ),
                LangChainDocument(
                    page_content="Machine Learning is a subset of artificial intelligence that enables machines to learn and improve automatically from experience without being explicitly programmed.",
                    metadata={
                        "id": "ml_definition",
                        "source": "ml_intro",
                        "category": "machine_learning",
                    },
                ),
                LangChainDocument(
                    page_content="Neural networks are computational models inspired by the human brain, consisting of interconnected nodes that process information.",
                    metadata={
                        "id": "nn_definition",
                        "source": "neural_networks",
                        "category": "deep_learning",
                    },
                ),
            ]
            self.vector_store = FAISS.from_documents(
                dummy_docs, self.embedding_service.embeddings
            )

    async def retrieve(
        self,
        query: ProcessedQuery,
        strategy: RetrievalStrategy = RetrievalStrategy.SIMILARITY,
        max_results: int = 5,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        try:
            await self._ensure_vector_store()

            query_text = query.processed_text or query.original_text

            # Search similar documents
            results = self.vector_store.similarity_search_with_score(
                query_text, k=max_results
            )

            retrieval_results = []
            for i, (doc, score) in enumerate(results):
                if score >= min_score:
                    chunk = DocumentChunk(
                        id=doc.metadata.get("id", f"doc_{i}"),
                        content=doc.page_content,
                        metadata=doc.metadata,
                    )

                    retrieval_results.append(
                        RetrievalResult(
                            chunk=chunk,
                            relevance_score=float(
                                1.0 - score
                            ),  # Convert distance to similarity
                            retrieval_method=strategy.value,
                            rank=i + 1,
                        )
                    )

            return retrieval_results

        except Exception as e:
            logger.error("Error retrieving documents", error=str(e))
            raise

    async def retrieve_by_ids(self, document_ids: list[str]) -> list[RetrievalResult]:
        """Retrieve documents by their IDs."""
        # Simple implementation - in production, this would query by ID
        await self._ensure_vector_store()

        # For now, return empty results
        return []

    async def get_similar_documents(
        self, document_id: str, max_results: int = 5
    ) -> list[RetrievalResult]:
        """Get documents similar to a given document."""
        # Simple implementation - in production, this would find similar docs
        await self._ensure_vector_store()

        return []

    async def health_check(self) -> dict[str, Any]:
        """Check the health of the retrieval system."""
        try:
            await self._ensure_vector_store()
            return {
                "status": "healthy",
                "vector_store": "initialized",
                "embedding_service": "connected",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


class LangChainGenerationService(GenerationService):
    """LangChain-based generation service."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
        )
        self.optimizer = GenerationOptimizer()

    async def generate(
        self,
        context: RAGContext,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text based on RAG context."""
        start_time = time.time()

        try:
            # Build prompt
            prompt_service = DefaultPromptService()
            prompt = await prompt_service.build_prompt(context)
            system_prompt = await prompt_service.build_system_prompt(context)

            # Optimize parameters
            optimized_params = self.optimizer.optimize_parameters(
                context, temperature, max_tokens
            )

            # Generate response
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            # Update LLM parameters
            self.llm.temperature = optimized_params["temperature"]
            if max_tokens:
                self.llm.max_tokens = max_tokens

            response = await self.llm.apredict_messages(messages)
            generated_text = response.content

            generation_time = time.time() - start_time

            return GenerationResult(
                generated_text=generated_text,
                confidence_score=0.8,  # Placeholder
                tokens_used=len(generated_text.split()),  # Rough estimate
                model_used=settings.llm_model,
                generation_time=generation_time,
                metadata={
                    "temperature": optimized_params["temperature"],
                    "system_prompt_length": len(system_prompt),
                    "prompt_length": len(prompt),
                },
            )

        except Exception as e:
            logger.error("Error generating response", error=str(e))
            raise

    async def generate_stream(
        self,
        context: RAGContext,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming response."""
        # Placeholder implementation
        result = await self.generate(context, temperature, max_tokens, stop_sequences)

        # Simulate streaming by yielding chunks
        words = result.generated_text.split()
        for i in range(0, len(words), 5):
            chunk = " ".join(words[i : i + 5])
            yield chunk + " "

    async def generate_with_citations(
        self,
        context: RAGContext,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate text with inline citations."""
        # Modify context to request citations
        citation_context = RAGContext(
            query=context.query,
            retrieved_chunks=context.retrieved_chunks,
            generation_strategy=context.generation_strategy,
            system_prompt=context.system_prompt
            + "\n\nInclude citations using [Source X] format.",
            context_window_size=context.context_window_size,
            max_chunks=context.max_chunks,
        )

        return await self.generate(citation_context, temperature, max_tokens)

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the generation model."""
        return {
            "model": settings.llm_model,
            "provider": "openai",
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }

    async def health_check(self) -> dict[str, Any]:
        """Check the health of the generation service."""
        try:
            # Simple health check with a basic prompt
            test_response = await self.llm.apredict("Say 'healthy' if you can respond.")
            return {
                "status": "healthy",
                "model": settings.llm_model,
                "test_response": test_response[:50],
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
