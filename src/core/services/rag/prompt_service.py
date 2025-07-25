"""Prompt service for RAG operations."""

from abc import ABC, abstractmethod

import structlog

from ...domain.rag import QueryType, RAGContext, RetrievalResult

logger = structlog.get_logger()


class PromptService(ABC):
    """Abstract service for prompt construction and optimization."""

    @abstractmethod
    async def build_prompt(self, context: RAGContext) -> str:
        """Build a prompt from RAG context."""
        pass

    @abstractmethod
    async def build_system_prompt(self, context: RAGContext) -> str:
        """Build system prompt for the context."""
        pass

    @abstractmethod
    async def optimize_context_window(
        self, context: RAGContext, max_tokens: int
    ) -> RAGContext:
        """Optimize context to fit within token limits."""
        pass

    @abstractmethod
    def get_prompt_templates(self) -> dict[str, str]:
        """Get available prompt templates."""
        pass


class DefaultPromptService(PromptService):
    """Default implementation of prompt service."""

    def __init__(self):
        self.templates = {
            QueryType.QUESTION_ANSWERING: """Based on the following context, answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Answer:""",
            QueryType.SUMMARIZATION: """Based on the following context, provide a comprehensive summary.

Context:
{context}

Summarization request: {query}

Summary:""",
            QueryType.ANALYSIS: """Based on the following context, provide a detailed analysis.

Context:
{context}

Analysis request: {query}

Analysis:""",
            QueryType.COMPARISON: """Based on the following context, provide a detailed comparison.

Context:
{context}

Comparison request: {query}

Comparison:""",
            QueryType.EXTRACTION: """Based on the following context, extract the requested information.

Context:
{context}

Extraction request: {query}

Extracted information:""",
        }

        self.system_prompts = {
            QueryType.QUESTION_ANSWERING: "You are a helpful assistant that answers questions based on provided context. Be accurate and cite sources when possible.",
            QueryType.SUMMARIZATION: "You are an expert at creating concise, comprehensive summaries. Focus on key points and main ideas.",
            QueryType.ANALYSIS: "You are an analytical expert. Provide thorough analysis with reasoning and evidence from the context.",
            QueryType.COMPARISON: "You are skilled at comparing and contrasting information. Highlight similarities, differences, and key insights.",
            QueryType.EXTRACTION: "You are precise at extracting specific information. Only include information that is explicitly stated in the context.",
        }

    async def build_prompt(self, context: RAGContext) -> str:
        """Build prompt from context."""
        try:
            # Get appropriate template
            template = self.templates.get(
                context.query.query_type, self.templates[QueryType.QUESTION_ANSWERING]
            )

            # Build context string from retrieved chunks
            context_text = self._build_context_text(context.retrieved_chunks)

            # Format prompt
            prompt = template.format(
                context=context_text,
                query=context.query.processed_text or context.query.original_text,
            )

            return prompt

        except Exception as e:
            logger.error("Error building prompt", error=str(e))
            raise

    async def build_system_prompt(self, context: RAGContext) -> str:
        """Build system prompt for context."""
        base_prompt = self.system_prompts.get(
            context.query.query_type, self.system_prompts[QueryType.QUESTION_ANSWERING]
        )

        # Add context-specific instructions
        if context.retrieved_chunks:
            source_count = len(context.retrieved_chunks)
            base_prompt += f"\n\nYou have access to {source_count} relevant source(s). Always base your response on the provided context."

        return base_prompt

    async def optimize_context_window(
        self, context: RAGContext, max_tokens: int
    ) -> RAGContext:
        """Optimize context to fit within token limits."""
        # Simple token estimation (4 chars ≈ 1 token)
        estimated_tokens_per_char = 0.25

        # Reserve tokens for query and response
        reserved_tokens = 500
        available_tokens = max_tokens - reserved_tokens
        available_chars = int(available_tokens / estimated_tokens_per_char)

        # Sort chunks by relevance score
        sorted_chunks = sorted(
            context.retrieved_chunks, key=lambda x: x.relevance_score, reverse=True
        )

        # Select chunks that fit within limit
        selected_chunks = []
        current_chars = 0

        for chunk_result in sorted_chunks:
            chunk_chars = len(chunk_result.chunk.content)
            if current_chars + chunk_chars <= available_chars:
                selected_chunks.append(chunk_result)
                current_chars += chunk_chars
            else:
                # Try to include partial content if it's the first chunk
                if not selected_chunks and chunk_chars > available_chars:
                    # Truncate the chunk content
                    truncated_content = (
                        chunk_result.chunk.content[: available_chars - 100] + "..."
                    )
                    chunk_result.chunk.content = truncated_content
                    selected_chunks.append(chunk_result)
                break

        # Update context with optimized chunks
        optimized_context = RAGContext(
            query=context.query,
            retrieved_chunks=selected_chunks,
            generation_strategy=context.generation_strategy,
            prompt_template=context.prompt_template,
            system_prompt=context.system_prompt,
            context_window_size=max_tokens,
            max_chunks=len(selected_chunks),
        )

        logger.info(
            "Context optimized",
            original_chunks=len(context.retrieved_chunks),
            optimized_chunks=len(selected_chunks),
            estimated_tokens=int(current_chars * estimated_tokens_per_char),
        )

        return optimized_context

    def get_prompt_templates(self) -> dict[str, str]:
        """Get available prompt templates."""
        return self.templates.copy()

    def _build_context_text(self, retrieved_chunks: list[RetrievalResult]) -> str:
        """Build context text from retrieved chunks."""
        if not retrieved_chunks:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(retrieved_chunks, 1):
            chunk = result.chunk
            source_info = f"Source {i}"

            # Add metadata if available
            if chunk.metadata:
                title = chunk.metadata.get("title", "")
                source = chunk.metadata.get("source", "")
                if title:
                    source_info += f" - {title}"
                if source:
                    source_info += f" ({source})"

            context_parts.append(f"{source_info}:\n{chunk.content}")

        return "\n\n".join(context_parts)


class PromptOptimizer:
    """Optimizer for prompt engineering and performance."""

    def __init__(self):
        self.optimization_strategies = {
            "concise": self._make_concise,
            "detailed": self._make_detailed,
            "structured": self._add_structure,
            "citations": self._add_citations,
        }

    async def optimize_prompt(
        self,
        prompt: str,
        strategy: str = "standard",
        context: RAGContext | None = None,
    ) -> str:
        """Optimize prompt based on strategy."""
        if strategy in self.optimization_strategies:
            return await self.optimization_strategies[strategy](prompt, context)
        return prompt

    async def _make_concise(self, prompt: str, context: RAGContext | None) -> str:
        """Make prompt more concise."""
        # Add instruction for conciseness
        return (
            prompt
            + "\n\nProvide a concise response focusing on the most important information."
        )

    async def _make_detailed(self, prompt: str, context: RAGContext | None) -> str:
        """Make prompt request more detail."""
        return (
            prompt
            + "\n\nProvide a detailed and comprehensive response with explanations."
        )

    async def _add_structure(self, prompt: str, context: RAGContext | None) -> str:
        """Add structure to the response."""
        return (
            prompt
            + "\n\nStructure your response with clear headings and bullet points where appropriate."
        )

    async def _add_citations(self, prompt: str, context: RAGContext | None) -> str:
        """Add citation requirements."""
        return (
            prompt
            + "\n\nInclude citations to the source material in your response using [Source X] format."
        )
