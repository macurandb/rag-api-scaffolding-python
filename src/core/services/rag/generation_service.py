"""Generation service for RAG operations."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from ...domain.rag import GenerationResult, GenerationStrategy, RAGContext

logger = structlog.get_logger()


class GenerationService(ABC):
    """Abstract service for text generation."""

    @abstractmethod
    async def generate(
        self,
        context: RAGContext,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text based on RAG context."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        context: RAGContext,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming response."""
        pass

    @abstractmethod
    async def generate_with_citations(
        self,
        context: RAGContext,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate text with inline citations."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the generation model."""
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Check the health of the generation service."""
        pass


class GenerationOptimizer:
    """Optimizer for generation parameters based on query type and context."""

    def __init__(self):
        self.strategy_configs = {
            GenerationStrategy.STANDARD: {
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "You are a helpful assistant. Answer based on the provided context.",
            },
            GenerationStrategy.CHAIN_OF_THOUGHT: {
                "temperature": 0.3,
                "max_tokens": 1500,
                "system_prompt": "Think step by step. Break down your reasoning process.",
            },
            GenerationStrategy.STEP_BY_STEP: {
                "temperature": 0.2,
                "max_tokens": 2000,
                "system_prompt": "Provide a detailed step-by-step explanation.",
            },
            GenerationStrategy.CREATIVE: {
                "temperature": 0.9,
                "max_tokens": 1200,
                "system_prompt": "Be creative and engaging in your response.",
            },
        }

    def optimize_parameters(
        self,
        context: RAGContext,
        base_temperature: float = 0.7,
        base_max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Optimize generation parameters based on context."""
        strategy_config = self.strategy_configs.get(
            context.generation_strategy,
            self.strategy_configs[GenerationStrategy.STANDARD],
        )

        # Adjust based on query type
        query_type_adjustments = {
            "question_answering": {"temperature": -0.1},
            "summarization": {"temperature": -0.2, "max_tokens": 500},
            "analysis": {"temperature": -0.1, "max_tokens": 1500},
            "creative": {"temperature": 0.2},
        }

        adjustments = query_type_adjustments.get(context.query.query_type.value, {})

        # Apply adjustments
        optimized = strategy_config.copy()
        optimized["temperature"] = max(
            0.0, min(1.0, optimized["temperature"] + adjustments.get("temperature", 0))
        )

        if "max_tokens" in adjustments:
            optimized["max_tokens"] = adjustments["max_tokens"]

        # Override with user-provided values
        if base_temperature != 0.7:
            optimized["temperature"] = base_temperature
        if base_max_tokens:
            optimized["max_tokens"] = base_max_tokens

        return optimized


class GenerationMetrics:
    """Metrics for generation operations."""

    def __init__(self):
        self.total_generations = 0
        self.total_tokens_generated = 0
        self.total_tokens_input = 0
        self.average_generation_time = 0.0
        self.strategy_usage = {}

    def record_generation(
        self,
        tokens_generated: int,
        tokens_input: int,
        generation_time: float,
        strategy: GenerationStrategy,
    ):
        """Record metrics for a generation operation."""
        self.total_generations += 1
        self.total_tokens_generated += tokens_generated
        self.total_tokens_input += tokens_input

        # Update average generation time
        total_time = self.average_generation_time * (self.total_generations - 1)
        self.average_generation_time = (
            total_time + generation_time
        ) / self.total_generations

        # Track strategy usage
        strategy_name = strategy.value
        self.strategy_usage[strategy_name] = (
            self.strategy_usage.get(strategy_name, 0) + 1
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        return {
            "total_generations": self.total_generations,
            "total_tokens_generated": self.total_tokens_generated,
            "total_tokens_input": self.total_tokens_input,
            "average_generation_time": self.average_generation_time,
            "tokens_per_second": (
                self.total_tokens_generated
                / (self.average_generation_time * self.total_generations)
                if self.total_generations > 0 and self.average_generation_time > 0
                else 0
            ),
            "strategy_usage": self.strategy_usage,
        }
