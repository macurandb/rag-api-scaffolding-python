"""LangChain-based service implementations.

DEPRECATED: This file contains legacy implementations.
Use src/infrastructure/rag/langchain_implementations.py instead.
This file is kept for reference during scaffolding cleanup.
"""

import structlog
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document as LangChainDocument
from langchain.vectorstores import FAISS

from ..config import settings
from ..core.domain.models import Document, Query, RAGContext, RetrievedDocument
from ..core.ports.services import (
    EmbeddingService,
    GeneratorService,
    PromptBuilderService,
    RetrieverService,
)

logger = structlog.get_logger()


class LangChainEmbeddingService(EmbeddingService):
    """LangChain-based embedding service."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model, openai_api_key=settings.openai_api_key
        )

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for text."""
        try:
            embedding = await self.embeddings.aembed_query(text)
            return embedding
        except Exception as e:
            logger.error("Error generating embedding", error=str(e))
            raise

    async def embed_documents(self, documents: list[Document]) -> list[Document]:
        """Generate embeddings for multiple documents."""
        try:
            texts = [doc.content for doc in documents]
            embeddings = await self.embeddings.aembed_documents(texts)

            for doc, embedding in zip(documents, embeddings, strict=False):
                doc.embedding = embedding

            return documents
        except Exception as e:
            logger.error("Error generating document embeddings", error=str(e))
            raise


class LangChainRetrieverService(RetrieverService):
    """LangChain-based retriever service."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.vector_store = None

    async def _ensure_vector_store(self):
        """Ensure vector store is initialized."""
        if self.vector_store is None:
            # Initialize with dummy data for now
            # In production, this would load from persistent storage
            dummy_docs = [
                LangChainDocument(
                    page_content="Sample document content",
                    metadata={"source": "sample"},
                )
            ]
            self.vector_store = FAISS.from_documents(
                dummy_docs, self.embedding_service.embeddings
            )

    async def retrieve(self, query: Query, k: int = 5) -> list[RetrievedDocument]:
        """Retrieve relevant documents for a query."""
        try:
            await self._ensure_vector_store()

            # Search similar documents
            results = self.vector_store.similarity_search_with_score(query.text, k=k)

            retrieved_docs = []
            for doc, score in results:
                document = Document(
                    id=doc.metadata.get("id", "unknown"),
                    content=doc.page_content,
                    metadata=doc.metadata,
                )
                retrieved_docs.append(RetrievedDocument(document=document, score=score))

            return retrieved_docs

        except Exception as e:
            logger.error("Error retrieving documents", error=str(e))
            raise


class DefaultPromptBuilderService(PromptBuilderService):
    """Default prompt builder implementation."""

    async def build_prompt(self, context: RAGContext) -> str:
        """Build prompt from query and retrieved documents."""
        try:
            # Build context from retrieved documents
            context_text = "\n\n".join(
                [
                    f"Document {i+1}:\n{doc.document.content}"
                    for i, doc in enumerate(context.retrieved_documents)
                ]
            )

            prompt = f"""Based on the following context, answer the user's question.

Context:
{context_text}

Question: {context.query.text}

Answer:"""

            return prompt

        except Exception as e:
            logger.error("Error building prompt", error=str(e))
            raise


class LangChainGeneratorService(GeneratorService):
    """LangChain-based text generator."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            openai_api_key=settings.openai_api_key,
        )

    async def generate(self, prompt: str) -> str:
        """Generate response from prompt."""
        try:
            response = await self.llm.apredict(prompt)
            return response.strip()
        except Exception as e:
            logger.error("Error generating response", error=str(e))
            raise
