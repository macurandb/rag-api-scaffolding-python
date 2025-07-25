"""Initialize vector store with sample data."""

import asyncio

import structlog
from langchain.schema import Document as LangChainDocument
from langchain_community.vectorstores import FAISS

from ..core.domain.rag import DocumentChunk
from ..infrastructure.rag.langchain_implementations import LangChainEmbeddingService

logger = structlog.get_logger()


async def init_sample_data():
    """Initialize vector store with sample documents."""
    logger.info("Initializing vector store with sample data")

    # Sample document chunks
    sample_chunks = [
        DocumentChunk(
            id="chunk_1",
            content="La inteligencia artificial (IA) es una rama de la informática que se ocupa de la creación de sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.",
            metadata={"source": "ai_basics", "category": "definition"},
            chunk_index=0,
            parent_document_id="doc_ai_basics",
        ),
        DocumentChunk(
            id="chunk_2",
            content="El machine learning es un subcampo de la inteligencia artificial que permite a las máquinas aprender y mejorar automáticamente a partir de la experiencia sin ser programadas explícitamente.",
            metadata={"source": "ml_intro", "category": "machine_learning"},
            chunk_index=0,
            parent_document_id="doc_ml_intro",
        ),
        DocumentChunk(
            id="chunk_3",
            content="Las redes neuronales artificiales son modelos computacionales inspirados en el funcionamiento del cerebro humano, compuestas por nodos interconectados que procesan información.",
            metadata={"source": "neural_networks", "category": "deep_learning"},
            chunk_index=0,
            parent_document_id="doc_neural_networks",
        ),
        DocumentChunk(
            id="chunk_4",
            content="El procesamiento de lenguaje natural (NLP) es una rama de la IA que ayuda a las computadoras a entender, interpretar y manipular el lenguaje humano.",
            metadata={"source": "nlp_guide", "category": "nlp"},
            chunk_index=0,
            parent_document_id="doc_nlp_guide",
        ),
        DocumentChunk(
            id="chunk_5",
            content="Los sistemas RAG (Retrieval-Augmented Generation) combinan la recuperación de información con la generación de texto para proporcionar respuestas más precisas y contextualizadas.",
            metadata={"source": "rag_systems", "category": "rag"},
            chunk_index=0,
            parent_document_id="doc_rag_systems",
        ),
    ]

    try:
        # Initialize embedding service
        embedding_service = LangChainEmbeddingService()

        # Convert to LangChain documents
        langchain_docs = [
            LangChainDocument(
                page_content=chunk.content,
                metadata={
                    **chunk.metadata,
                    "id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "parent_document_id": chunk.parent_document_id,
                },
            )
            for chunk in sample_chunks
        ]

        # Create vector store
        vector_store = FAISS.from_documents(
            langchain_docs, embedding_service.embeddings
        )

        # Save vector store
        vector_store.save_local("./data/vector_store")

        logger.info(
            "Sample data initialized successfully", chunk_count=len(sample_chunks)
        )
        print(f"✅ Vector store initialized with {len(sample_chunks)} document chunks")

    except Exception as e:
        logger.error("Error initializing sample data", error=str(e))
        print(f"❌ Error initializing vector store: {e}")
        raise


def init_sample_data_sync():
    """Synchronous wrapper for init_sample_data."""
    asyncio.run(init_sample_data())


if __name__ == "__main__":
    init_sample_data_sync()
