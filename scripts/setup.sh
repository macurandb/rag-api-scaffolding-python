#!/bin/bash

# RAG Application Setup Script
set -e

echo "🚀 Setting up RAG Application..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "✅ uv is available"

# Setup environment
echo "🔧 Setting up environment..."
cp .env.example .env 2>/dev/null || echo "⚠️  .env already exists"

# Install dependencies
echo "📦 Installing dependencies..."
uv sync --extra dev

# Setup pre-commit
echo "🔧 Setting up pre-commit hooks..."
uv run pre-commit install

# Initialize vector store
echo "🗄️ Initializing vector store..."
mkdir -p data/vector_store

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run 'make run' to start the application"
echo "3. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "Available commands:"
echo "  make help     - Show all available commands"
echo "  make run      - Start the application"
echo "  make test     - Run tests"
echo "  make qa       - Run quality assurance pipeline"
echo ""