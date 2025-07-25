# RAG Application Makefile
# Requires: uv (https://docs.astral.sh/uv/)

.PHONY: help install install-dev clean test test-cov lint format check run docker-build docker-run setup-env

# Default target
help: ## Show this help message
	@echo "RAG Application - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
setup-env: ## Setup development environment
	@echo "🔧 Setting up development environment..."
	@if ! command -v uv &> /dev/null; then \
		echo "❌ uv is not installed. Please install it first: https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	fi
	@echo "✅ uv is installed"
	@cp .env.example .env || echo "⚠️  .env already exists, skipping copy"
	@echo "📝 Please edit .env file with your API keys"
	@echo "🚀 Environment setup complete!"

install: ## Install production dependencies
	@echo "📦 Installing production dependencies..."
	uv sync --no-dev

install-dev: ## Install all dependencies including dev tools
	@echo "📦 Installing all dependencies..."
	uv sync --extra dev
	@echo "🔧 Setting up pre-commit hooks..."
	uv run pre-commit install

clean: ## Clean cache and temporary files
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "✅ Cleanup complete!"

# Code quality
format: ## Format code with black and ruff
	@echo "🎨 Formatting code..."
	uv run black src/ tests/
	uv run ruff check --fix src/ tests/

lint: ## Run linting checks
	@echo "🔍 Running linting checks..."
	uv run ruff check src/ tests/
	uv run black --check src/ tests/
	uv run mypy src/

check: lint ## Alias for lint

# Testing
test: ## Run tests
	@echo "🧪 Running tests..."
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	uv run pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml
	@echo "📊 Coverage report generated in htmlcov/"

test-watch: ## Run tests in watch mode
	@echo "👀 Running tests in watch mode..."
	uv run pytest-watch tests/ -- -v

# Application
run: ## Run the application
	@echo "🚀 Starting RAG application..."
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

run-prod: ## Run the application in production mode
	@echo "🚀 Starting RAG application (production)..."
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# Development workflow
dev: install-dev ## Setup development environment and run app
	@echo "🔄 Starting development workflow..."
	@make run

# Quality assurance workflow
qa: clean format lint test-cov ## Run complete quality assurance pipeline
	@echo "✅ Quality assurance pipeline completed!"

# Docker
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t rag-app:latest .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p 8000:8000 --env-file .env rag-app:latest

docker-dev: ## Run Docker container with volume mount for development
	@echo "🐳 Running Docker container (development)..."
	docker run -p 8000:8000 --env-file .env -v $(PWD):/app rag-app:latest

# Database/Vector Store
init-vectorstore: ## Initialize vector store with sample data
	@echo "🗄️ Initializing vector store..."
	@echo "⚠️  Note: Requires valid OPENAI_API_KEY in .env file"
	uv run python -c "from src.infrastructure.vector_store_init import init_sample_data_sync; init_sample_data_sync()"

# Utilities
deps-update: ## Update dependencies
	@echo "📦 Updating dependencies..."
	uv lock --upgrade

deps-audit: ## Audit dependencies for security issues
	@echo "🔒 Auditing dependencies..."
	uv run pip-audit

logs: ## Show application logs (if running in Docker)
	@echo "📋 Showing application logs..."
	docker logs -f rag-app 2>/dev/null || echo "No running container found"

health: ## Check application health
	@echo "🏥 Checking application health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "❌ Application not responding"

# Pre-commit and CI/CD
pre-commit: ## Run pre-commit hooks on all files
	@echo "🔍 Running pre-commit hooks..."
	uv run pre-commit run --all-files

ci: clean install-dev lint test-cov ## Run CI pipeline locally
	@echo "🔄 Running CI pipeline..."
	@echo "✅ CI pipeline completed successfully!"

# Documentation
docs-serve: ## Serve API documentation
	@echo "📚 Starting API documentation server..."
	@echo "📖 API docs available at: http://localhost:8000/docs"
	@echo "📖 ReDoc available at: http://localhost:8000/redoc"
	@make run

# Environment info
info: ## Show environment information
	@echo "ℹ️  Environment Information:"
	@echo "Python version: $$(python --version)"
	@echo "uv version: $$(uv --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Virtual environment: $$(uv run python -c 'import sys; print(sys.prefix)')"

# Quick start for new developers
quickstart: setup-env install-dev init-vectorstore ## Complete setup for new developers
	@echo ""
	@echo "🎉 Quickstart completed!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env file with your API keys"
	@echo "2. Run 'make run' to start the application"
	@echo "3. Visit http://localhost:8000/docs for API documentation"
	@echo ""