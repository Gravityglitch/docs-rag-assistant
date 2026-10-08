.PHONY: help install lint fmt test run ingest ingest-full eval up down logs docker-build

.DEFAULT_GOAL := help

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependency groups with uv
	uv sync --all-groups

lint: ## Run ruff lint and format checks
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Auto-format and auto-fix with ruff
	uv run ruff format .
	uv run ruff check --fix .

test: ## Run tests with coverage (fail under 80%)
	uv run pytest -q --cov=rag --cov-report=term-missing --cov-fail-under=80

run: ## Start the API server locally
	uv run rag serve

ingest: ## Ingest documents incrementally
	uv run rag ingest

ingest-full: ## Re-ingest all documents from scratch
	uv run rag ingest --full

eval: ## Run the evaluation module
	uv run python -m rag.eval

up: ## Build and start the stack with docker compose
	docker compose up --build -d

down: ## Stop the docker compose stack
	docker compose down

logs: ## Follow API container logs
	docker compose logs -f api

docker-build: ## Build the API image locally
	docker build -t docs-rag-assistant:local .
