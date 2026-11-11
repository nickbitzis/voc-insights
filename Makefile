PORT ?= 8873
TEXTS ?= 25000

.PHONY: install test lint format run eval gate report screenshot docker-build clean

install:
	uv sync --python 3.12

test:
	uv run pytest -q

lint:
	uvx ruff check .
	uvx ruff format --check .

format:
	uvx ruff format .
	uvx ruff check . --fix

run:
	PORT=$(PORT) uv run vocinsights --texts $(TEXTS) serve

eval:
	uv run vocinsights --texts $(TEXTS) eval --out reports/eval.json

gate:  ## same check CI runs: fail if micro F1 drops below the floor or the Q4 bug is not flagged
	uv run vocinsights --texts $(TEXTS) eval --gate

report:
	uv run vocinsights --texts $(TEXTS) report --out reports/insights-2025Q4.md
	uv run vocinsights --texts $(TEXTS) report --format html --out reports/insights-2025Q4.html

screenshot:  ## start the server first with make run
	uv run python scripts/screenshot.py --port $(PORT)

docker-build:
	docker build -t voc-insights .

clean:
	rm -rf .pytest_cache .ruff_cache
