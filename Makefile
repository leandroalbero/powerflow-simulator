.PHONY: lint
lint:
	mypy --disable-error-code=attr-defined src
	ruff check --fix src

# Pytest debug on fail
.PHONY: test
test:
	pytest -s tests --pdb

.PHONY: coverage
coverage:
	pytest -s --cov=src --cov-fail-under=80 --cov-branch tests

# ---- Web UI ----

.PHONY: web-dev
web-dev:
	@echo "Starting backend and frontend dev servers..."
	@trap 'kill 0' EXIT; \
	cd web/frontend && npm run dev & \
	python -m uvicorn web.backend.main:app --reload --port 8000

.PHONY: web-build
web-build:
	cd web/frontend && npm run build
