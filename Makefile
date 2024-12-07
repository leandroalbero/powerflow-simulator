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
