.PHONY: lint
lint:
	mypy --disable-error-code=attr-defined .
	ruff check --fix .

# Pytest debug on fail
.PHONY: test
test:
	pytest -s tests --pdb

.PHONY: coverage
coverage:
	pytest -s --cov=src --cov-fail-under=80 --cov-branch tests
