.PHONY: lint
lint:
	mypy --disable-error-code=attr-defined .
	ruff check --fix .

.PHONY: test
test:
	pytest -s tests

.PHONY: coverage
coverage:
	pytest -s --cov=src --cov-fail-under=80 --cov-branch tests
