.PHONY: lint
lint:
	mypy --disable-error-code=attr-defined .
	ruff check --fix .
