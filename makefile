.PHONY: lint
lint:
	@echo "Linting with Ruff:"
	@uv run ruff check --fix-only wikipedia_processing tests coding_style_format_example.py
	@uv run ruff check wikipedia_processing tests coding_style_format_example.py
	@echo "Type checking with Ty"
	@uv run ty check wikipedia_processing tests coding_style_format_example.py

.PHONY: test
test:
	@echo "Testing with coverage"
	@uv run coverage run
	@uv run coverage report