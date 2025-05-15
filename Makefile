quality:
	ruff check src/c5 scripts/
	ruff format --check src/c5 scripts/

style:
	ruff check src/c5 scripts/ --fix
	ruff format src/c5 scripts/
