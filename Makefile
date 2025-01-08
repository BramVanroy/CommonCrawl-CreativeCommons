quality:
	ruff check src/gpt_nl_copyright scripts/
	ruff format --check src/gpt_nl_copyright scripts/

style:
	ruff check src/gpt_nl_copyright scripts/ --fix
	ruff format src/gpt_nl_copyright scripts/
