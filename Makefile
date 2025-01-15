quality:
	ruff check src/commoncrawl_cc_annotation scripts/
	ruff format --check src/commoncrawl_cc_annotation scripts/

style:
	ruff check src/commoncrawl_cc_annotation scripts/ --fix
	ruff format src/commoncrawl_cc_annotation scripts/
