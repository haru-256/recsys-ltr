.PHONY: help build_dataset lint
.DEFAULT_GOAL := help

lint: ## Run Linter
	ruff .

fmt: ## Run formatter
	black .

build_dataset: ## Build RPM public dataset. For more info: https://github.com/LibRerank-Community/LibRerank/blob/master/README.md#prm-public
	python build_dataset.py

help: ## Show options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
