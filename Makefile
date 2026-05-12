export UID := $(shell id -u)

.PHONY: build
build:
	docker compose build

.PHONY: build-no-cache
build-no-cache:
	docker compose build --no-cache

.PHONY: up
up:
	docker compose up -d

.PHONY: figures
figures:
	uv run python docs/presentation/assets/make_figures.py

.PHONY: slides
slides: figures
	cd docs/presentation && npx -y @marp-team/marp-cli@latest 02_slides.md --pdf --allow-local-files

.PHONY: slides-pptx
slides-pptx:
	cd docs/presentation && npx -y @marp-team/marp-cli@latest 02_slides.md --pptx --allow-local-files

.PHONY: slides-html
slides-html:
	cd docs/presentation && npx -y @marp-team/marp-cli@latest 02_slides.md --html --allow-local-files
