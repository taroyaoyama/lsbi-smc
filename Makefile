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
