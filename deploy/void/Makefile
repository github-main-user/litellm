ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ENV_FILE ?= $(ROOT)/.env
SERVICES ?=
UP_FLAGS ?=

COMPOSE = docker compose --project-directory "$(ROOT)" --env-file "$(ENV_FILE)" -f "$(ROOT)/compose.yml"
DEV_COMPOSE = $(COMPOSE) -p "$(or $(COMPOSE_PROJECT_NAME),void-litellm-dev)" -f "$(ROOT)/compose.dev.yml"
PROD_COMPOSE = $(COMPOSE) -p "$(or $(COMPOSE_PROJECT_NAME),void-litellm)" -f "$(ROOT)/compose.prod.yml"

.PHONY: dev-up dev-down dev-ps dev-network prod-up prod-down prod-ps prod-network prod-check-volume

dev-network prod-network:
	@set -eu; \
	network="$$($(if $(filter dev-network,$@),$(DEV_COMPOSE),$(PROD_COMPOSE)) config --format json | python3 -c 'import json, sys; print(json.load(sys.stdin)["networks"]["inference"]["name"])')"; \
	if docker network inspect "$$network" >/dev/null 2>&1; then \
		test "$$(docker network inspect --format '{{.Internal}}' "$$network")" = true || { printf 'Network %s must be internal\n' "$$network" >&2; exit 1; }; \
	else \
		docker network create --internal "$$network"; \
	fi

dev-up:
	$(DEV_COMPOSE) up --build -d $(UP_FLAGS) $(SERVICES)

dev-down:
	$(DEV_COMPOSE) down

dev-ps:
	$(DEV_COMPOSE) ps

prod-check-volume:
	@set -eu; \
	volume="$$($(PROD_COMPOSE) config --format json | python3 -c 'import json, sys; print(json.load(sys.stdin)["volumes"]["litellm_postgres_data"]["name"])')"; \
	docker volume inspect "$$volume" >/dev/null; \
	current="$$($(PROD_COMPOSE) ps -q litellm-db)"; \
	for container in $$(docker ps -q --no-trunc --filter "volume=$$volume"); do \
		test "$$container" = "$$current" || { printf 'Stop container %s before handing off volume %s\n' "$$container" "$$volume" >&2; exit 1; }; \
	done; \
	if test -z "$$current"; then \
		docker run --rm --network none --read-only --mount "type=volume,src=$$volume,dst=/data,readonly" --entrypoint sh postgres:16-alpine -ec 'test "$$(cat /data/PG_VERSION)" = 16 && test -d /data/base && test -s /data/global/pg_control'; \
	fi

prod-up: prod-check-volume
	$(PROD_COMPOSE) up --build -d $(UP_FLAGS) $(SERVICES)

prod-down:
	$(PROD_COMPOSE) down

prod-ps:
	$(PROD_COMPOSE) ps
