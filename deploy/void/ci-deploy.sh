#!/usr/bin/env bash
set -euo pipefail

deploy() {
    if [[ ! ${SSH_ORIGINAL_COMMAND:-} =~ ^litellm-deploy\ ([0-9a-f]{40})$ ]]; then
        printf 'Expected: litellm-deploy <40-character commit SHA>\n' >&2
        exit 1
    fi
    local sha=${BASH_REMATCH[1]}
    local fetched container status

    exec 9>/var/lock/void-deploy.lock
    flock --exclusive --timeout 1800 9
    cd /root/litellm
    [[ $(git symbolic-ref --short HEAD) == main ]]
    status=$(git status --porcelain --untracked-files=all)
    [[ -z $status ]]
    git fetch --no-tags origin refs/heads/main
    fetched=$(git rev-parse 'FETCH_HEAD^{commit}')
    if [[ $fetched != "$sha" ]]; then
        printf 'Skipping stale deployment %s; origin/main is %s\n' "$sha" "$fetched"
        exit 0
    fi
    git merge --ff-only "$fetched"
    [[ $(git rev-parse HEAD) == "$sha" ]]

    make -C deploy/void prod-up UP_FLAGS='--wait --wait-timeout 180'
    container=$(docker compose --project-directory /root/litellm/deploy/void \
        --env-file /root/litellm/deploy/void/.env \
        -f deploy/void/compose.yml -f deploy/void/compose.prod.yml \
        -p void-litellm ps -q litellm)
    [[ -n $container ]]
    docker exec "$container" python3 -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:4000/health/liveliness', timeout=10).close()"
    curl --fail --silent --show-error --max-time 30 --retry 5 --retry-all-errors \
        https://void-api.tech/v1/models | python3 -c \
        'import json, sys; result = json.load(sys.stdin); assert result["object"] == "list" and isinstance(result["data"], list) and result["data"], "Empty or invalid public model catalog"'
    printf 'Deployed LiteLLM %s\n' "$sha"
    exit 0
}

deploy
