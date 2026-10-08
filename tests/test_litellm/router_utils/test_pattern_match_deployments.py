"""Behavior pins for ``litellm/router_utils/pattern_match_deployments.py``."""

from __future__ import annotations

from unittest.mock import Mock

from litellm.router_utils import pattern_match_deployments
from litellm.router_utils.pattern_match_deployments import PatternMatchRouter, PatternUtils


def _wildcard_deployment(model_name: str) -> dict:
    return {"model_name": model_name, "litellm_params": {"model": model_name}}


def _matched_models(matches: list[dict] | None) -> list[str]:
    return [deployment["litellm_params"]["model"] for deployment in matches or []]






def test_get_pattern_missing_model_returns_none(monkeypatch):
    """Regression: a request without a model reaches the auth layer's pattern walk as ``None``; the
    declared-provider guard raised ``TypeError`` where the old inline resolve swallowed every
    resolver error, so the proxy's missing-model 400 became a crash."""

    def _unknown_provider(model, *args, **kwargs):
        raise ValueError(f"unknown provider for {model}")

    monkeypatch.setattr(pattern_match_deployments, "get_llm_provider", _unknown_provider)
    router = PatternMatchRouter()
    router.add_pattern("openai/*", _wildcard_deployment("openai/*"))
    assert router.get_pattern(None) is None


def test_get_pattern_still_resolves_unqualified_names(monkeypatch):
    monkeypatch.setattr(
        pattern_match_deployments,
        "get_llm_provider",
        lambda model, **kwargs: (model, "openai", None, None),
    )
    router = PatternMatchRouter()
    router.add_pattern("openai/*", _wildcard_deployment("openai/*"))
    assert _matched_models(router.get_pattern("gpt-4o")) == ["openai/gpt-4o"]


class _CountingPatternUtils(PatternUtils):
    sorted_patterns = staticmethod(Mock(wraps=PatternUtils.sorted_patterns))


def test_route_never_sorts_and_the_most_specific_pattern_still_wins_after_registry_changes():
    """Regression for LIT-6886: the auth layer walks the wildcard registry for every request, so an
    unmatched model name (an invalid-model 403) re-sorted every pattern by specificity per request and
    a burst of rejections saturated the worker CPU. Lookups must not sort; adding a pattern or removing
    a deployment must still leave the most specific pattern winning."""
    router = PatternMatchRouter(pattern_utils=_CountingPatternUtils)
    router.add_pattern("openai/*", _wildcard_deployment("openai/*"))
    router.add_pattern("anthropic/*", _wildcard_deployment("anthropic/*"))
    router.add_pattern("openai/gpt-*", {"model_name": "openai/gpt-*", "litellm_params": {"model": "azure/gpt-*"}})
    sorts_after_setup = _CountingPatternUtils.sorted_patterns.call_count

    for _ in range(3):
        assert router.route("does-not-exist") is None
    assert _matched_models(router.route("openai/gpt-4o")) == ["azure/gpt-4o"]
    assert _matched_models(router.route("openai/o3")) == ["openai/o3"]
    assert _CountingPatternUtils.sorted_patterns.call_count == sorts_after_setup

    router.add_pattern("openai/*", {**_wildcard_deployment("openai/*"), "model_info": {"id": "id-1"}})
    assert len(_matched_models(router.route("openai/o3"))) == 2
    router.remove_deployment("id-1")
    assert _matched_models(router.route("openai/gpt-4o")) == ["azure/gpt-4o"]
    assert _matched_models(router.route("openai/o3")) == ["openai/o3"]
