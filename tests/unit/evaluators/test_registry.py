"""Unit tests for evaluators.registry.

Covers EVALUATOR_REGISTRY dict and load_evaluators() function.
All evaluator constructors are mocked so no real models are loaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_evaluator_registry_has_all_seven_evaluators():
    from sentinel.evaluators.registry import EVALUATOR_REGISTRY

    assert len(EVALUATOR_REGISTRY) == 7
    assert "pii" in EVALUATOR_REGISTRY
    assert "prompt_injection" in EVALUATOR_REGISTRY
    assert "topic_guardrail" in EVALUATOR_REGISTRY
    assert "toxicity" in EVALUATOR_REGISTRY
    assert "relevance" in EVALUATOR_REGISTRY
    assert "hallucination" in EVALUATOR_REGISTRY
    assert "faithfulness" in EVALUATOR_REGISTRY


def test_load_evaluators_empty_config_returns_empty_list():
    """No evaluators key in config → all disabled → empty list."""
    from sentinel.evaluators.registry import load_evaluators

    result = load_evaluators({})
    assert result == []


def test_load_evaluators_all_explicitly_disabled():
    """All evaluators with enabled=False → empty list."""
    from sentinel.evaluators.registry import EVALUATOR_REGISTRY, load_evaluators

    config = {"evaluators": {name: {"enabled": False} for name in EVALUATOR_REGISTRY}}
    result = load_evaluators(config)
    assert result == []


def test_load_evaluators_one_enabled():
    """Enabling one evaluator instantiates it with the full config."""
    from sentinel.evaluators.registry import EVALUATOR_REGISTRY, load_evaluators

    mock_instance = MagicMock()
    mock_cls = MagicMock(return_value=mock_instance)
    config = {"evaluators": {"pii": {"enabled": True}}}

    # Patch the dict entry directly since EVALUATOR_REGISTRY is built at import time
    with patch.dict(EVALUATOR_REGISTRY, {"pii": mock_cls}):
        result = load_evaluators(config)

    assert len(result) == 1
    assert result[0] is mock_instance
    mock_cls.assert_called_once_with(config)


def test_load_evaluators_all_enabled():
    """All seven evaluators enabled → seven instances returned in registry order."""
    from sentinel.evaluators.registry import EVALUATOR_REGISTRY, load_evaluators

    ev_names = list(EVALUATOR_REGISTRY.keys())
    mock_ev = MagicMock()
    mock_registry = {name: MagicMock(return_value=mock_ev) for name in ev_names}
    config = {"evaluators": {name: {"enabled": True} for name in ev_names}}

    with patch.dict(EVALUATOR_REGISTRY, mock_registry):
        result = load_evaluators(config)

    assert len(result) == 7


def test_load_evaluators_preserves_registry_order():
    """Returned evaluators follow the dict insertion order of EVALUATOR_REGISTRY."""
    from sentinel.evaluators.registry import EVALUATOR_REGISTRY, load_evaluators

    ev_names = list(EVALUATOR_REGISTRY.keys())
    instances = {name: MagicMock() for name in ev_names}
    mock_registry = {name: MagicMock(return_value=instances[name]) for name in ev_names}
    config = {"evaluators": {name: {"enabled": True} for name in ev_names}}

    with patch.dict(EVALUATOR_REGISTRY, mock_registry):
        result = load_evaluators(config)

    assert result == [instances[n] for n in ev_names]
