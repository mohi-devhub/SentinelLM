"""Unit tests for pure helper functions in sentinel.api.proxy.

No app/DB/Redis fixtures needed — these are plain functions over dicts.
"""

from __future__ import annotations

from sentinel.api.proxy import _UNTRUSTED_BOUNDARY_INSTRUCTION, _wrap_untrusted_content


def test_wraps_user_message_content():
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    wrapped = _wrap_untrusted_content(messages)

    assert wrapped[-1]["role"] == "user"
    assert wrapped[-1]["content"] == (
        "<untrusted_user_input>\nWhat is the capital of France?\n</untrusted_user_input>"
    )


def test_adds_system_message_when_none_present():
    messages = [{"role": "user", "content": "hi"}]
    wrapped = _wrap_untrusted_content(messages)

    assert wrapped[0]["role"] == "system"
    assert wrapped[0]["content"] == _UNTRUSTED_BOUNDARY_INSTRUCTION
    assert wrapped[1]["role"] == "user"


def test_appends_to_existing_system_message_instead_of_replacing_it():
    messages = [
        {"role": "system", "content": "You are a helpful support assistant."},
        {"role": "user", "content": "hi"},
    ]
    wrapped = _wrap_untrusted_content(messages)

    assert len(wrapped) == 2
    assert wrapped[0]["role"] == "system"
    assert wrapped[0]["content"].startswith("You are a helpful support assistant.")
    assert _UNTRUSTED_BOUNDARY_INSTRUCTION in wrapped[0]["content"]


def test_does_not_wrap_assistant_or_system_messages():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "prior reply"},
        {"role": "user", "content": "follow-up"},
    ]
    wrapped = _wrap_untrusted_content(messages)

    assistant_msg = next(m for m in wrapped if m["role"] == "assistant")
    assert assistant_msg["content"] == "prior reply"


def test_wraps_every_user_turn_in_multi_turn_conversation():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    wrapped = _wrap_untrusted_content(messages)

    user_msgs = [m for m in wrapped if m["role"] == "user"]
    assert all(m["content"].startswith("<untrusted_user_input>") for m in user_msgs)


def test_does_not_mutate_input_messages():
    messages = [{"role": "user", "content": "hi"}]
    _wrap_untrusted_content(messages)

    assert messages[0]["content"] == "hi"
