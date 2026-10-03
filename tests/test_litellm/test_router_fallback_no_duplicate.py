"""Regressions for duplicate dispatches in nested order and external fallbacks."""

from unittest.mock import patch

import pytest

from litellm import Router


@pytest.mark.asyncio
async def test_order_failures_do_not_retry_external_fallback_twice():
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 1"),
                    "order": 1,
                },
                "model_info": {"id": "1"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail order 2"),
                    "order": 2,
                },
                "model_info": {"id": "2"},
            },
            {
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "bad",
                    "mock_response": Exception("fail external fallback"),
                },
                "model_info": {"id": "fallback"},
            },
        ],
        fallbacks=[{"test-model": ["fallback-model"]}],
        num_retries=0,
    )
    attempted_groups = []

    async def fail_dispatch(original_function, *args, **kwargs):
        attempted_groups.append(kwargs.get("model"))
        raise Exception("synthetic dispatch failure")

    with patch.object(router, "make_call", side_effect=fail_dispatch):
        with pytest.raises(Exception):
            await router.acompletion(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
            )

    assert attempted_groups.count("fallback-model") == 1
