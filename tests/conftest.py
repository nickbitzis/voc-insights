from types import SimpleNamespace

import pytest

from vocinsights.pipeline import build


@pytest.fixture(scope="session")
def bench():
    """A full pipeline on 8,000 texts with the simulated LLM tier."""
    return build(n=8000, seed=7, discovery_sample=3000)


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        text, stop = (r, "end_turn") if isinstance(r, str) else r
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=520, output_tokens=90),
            stop_reason=stop,
            model=kw["model"],
        )


class FakeClient:
    def __init__(self, *replies):
        self.messages = FakeMessages(replies)


@pytest.fixture
def fake_client():
    return FakeClient
