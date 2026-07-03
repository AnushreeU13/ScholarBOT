from app import llm


def test_complete_returns_empty_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm.complete("system", "prompt", 10, "gpt-4o-mini") == ""


def test_complete_returns_empty_on_client_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class BrokenClient:
        def __init__(self):
            raise RuntimeError("network down")

    monkeypatch.setattr("openai.OpenAI", BrokenClient)
    assert llm.complete("system", "prompt", 10, "gpt-4o-mini") == ""


def test_complete_returns_content_on_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeMessage:
        content = "hello world"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    monkeypatch.setattr("openai.OpenAI", lambda: FakeClient())
    assert llm.complete("system", "prompt", 10, "gpt-4o-mini") == "hello world"


def test_parse_json_direct():
    assert llm.parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_extracts_embedded_object():
    assert llm.parse_json('Here is the answer: {"a": 1} — done') == {"a": 1}


def test_parse_json_returns_empty_on_garbage():
    assert llm.parse_json("not json at all") == {}
