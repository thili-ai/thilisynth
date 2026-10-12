from thilisynth.renderers.ollama import OllamaRenderer


def test_the_default_is_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert OllamaRenderer().host == "http://127.0.0.1:11434"


def test_ollama_host_is_used_when_set(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://host.docker.internal:11434/")
    assert OllamaRenderer().host == "http://host.docker.internal:11434"


def test_a_bare_host_and_port_gets_a_scheme(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.20:11434")
    assert OllamaRenderer().host == "http://192.168.1.20:11434"


def test_an_explicit_host_beats_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://elsewhere:1")
    assert OllamaRenderer(host="http://127.0.0.1:9999").host == "http://127.0.0.1:9999"
