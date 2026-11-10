"""Static checks on the container files: what must never change without a reason."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def lines(name):
    return [x.strip() for x in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def test_secrets_and_local_data_stay_out_of_the_image():
    ignored = lines(".dockerignore")
    for entry in (".env", "storage/", "*.docx", ".git/"):
        assert entry in ignored
    assert "!.env.example" in ignored


def test_the_dashboard_is_published_on_localhost_only():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"127.0.0.1:8501:8501"' in compose
    assert "11434:" not in compose  # Ollama is not published on the host


def test_the_image_runs_as_an_unprivileged_user_with_cpu_torch():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER app" in dockerfile and "HEALTHCHECK" in dockerfile
    assert "download.pytorch.org/whl/cpu" in dockerfile
