"""The application starts: CLI check and Streamlit page render without errors."""

import os
import subprocess
import sys

from streamlit.testing.v1 import AppTest

from app.config import PROJECT_ROOT, Settings

SETTING_VARIABLES = [name.upper() for name in Settings.model_fields]


def clean_env(**extra) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in SETTING_VARIABLES}
    return {**env, **extra}


def test_cli_reports_a_valid_configuration(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "app.main"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        env=clean_env(STORAGE_DIR=str(tmp_path / "storage")), timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "configuration OK" in result.stdout
    assert (tmp_path / "storage" / "documents").is_dir()


def test_cli_fails_clearly_on_invalid_configuration():
    result = subprocess.run(
        [sys.executable, "-m", "app.main"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
        env=clean_env(LLM_PROVIDER="anthropic"), timeout=60,
    )
    assert result.returncode == 1
    assert "ALLOW_PAID_APIS" in result.stderr


def test_dashboard_renders(monkeypatch):
    for name in SETTING_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    app = AppTest.from_file(str(PROJECT_ROOT / "frontend" / "dashboard.py"), default_timeout=30)
    app.run()
    assert not app.exception
    assert app.title[0].value == "DeepResearch AI"
    assert any("Configuration loaded" in s.value for s in app.success)
