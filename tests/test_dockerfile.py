
from pathlib import Path


DOCKERFILE = Path("Dockerfile").read_text(encoding="utf-8")


def test_base_image_and_port():
    assert "FROM python:3.12-slim" in DOCKERFILE
    assert "EXPOSE 8000" in DOCKERFILE
    assert "--host" in DOCKERFILE
    assert "0.0.0.0" in DOCKERFILE
    assert "8000" in DOCKERFILE


def test_requirements_are_installed_before_code_is_copied():
    assert DOCKERFILE.index("COPY requirements.txt") < DOCKERFILE.index("pip install")
    assert DOCKERFILE.index("pip install") < DOCKERFILE.index("COPY app")


def test_dockerignore_keeps_data_and_venv_out():
    ignore = Path(".dockerignore").read_text(encoding="utf-8")
    assert ".venv" in ignore
    assert "data/" in ignore
