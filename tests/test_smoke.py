from __future__ import annotations

"""Smoke tests for QuenStar — requires a GGUF model and GPU to run."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"


def _find_model():
    """Find any valid GGUF model in the models/ directory."""
    if not MODEL_DIR.is_dir():
        return None
    for p in sorted(MODEL_DIR.glob("*.gguf"), key=lambda x: x.stat().st_size, reverse=True):
        if p.stat().st_size > 1024 * 1024:  # at least 1 MB
            return str(p)
    return None


def _venv_python():
    """Return the path to the .venv python, or sys.executable."""
    venv = PROJECT_DIR / ".venv" / "bin" / "python"
    if venv.is_file():
        return str(venv)
    return sys.executable


def _set_env():
    """Return env with CUDA library path set for ollama-bundled CUDA."""
    env = os.environ.copy()
    cuda_paths = [
        "/usr/local/lib/ollama/cuda_v12",
        "/usr/local/lib/ollama/cuda_v13",
    ]
    for p in cuda_paths:
        if os.path.isdir(p):
            env["LD_LIBRARY_PATH"] = p + ":" + env.get("LD_LIBRARY_PATH", "")
            break
    return env


# ── Tests ──────────────────────────────────────────────────────────


class TestCLI:
    """Test the standalone CLI smoke-test tool."""

    @pytest.mark.slow
    def test_cli_loads_and_generates(self):
        """Model loads and produces output tokens."""
        model = _find_model()
        if not model:
            pytest.skip("No GGUF model found in models/")

        result = subprocess.run(
            [
                _venv_python(), "-m", "quenstar.cli",
                "-m", model,
                "--ctx", "512",
                "--max-tokens", "5",
                "--temp", "0",
                "--prompt", "Say hi",
            ],
            capture_output=True, text=True, timeout=120,
            env=_set_env(),
        )

        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert "Assistant:" in result.stdout, f"No output:\n{result.stdout}"
        assert "tok/s:" in result.stdout, f"No speed stats:\n{result.stdout}"

    @pytest.mark.slow
    def test_cli_rejects_bad_file(self):
        """CLI exits non-zero for invalid GGUF."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"not a model")
            bad_path = f.name

        try:
            result = subprocess.run(
                [_venv_python(), "-m", "quenstar.cli", "-m", bad_path, "--ctx", "512"],
                capture_output=True, text=True, timeout=30,
                env=_set_env(),
            )
            assert result.returncode != 0
        finally:
            os.unlink(bad_path)

    @pytest.mark.slow
    def test_cli_rejects_missing_file(self):
        """CLI exits non-zero for missing file."""
        result = subprocess.run(
            [_venv_python(), "-m", "quenstar.cli", "-m", "/nonexistent/path.gguf", "--ctx", "512"],
            capture_output=True, text=True, timeout=30,
            env=_set_env(),
        )
        assert result.returncode != 0


class TestInteractiveCLI:
    """Test the interactive CLI chat mode."""

    @pytest.mark.slow
    def test_interactive_chat_generates_and_quits(self):
        """Interactive mode loads model, responds to prompt, and quits cleanly."""
        model = _find_model()
        if not model:
            pytest.skip("No GGUF model found in models/")

        env = _set_env()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            [
                _venv_python(), "-m", "quenstar.cli",
                "-m", model,
                "--ctx", "512",
                "--max-tokens", "10",
                "--temp", "0",
                "-i",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            stdout, stderr = proc.communicate(input="Say hi\n/quit\n", timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            pytest.fail(f"Interactive CLI timed out\nstdout:\n{stdout}\nstderr:\n{stderr}")

        assert proc.returncode == 0, f"CLI exited {proc.returncode}:\n{stderr}"
        assert "Assistant:" in stdout, f"No assistant response:\n{stdout}"
        assert "tok/s:" in stdout or "tok/s" in stdout, f"No speed stats:\n{stdout}"

    @pytest.mark.slow
    def test_interactive_rejects_bad_file(self):
        """Interactive CLI exits non-zero for invalid GGUF."""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            f.write(b"not a model")
            bad_path = f.name

        try:
            result = subprocess.run(
                [_venv_python(), "-m", "quenstar.cli", "-m", bad_path, "--ctx", "512", "-i"],
                capture_output=True, text=True, timeout=30,
                env=_set_env(),
            )
            assert result.returncode != 0
        finally:
            os.unlink(bad_path)

    @pytest.mark.slow
    def test_interactive_rejects_missing_file(self):
        """Interactive CLI exits non-zero for missing file."""
        result = subprocess.run(
            [_venv_python(), "-m", "quenstar.cli", "-m", "/nonexistent/path.gguf", "--ctx", "512", "-i"],
            capture_output=True, text=True, timeout=30,
            env=_set_env(),
        )
        assert result.returncode != 0


class TestServer:
    """Test the HTTP server endpoints."""

    @staticmethod
    def _start_server(model_path, ctx=512, port=18990):
        """Start the server and return the process. Caller must terminate."""
        env = _set_env()
        env["QUENSTAR_LOG_LEVEL"] = "WARNING"
        proc = subprocess.Popen(
            [
                _venv_python(), "-m", "quenstar",
                "-m", model_path,
                "--ctx", str(ctx),
                "--port", str(port),
                "--host", "127.0.0.1",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env,
        )
        # wait for server to be ready
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                import urllib.request
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            proc.terminate()
            proc.wait()
            pytest.fail("Server did not become ready within 60s")
        return proc

    @staticmethod
    def _fetch(url, method="GET", body=None):
        import urllib.error
        import urllib.request

        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    @pytest.mark.slow
    def test_health_endpoint(self):
        model = _find_model()
        if not model:
            pytest.skip("No GGUF model found in models/")

        port = 18991
        proc = self._start_server(model, port=port)
        try:
            status, body = self._fetch(f"http://127.0.0.1:{port}/health")
            assert status == 200
            data = json.loads(body)
            assert data["status"] == "ok"
            assert "model" in data
            assert "n_ctx" in data
        finally:
            proc.terminate()
            proc.wait()

    @pytest.mark.slow
    def test_models_endpoint(self):
        model = _find_model()
        if not model:
            pytest.skip("No GGUF model found in models/")

        port = 18992
        proc = self._start_server(model, port=port)
        try:
            status, body = self._fetch(f"http://127.0.0.1:{port}/v1/models")
            assert status == 200
            data = json.loads(body)
            assert data["object"] == "list"
            assert len(data["data"]) >= 1
        finally:
            proc.terminate()
            proc.wait()

    @pytest.mark.slow
    def test_chat_non_stream(self):
        model = _find_model()
        if not model:
            pytest.skip("No GGUF model found in models/")

        port = 18993
        proc = self._start_server(model, port=port)
        try:
            status, body = self._fetch(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                method="POST",
                body={
                    "model": "test",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "temperature": 0,
                    "max_tokens": 10,
                    "stream": False,
                },
            )
            assert status == 200, f"Status {status}: {body}"
            data = json.loads(body)
            assert "choices" in data
            assert len(data["choices"]) >= 1
            msg = data["choices"][0].get("message", {})
            assert "content" in msg or "tool_calls" in msg
        finally:
            proc.terminate()
            proc.wait()

    @pytest.mark.slow
    def test_chat_stream(self):
        model = _find_model()
        if not model:
            pytest.skip("No GGUF model found in models/")

        port = 18994
        proc = self._start_server(model, port=port)
        try:
            import urllib.request

            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=json.dumps({
                    "model": "test",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "temperature": 0,
                    "max_tokens": 5,
                    "stream": True,
                }).encode(),
                method="POST",
            )
            req.add_header("Content-Type", "application/json")

            chunks = []
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line in resp.read().decode().splitlines():
                    line = line.strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        chunks.append(json.loads(data_str))

            assert len(chunks) >= 1, "No streaming chunks received"
            # At least one chunk should have content
            contents = [
                c["choices"][0].get("delta", {}).get("content", "")
                for c in chunks
            ]
            assert any(c for c in contents), f"No content tokens in chunks: {contents}"
        finally:
            proc.terminate()
            proc.wait()

    @pytest.mark.slow
    def test_sessions_endpoint(self):
        model = _find_model()
        if not model:
            pytest.skip("No GGUF model found in models/")

        port = 18995
        proc = self._start_server(model, port=port)
        try:
            status, body = self._fetch(f"http://127.0.0.1:{port}/sessions")
            assert status == 200
            data = json.loads(body)
            assert "sessions" in data
        finally:
            proc.terminate()
            proc.wait()
