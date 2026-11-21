"""Start and stop a local llama.cpp server for one model configuration."""

import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from inklab.constants import (
    CONTEXT_PER_SLOT,
    DEFAULT_N_CPU_MOE,
    DEFAULT_PARALLEL,
    LOGICAL_BATCH,
    LOOP_COUNT_KEY,
    MICRO_BATCH,
    SERVER_EXE,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_READY_TIMEOUT_S,
)
from inklab.errors import ServerError


@dataclass(frozen=True)
class ServerConfig:
    """Settings for one llama-server process.

    Attributes:
        model: GGUF language model file.
        mmproj: GGUF vision projector file, or None for text only.
        n_cpu_moe: Number of layers whose expert weights stay in CPU RAM.
        parallel: Number of request slots that run at the same time.
        loop_passes: Override for the number of loop passes, or None for the file default.
        port: TCP port on localhost.
    """

    model: Path
    mmproj: Path | None = None
    n_cpu_moe: int = DEFAULT_N_CPU_MOE
    parallel: int = DEFAULT_PARALLEL
    loop_passes: int | None = None
    port: int = SERVER_PORT

    def args(self) -> list[str]:
        """Return the command line for llama-server."""
        args = [
            str(SERVER_EXE), "-m", str(self.model),
            "-ngl", "99", "--n-cpu-moe", str(self.n_cpu_moe),
            "-c", str(CONTEXT_PER_SLOT * self.parallel), "-np", str(self.parallel),
            "-ub", str(MICRO_BATCH), "-b", str(LOGICAL_BATCH),
            "--host", SERVER_HOST, "--port", str(self.port),
        ]
        if self.mmproj is not None:
            args += ["--mmproj", str(self.mmproj)]
        if self.loop_passes is not None:
            args += ["--override-kv", f"{LOOP_COUNT_KEY}=int:{self.loop_passes}"]
        return args


class LlamaServer:
    """Run llama-server as a child process; use it as a context manager."""

    def __init__(self, config: ServerConfig, log_path: Path) -> None:
        self.config = config
        self.log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        """Return the HTTP base URL of the server."""
        return f"http://{SERVER_HOST}:{self.config.port}"

    def start(self) -> None:
        """Start the process and block until the health endpoint answers.

        Raises:
            ServerError: If the process exits or is not ready in time.
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("wb")
        self._process = subprocess.Popen(self.config.args(), stdout=log, stderr=subprocess.STDOUT)
        self._wait_ready()

    def stop(self) -> None:
        """Stop the process if it runs."""
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + SERVER_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise ServerError(f"llama-server exited with code {self._process.returncode}; see {self.log_path.name}")
            if self._healthy():
                return
            time.sleep(2.0)
        self.stop()
        raise ServerError(f"llama-server was not ready after {SERVER_READY_TIMEOUT_S:.0f} s")

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return False

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.stop()
