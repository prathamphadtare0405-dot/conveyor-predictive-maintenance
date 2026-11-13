"""Start backend, simulator and dashboard together for a demo.

    python run.py                 # 60x time acceleration, one bearing fault injected
    python run.py --speedup 1     # real time
    python run.py --no-inject     # random faults only

Stop everything with Ctrl+C. Each part can also be started on its own; see the README.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import requests

from backend.config import PROJECT_ROOT, settings
from ml.train import model_path


def wait_for_backend(url: str, process: subprocess.Popen, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            if requests.get(f"{url}/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--speedup", type=float, default=60.0, help="simulator time acceleration (default %(default)s)")
    parser.add_argument("--inject", action="append", default=None, metavar="MODE:SENSOR:HOURS",
                        help="fault to start at once (default: bearing_wear:S04:3). Repeatable.")
    parser.add_argument("--no-inject", action="store_true", help="do not inject a fault; rely on random ones")
    args = parser.parse_args()

    if not model_path().exists():
        print(
            "No trained model found. Anomalies and predictions will be empty until you run:\n"
            "  python -m simulator.sensor_simulator --backfill 48\n"
            "  python -m ml.train\n"
        )

    python = sys.executable
    url = f"http://{settings.backend_host}:{settings.backend_port}"
    injections = [] if args.no_inject else (args.inject or ["bearing_wear:S04:3"])
    processes: list[tuple[str, subprocess.Popen]] = []

    def start(name: str, command: list[str]) -> subprocess.Popen:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT)
        processes.append((name, process))
        return process

    try:
        backend = start(
            "backend",
            [python, "-m", "uvicorn", "backend.main:app", "--host", settings.backend_host, "--port", str(settings.backend_port), "--log-level", "warning"],
        )
        if not wait_for_backend(url, backend):
            print(f"The backend did not start on {url}. Is the port already in use?")
            return 1
        print(f"Backend:   {url}   (API docs: {url}/docs)")

        simulator = [python, "-m", "simulator.sensor_simulator", "--speedup", str(args.speedup), "--backend-url", url]
        for spec in injections:
            simulator += ["--inject", spec]
        start("simulator", simulator)
        print(f"Simulator: {args.speedup:g}x time acceleration" + (f", injected {', '.join(injections)}" if injections else ""))

        start("dashboard", [python, "-m", "streamlit", "run", "dashboard/app.py", "--server.headless", "true", "--server.address", "localhost"])
        print("Dashboard: http://localhost:8501")
        print("Ctrl+C stops all three.\n")

        while True:
            for name, process in processes:
                if process.poll() is not None:
                    print(f"{name} stopped (exit code {process.returncode}); shutting the rest down.")
                    return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping ...")
        return 0
    finally:
        for _name, process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for _name, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
