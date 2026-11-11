from __future__ import annotations

import subprocess
import sys
import time
import webbrowser

from LUMIA.core.concurrency import configure_global_runtime


def main() -> None:
    print("Starting Project LUMIA development environment...")
    configure_global_runtime()

    # 1. FastAPI Backend
    backend_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "LUMIA.api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ]
    )

    # 2. Vite Frontend Development Server
    print("Launching React/Vite frontend (Port 5173)...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend",
        shell=sys.platform == "win32",
    )

    # Allow servers to initialize
    print("Waiting for server initialization (3 seconds)...")
    time.sleep(3)

    # 3. Open Browser
    url = "http://localhost:5173"
    print(f"Opening browser at: {url}")
    webbrowser.open(url)

    print("\nProject LUMIA is active.")
    print("Press Ctrl+C in this terminal to terminate all processes.\n")

    try:
        backend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()