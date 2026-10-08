"""Checks this machine is ready to run the backend on its GPU, and prints
how to fix anything that isn't. start.bat runs it before launching
anything; run it by hand (from Backend/, with the venv's python) after
changing drivers, reinstalling packages or copying in model files.

Exit codes - start.bat keys off these:
    0  ready (possibly with warnings, e.g. running on CPU)
    1  the app starts, but processing a video will fail (missing models)
    2  the backend itself can't start (missing packages / broken PyTorch)
"""

import importlib.util
import sys

from API.services import system_check

MIN_PYTHON = (3, 11)
BACKEND_PACKAGES = ("fastapi", "uvicorn", "torch", "ultralytics", "transformers", "cv2", "onnxruntime")
MARKS = {"ok": "[ OK ]", "error": "[FAIL]", "warning": "[WARN]", "info": "[INFO]"}


def _line(mark: str, text: str):
    print(f"  {MARKS[mark]} {text}")


def check_python() -> list[dict]:
    print("Python")
    problems = []
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info[:2] < MIN_PYTHON:
        problems.append({"severity": "error", "message": f"Python {version} is too old.",
                         "fix": f"Install Python {'.'.join(map(str, MIN_PYTHON))}+ and recreate Backend/.venv."})
        _line("error", f"Python {version}")
    else:
        _line("ok", f"Python {version} ({sys.executable})")

    if sys.prefix == sys.base_prefix:
        problems.append({"severity": "warning", "message": "Not running inside a virtual environment.",
                         "fix": r"Run this with Backend\.venv\Scripts\python (Windows) or Backend/.venv/bin/python."})

    missing = [name for name in BACKEND_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        problems.append({"severity": "error", "message": f"Missing packages: {', '.join(missing)}.",
                         "fix": "From Backend/ (venv active): pip install -r requirements.txt"})
        _line("error", f"Missing packages: {', '.join(missing)}")
    else:
        _line("ok", "Backend packages installed")
    return problems


def check_gpu() -> list[dict]:
    print("GPU")
    gpu = system_check.probe_gpu()
    if gpu["kernels_ok"]:
        _line("ok", f"{gpu['device_name']} ({gpu['vram_gb']} GB) - PyTorch {gpu['torch_version']}, "
                    f"CUDA {gpu['cuda_build']}")
        _line("ok" if gpu["onnx_cuda"] else "warning",
              "onnxruntime CUDA provider " + ("available" if gpu["onnx_cuda"] else "missing"))
    else:
        _line("warning", "No usable GPU - everything will run on the CPU")
    return system_check.gpu_problems(gpu)


def check_models() -> list[dict]:
    print("Models")
    models = system_check.model_status()
    for model in models:
        if model["present"]:
            mark = "ok"
        elif model["required"]:
            mark = "error"
        else:
            mark = "info"
        suffix = "" if model["present"] else (" - downloads on first run" if model["auto_download"]
                                             else " - MISSING" if model["required"] else " - optional, not found")
        _line(mark, f"{model['label']}: {model['path']}{suffix}")
    return system_check.model_problems(models)


def main() -> int:
    print("Volleyball Metrics - setup check\n")
    problems = check_python()
    # Every later check imports torch/ultralytics; a missing package would
    # only turn into a traceback there.
    if any(problem["severity"] == "error" for problem in problems):
        return _report(problems, fatal=True)
    print()
    gpu_problems = check_gpu()
    problems += gpu_problems
    # The only GPU-check error is PyTorch failing to import (on Windows,
    # typically a DLL load failure) - the stage modules check_models imports
    # would just fail the same way.
    if any(problem["severity"] == "error" for problem in gpu_problems):
        return _report(problems, fatal=True)
    print()
    problems += check_models()
    return _report(problems, fatal=False)


def _report(problems: list[dict], fatal: bool) -> int:
    actionable = [problem for problem in problems if problem["severity"] in ("error", "warning")]
    print()
    if not actionable:
        print("Everything looks good.")
        return 0
    print("To fix:")
    for problem in actionable:
        print(f"  {MARKS[problem['severity']]} {problem['message']}")
        print(f"         -> {problem['fix']}")
    if fatal:
        return 2
    return 1 if any(problem["severity"] == "error" for problem in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
