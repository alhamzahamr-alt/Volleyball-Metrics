"""Setup checks shared by doctor.py (command line, run by start.bat before
anything launches) and the API (GET /api/system/status for the frontend's
banner, plus process_job's refusal to start a run that can't finish).

Two things are checked:

- The GPU: whether the installed PyTorch is a CUDA build at all, whether it
  can actually see and run kernels on a GPU, and whether onnxruntime offers
  CUDA for the player ReID encoder. Every stage silently falls back to CPU
  when CUDA isn't usable, so without this check a CPU-only torch install
  just looks like "processing is very slow" rather than an error.
- The model files each stage loads. Most are gitignored (see ../../.gitignore),
  so a fresh checkout has none of them - and a missing one otherwise only
  surfaces as a traceback once its stage is reached, which for
  ball_detection is after player_tracking (the slowest stage) has already
  run to completion.
"""

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
REPO_DIR = BACKEND_DIR.parent
ANALYSIS_DIR = BACKEND_DIR / "Analysis"
for directory in (BACKEND_DIR, ANALYSIS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

REQUIREMENTS_FILE = BACKEND_DIR / "requirements.txt"
CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu132"

# Files a VideoMAE checkpoint folder needs for gameStatusDetection's
# from_pretrained calls (model + processor) and its own
# _patch_legacy_attention_bias, which reads model.safetensors directly.
VIDEOMAE_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")

GPU_PROBE_TIMEOUT_S = 120


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_DIR).as_posix()
    except ValueError:
        return str(path)


def _pinned_version(package: str) -> Optional[str]:
    """The exact version requirements.txt pins for `package`, so fix-up
    commands reinstall the same build everything else was tested against."""
    try:
        lines = REQUIREMENTS_FILE.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    for line in lines:
        name, sep, version = line.strip().partition("==")
        if sep and name.lower() == package.lower():
            return version
    return None


# --- GPU ---------------------------------------------------------------------


def _empty_gpu_result(error: Optional[str] = None) -> dict:
    return {
        "torch_version": None,
        "cuda_build": None,
        "cuda_available": False,
        "device_name": None,
        "vram_gb": None,
        "kernels_ok": False,
        "onnx_cuda": False,
        "error": error,
    }


def probe_gpu() -> dict:
    """Runs in-process. The API calls gpu_status() instead, which runs this
    in a throwaway subprocess - see its docstring."""
    result = _empty_gpu_result()
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - any import failure means the same thing here
        result["error"] = f"PyTorch could not be imported: {exc}"
        return result

    result["torch_version"] = torch.__version__
    result["cuda_build"] = torch.version.cuda
    try:
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            result["device_name"] = props.name
            result["vram_gb"] = round(props.total_memory / 1024**3, 1)
            # is_available() only proves a driver and a device exist. A wheel
            # built without kernels for this GPU's architecture (a new card on
            # an older torch) still fails on its first real op, so run one.
            x = torch.ones(8, 8, device="cuda")
            (x @ x).sum().item()
            result["kernels_ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"CUDA is present but unusable: {exc}"

    try:
        import onnxruntime

        result["onnx_cuda"] = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:  # noqa: BLE001 - onnx is only needed for ReID; reported as onnx_cuda=False
        pass

    return result


@lru_cache(maxsize=1)
def gpu_status() -> dict:
    """probe_gpu(), run once per API process in a subprocess of the same
    interpreter the pipeline stages use (see pipeline._run_stage_process).
    Probing in-process would leave a CUDA context open in the API server for
    its whole lifetime - a few hundred MB of VRAM the stage subprocesses
    then can't have."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "API.services.system_check", "gpu"],
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
            timeout=GPU_PROBE_TIMEOUT_S,
        )
        stdout_lines = completed.stdout.strip().splitlines()
        if completed.returncode != 0 or not stdout_lines:
            stderr_lines = completed.stderr.strip().splitlines()
            detail = stderr_lines[-1] if stderr_lines else f"exit code {completed.returncode}"
            return _empty_gpu_result(f"GPU check crashed: {detail}")
        return json.loads(stdout_lines[-1])
    except Exception as exc:  # noqa: BLE001 - reported to the user as a failed check, never raised
        return _empty_gpu_result(f"GPU check failed to run: {exc}")


def gpu_problems(gpu: dict) -> list[dict]:
    pip = r".venv\Scripts\pip" if sys.platform == "win32" else ".venv/bin/pip"
    torch_pin = _pinned_version("torch")
    vision_pin = _pinned_version("torchvision")
    torch_spec = f"torch=={torch_pin}" if torch_pin else "torch"
    vision_spec = f"torchvision=={vision_pin}" if vision_pin else "torchvision"
    reinstall_torch = (
        f"{pip} install --force-reinstall --no-deps {torch_spec} {vision_spec} "
        f"--extra-index-url {CUDA_WHEEL_INDEX}"
    )

    if gpu["torch_version"] is None:
        return [{
            "severity": "error",
            "message": gpu["error"] or "PyTorch isn't installed.",
            "fix": f"From Backend/: {pip} install -r requirements.txt",
        }]
    if gpu["cuda_build"] is None:
        return [{
            "severity": "warning",
            "message": (f"PyTorch {gpu['torch_version']} is the CPU-only build, so the GPU is never used "
                        "and processing will be very slow."),
            "fix": f"From Backend/: {reinstall_torch}",
        }]
    if not gpu["cuda_available"]:
        return [{
            "severity": "warning",
            "message": (f"PyTorch was built for CUDA {gpu['cuda_build']} but can't see a GPU, so everything "
                        "runs on the CPU."),
            "fix": "Install or update the NVIDIA driver - `nvidia-smi` should list your GPU.",
        }]
    if not gpu["kernels_ok"]:
        return [{
            "severity": "warning",
            "message": f"Found {gpu['device_name']}, but a test run on it failed: {gpu['error']}",
            "fix": ("Usually this PyTorch build doesn't support this GPU generation - update the NVIDIA "
                    f"driver, then reinstall PyTorch from Backend/: {reinstall_torch}"),
        }]

    problems = []
    if not gpu["onnx_cuda"]:
        ort_pin = _pinned_version("onnxruntime-gpu")
        ort_spec = f"onnxruntime-gpu=={ort_pin}" if ort_pin else "onnxruntime-gpu"
        problems.append({
            "severity": "warning",
            "message": "onnxruntime has no CUDA support, so player re-identification runs on the CPU.",
            "fix": (f"From Backend/: {pip} uninstall -y onnxruntime onnxruntime-gpu, then "
                    f"{pip} install {ort_spec}"),
        })
    return problems


# --- Model files -------------------------------------------------------------


def _videomae_present(directory: Path) -> bool:
    return all((directory / name).is_file() for name in VIDEOMAE_FILES)


def _auto_download_location(name: str) -> Optional[Path]:
    """Where Ultralytics would find an already-downloaded `name` - the
    stage's working directory (Backend/, see pipeline._run_stage_process)
    first, then its own weights_dir, the same order
    ultralytics.utils.downloads.attempt_download_asset checks."""
    candidates = [BACKEND_DIR / name]
    try:
        from ultralytics.utils import SETTINGS

        candidates.append(Path(SETTINGS["weights_dir"]) / name)
    except Exception:  # noqa: BLE001
        pass
    return next((path for path in candidates if path.is_file()), None)


def model_status() -> list[dict]:
    """One entry per model file the pipeline loads. Paths come from the
    stage modules' own constants, so this never drifts from what they
    actually open. `required` means the stage has no fallback and fails
    without it; `auto_download` means Ultralytics fetches it on first use
    (needs internet once)."""
    from ActionDetection import actionDetection
    from BallDetection import ballDetection
    from GameStatusDetection import gameStatusDetection
    from PlayerDetection import tracker

    custom_dir = gameStatusDetection.CUSTOM_MODEL_DIR
    base_dir = gameStatusDetection.BASE_MODEL_DIR
    game_status_dir = custom_dir if _videomae_present(custom_dir) else base_dir
    entries = [
        {
            "key": "game_status",
            "label": "Rally classifier (VideoMAE)",
            "stage": "game_status",
            "required": True,
            "auto_download": False,
            "present": _videomae_present(game_status_dir),
            "path": _display_path(game_status_dir),
            "how_to_get": ("Copy the fine-tuned VideoMAE checkpoint folder from masouduut94/volleyball_analytics "
                           f"(it needs {', '.join(VIDEOMAE_FILES)}) into place."),
        },
        {
            "key": "ball_primary",
            "label": "Ball detector (YOLO26x)",
            "stage": "ball_detection",
            "required": True,
            "auto_download": False,
            "present": Path(ballDetection.MODEL_PATH).is_file(),
            "path": _display_path(Path(ballDetection.MODEL_PATH)),
            "how_to_get": ("Copy your trained weights into place, or train with "
                           "`python MachineLearning/mainTrainingModels.py ball` and rename its best.pt."),
        },
        {
            "key": "ball_secondary",
            "label": "Ball detector (YOLO11x)",
            "stage": "ball_detection",
            "required": True,
            "auto_download": False,
            "present": Path(ballDetection.SECONDARY_MODEL_PATH).is_file(),
            "path": _display_path(Path(ballDetection.SECONDARY_MODEL_PATH)),
            "how_to_get": "Copy your trained weights into place (same training script as the YOLO26x detector).",
        },
        {
            "key": "action_detector",
            "label": "Action detector",
            "stage": "action_detection",
            "required": False,
            "auto_download": False,
            "present": Path(actionDetection.ACTION_DETECTOR_PATH).is_file(),
            "path": _display_path(Path(actionDetection.ACTION_DETECTOR_PATH)),
            "how_to_get": ("Without it every hit falls back to the geometric heuristic. Train one with "
                           "`python MachineLearning/mainTrainingModels.py action`."),
        },
    ]
    for key, label, name in (
        ("player_detector", "Player detector", tracker.MODEL_PATH),
        ("player_reid", "Player re-identification", tracker.REID_MODEL_NAME),
    ):
        found = _auto_download_location(name)
        entries.append({
            "key": key,
            "label": label,
            "stage": "player_tracking",
            "required": False,
            "auto_download": True,
            "present": found is not None,
            "path": _display_path(found or BACKEND_DIR / name),
            "how_to_get": "Downloaded automatically by Ultralytics on the first run (needs internet once).",
        })
    return entries


def model_problems(models: list[dict]) -> list[dict]:
    problems = []
    for model in models:
        if model["present"]:
            continue
        if model["required"]:
            problems.append({
                "severity": "error",
                "message": f"{model['label']} is missing, so the {model['stage']} stage will fail.",
                "fix": f"{model['how_to_get']} Expected at: {model['path']}",
            })
        elif model["auto_download"]:
            problems.append({
                "severity": "info",
                "message": f"{model['label']} ({Path(model['path']).name}) will be downloaded on the first run.",
                "fix": model["how_to_get"],
            })
        else:
            problems.append({
                "severity": "info",
                "message": f"{model['label']} not found (optional).",
                "fix": f"{model['how_to_get']} Expected at: {model['path']}",
            })
    return problems


def missing_required_models() -> list[dict]:
    return [model for model in model_status() if model["required"] and not model["present"]]


def preflight_error() -> Optional[str]:
    """A user-facing reason a full pipeline run can't succeed right now, or
    None. Checked before a run is queued, so it fails in a second instead
    of minutes into the run."""
    missing = missing_required_models()
    if not missing:
        return None
    listed = "; ".join(f"{model['label']} ({model['path']})" for model in missing)
    return (f"Can't start processing - required model files are missing: {listed}. "
            "Run `python doctor.py` in Backend/ for how to get them.")


def status() -> dict:
    gpu = gpu_status()
    models = model_status()
    problems = gpu_problems(gpu) + model_problems(models)
    return {
        "gpu": gpu,
        "models": models,
        "problems": problems,
        "ready": not any(problem["severity"] == "error" for problem in problems),
    }


if __name__ == "__main__":
    # `python -m API.services.system_check gpu` - the subprocess gpu_status()
    # runs. Last stdout line is the JSON result; anything a library prints
    # while importing lands above it.
    if sys.argv[1:] == ["gpu"]:
        print(json.dumps(probe_gpu()))
