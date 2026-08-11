"""Read-only ROCm/PyTorch and local-LLM configuration diagnostics."""

import importlib
import platform
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings
from app.llm.network import validate_local_llm_base_url


def collect_runtime_info(
    torch_loader: Callable[[str], Any] = importlib.import_module,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    """Collect safe facts without running shell commands or modifying the machine."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "utilities": {
            name: which(name) is not None
            for name in ("rocminfo", "rocm-smi", "amd-smi", "docker", "vllm")
        },
    }
    try:
        torch = torch_loader("torch")
    except Exception as exc:
        info["torch"] = {
            "importable": False,
            "warning": f"PyTorch unavailable: {type(exc).__name__}",
        }
        return info

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    torch_info: dict[str, Any] = {
        "importable": True,
        "version": str(torch.__version__),
        "hip": hip_version,
        "rocm_build": bool(hip_version),
        "accelerator_available": None,
        "rocm_device_available": False,
        "device_count": None,
        "device_names": [],
        "gfx_architectures": [],
    }
    try:
        accelerator_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
        torch_info["accelerator_available"] = accelerator_available
        torch_info["rocm_device_available"] = bool(hip_version) and accelerator_available
        torch_info["device_count"] = device_count
        if accelerator_available:
            torch_info["device_names"] = [
                str(torch.cuda.get_device_name(index))
                for index in range(device_count)
            ]
        if torch_info["rocm_device_available"]:
            for index in range(device_count):
                try:
                    properties = torch.cuda.get_device_properties(index)
                    architecture = getattr(properties, "gcnArchName", None)
                    if architecture:
                        torch_info["gfx_architectures"].append(str(architecture))
                except Exception:
                    # Property names and availability vary across ROCm/PyTorch builds.
                    continue
    except Exception as exc:
        torch_info["warning"] = f"accelerator inspection unavailable: {type(exc).__name__}"
    info["torch"] = torch_info
    return info


def check_llm_configuration(settings: Settings) -> tuple[bool, str]:
    try:
        validate_local_llm_base_url(
            settings.llm_base_url,
            allow_private_network=settings.llm_allow_private_network,
        )
    except ValueError as exc:
        return False, str(exc)
    return True, "local endpoint policy accepted"


def format_runtime_report(info: dict[str, Any], config_check: tuple[bool, str]) -> str:
    """Format facts without printing endpoint addresses, usernames, or tokens."""
    utilities = info["utilities"]
    torch = info["torch"]
    lines = [
        f"Python: {info['python']}",
        f"Platform: {info['platform']}",
        "Utilities: " + ", ".join(
            f"{name}={'yes' if available else 'no'}"
            for name, available in utilities.items()
        ),
    ]
    if torch["importable"]:
        accelerator = torch["accelerator_available"]
        lines.extend(
            [
                f"PyTorch: {torch['version']}",
                f"HIP: {torch['hip'] or 'not detected'}",
                f"HIP/ROCm PyTorch build: {'yes' if torch['rocm_build'] else 'no'}",
                "Generic accelerator available through torch.cuda: "
                + ("unknown" if accelerator is None else ("yes" if accelerator else "no")),
                "ROCm device available to PyTorch: "
                + ("yes" if torch["rocm_device_available"] else "no"),
                f"Device count reported: {torch['device_count'] if torch['device_count'] is not None else 'unknown'}",
            ]
        )
        lines.extend(f"Device: {name}" for name in torch["device_names"])
        lines.extend(f"GFX architecture: {name}" for name in torch["gfx_architectures"])
        if accelerator and not torch["rocm_build"]:
            lines.append(
                "Warning: an accelerator is available, but this is not a HIP/ROCm PyTorch build."
            )
        if "warning" in torch:
            lines.append(f"Warning: {torch['warning']}")
    else:
        lines.append(f"Warning: {torch['warning']}")
    status, message = config_check
    lines.append(f"Local LLM configuration: {'ok' if status else 'invalid'} ({message})")
    return "\n".join(lines)


def main() -> int:
    settings = Settings()
    info = collect_runtime_info()
    config_check = check_llm_configuration(settings)
    print(format_runtime_report(info, config_check))
    return 0 if config_check[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
