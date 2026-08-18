"""Compute-device selection and description: the single source of truth.

Every model loader picks its device the same way: prefer Apple's GPU (Metal/MPS), then
an NVIDIA GPU (CUDA), then fall back to CPU. That choice used to be copied into both
``cli.pipeline`` and ``pause_check``; it now lives here in :func:`resolve_device` so the
two can never drift, and so the GUI's status readout (via :func:`describe_compute`,
surfaced by ``cli.gpu_support probe``) reports exactly what a pipeline run will use.

torch is imported lazily inside the functions: JSON-only runs and the torch-free GUI
process can import this module without paying for (or requiring) torch.
"""

from __future__ import annotations


def resolve_device() -> str:
    """The device string the pipeline runs on: ``"mps"`` → ``"cuda"`` → ``"cpu"``."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _compute_label(device: str, *, cuda_installed: bool, has_nvidia_gpu: bool,
                   cuda_version: str | None, gpu_name: str | None) -> str:
    """Human-readable one-liner for the Settings/menu compute status."""
    if device == "mps":
        return "Apple GPU (Metal/MPS)"
    if device == "cuda":
        base = f"NVIDIA GPU (CUDA {cuda_version})" if cuda_version else "NVIDIA GPU (CUDA)"
        return f"{base}: {gpu_name}" if gpu_name else base
    # device == "cpu": say WHY a present CUDA install isn't in use; and never blame a missing
    # GPU when nvidia-smi reports one. Install + GPU + still-CPU means the CUDA build didn't
    # load (e.g. it was never activated: only the frozen entry calls activate_cuda_site).
    # Install + no GPU is the expected state after a forced test install on a GPU-less box,
    # and confirms the download/activation plumbing itself succeeded.
    if cuda_installed and has_nvidia_gpu:
        return "CPU (CUDA torch installed but not loaded: analysis still runs on the CPU build)"
    if cuda_installed:
        return "CPU (CUDA torch installed, but no compatible NVIDIA GPU detected)"
    return "CPU"


def describe_compute() -> dict[str, object]:
    """A snapshot of the active compute setup for display. Runs in a torch-importing
    process (``cli.gpu_support probe``); the GUI consumes the JSON rather than importing torch.

    Includes ``has_nvidia_gpu`` (via ``nvidia-smi``) so the GUI's first-run offer can learn
    whether to prompt without running that potentially-slow probe on the GUI thread: it rides
    along in this already-off-thread subprocess. Also reports ``torch_path`` (the loaded
    ``torch.__file__``): after a GPU install it should point into the per-user ``site/`` dir,
    which confirms the CUDA build actually shadowed the bundled CPU torch (see
    ``gpu_support.activate_cuda_site``)."""
    import torch

    from vodparse import gpu_support

    device = resolve_device()
    cuda_version = getattr(torch.version, "cuda", None)
    gpu_name: str | None = None
    if device == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 - display only; never let a name lookup break the probe
            gpu_name = None
    cuda_installed = gpu_support.active_site_dir() is not None
    nvidia_present = gpu_support.has_nvidia_gpu()
    return {
        "device": device,
        "label": _compute_label(device, cuda_installed=cuda_installed,
                                has_nvidia_gpu=nvidia_present,
                                cuda_version=cuda_version, gpu_name=gpu_name),
        "torch_version": torch.__version__,
        "torch_path": getattr(torch, "__file__", None),
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "cuda_installed": cuda_installed,
        "has_nvidia_gpu": nvidia_present,
    }
