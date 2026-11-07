"""Concrete PyTorch execution backends; selection and validation stay outside core."""
import os

import torch


class _TorchCompute:
    """Shared placement operations, not a second learning algorithm."""
    @property
    def device(self):
        return self._device

    def place_model(self, model):
        return model.to(self.device)

    def tensor(self, data, *, dtype=None):
        return torch.as_tensor(data, dtype=dtype, device=self.device)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def randperm(self, size):
        return torch.randperm(size, device=self.device)


class CPUBackend(_TorchCompute):
    """CPU execution never probes GPU availability or initializes CUDA."""
    name = "cpu"

    def __init__(self):
        self._device = torch.device("cpu")

    def synchronize(self):
        pass

    def preserve_rng(self):
        return torch.random.fork_rng(devices=[])


class CUDABackend(_TorchCompute):
    """CUDA-only GPU execution. Unsupported/unavailable devices never fall back."""
    name = "cuda"

    def __init__(self, device="cuda"):
        requested = _parse_device(device)
        if requested.type != "cuda":
            raise ValueError(f"CUDA backend requires cuda or cuda:N, got {device!r}")
        if not torch.cuda.is_available():
            raise ValueError("CUDA GPU requested but unavailable; install CUDA-enabled PyTorch and a working NVIDIA driver, or use --cpu")
        index = requested.index if requested.index is not None else torch.cuda.current_device()
        count = torch.cuda.device_count()
        if not 0 <= index < count:
            raise ValueError(f"CUDA device index {index} is invalid: {count} visible CUDA device(s)")
        self._device = torch.device("cuda", index)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            # Validate the actual context/allocation before any output or UI exists.
            torch.empty(0, device=self.device)
            self.synchronize()
        except Exception as exc:
            raise ValueError(f"CUDA device {self.device} could not be initialized: {exc}") from exc

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def preserve_rng(self):
        return torch.random.fork_rng(devices=[self.device.index])


def _parse_device(device):
    try:
        parsed = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Invalid compute device {device!r}; expected cpu, cuda, or cuda:N") from exc
    if parsed.type not in ("cpu", "cuda") or (parsed.type == "cpu" and parsed.index is not None):
        raise ValueError(f"Unsupported compute device {device!r}; expected cpu, cuda, or cuda:N")
    # PyTorch stores device indices in a small integer: e.g. cuda:256 can wrap
    # silently to cuda:0. Reject that alias before it can select the wrong GPU.
    if isinstance(device, str) and device.startswith("cuda:"):
        if parsed.index != int(device.split(":", 1)[1]):
            raise ValueError(f"Invalid compute device {device!r}: CUDA index exceeds the supported range")
    return parsed


def create_compute(device="cpu"):
    """Composition-root factory. GPU means CUDA, never MPS or automatic fallback."""
    parsed = _parse_device(device)
    return CPUBackend() if parsed.type == "cpu" else CUDABackend(parsed)


def compute_metadata(compute):
    """Keep effective execution identity distinct from requested config identity."""
    return {"device": str(compute.device), "compute_device": str(compute.device),
            "compute_backend": type(compute).__name__, "compute_backend_name": compute.name}
