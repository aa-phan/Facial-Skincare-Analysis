# gpu_utils.py
import torch

def is_cuda_available() -> bool:
    """🔍 Check if CUDA (GPU) is available."""
    return torch.cuda.is_available()

def get_gpu_info() -> list[dict]:
    """💻 Retrieve GPU information if available."""
    if not is_cuda_available():
        return []

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_info.append({"id": i, "name": gpu_name})
    return gpu_info

def tensor_operation() -> torch.Tensor:
    """🧮 Perform a simple tensor operation on the GPU."""
    if not is_cuda_available():
        raise RuntimeError("❌ CUDA is not available.")

    x = torch.rand(3, 3).cuda()
    y = torch.rand(3, 3).cuda()
    z = x + y
    return z
