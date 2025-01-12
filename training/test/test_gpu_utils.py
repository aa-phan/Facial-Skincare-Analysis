# test_gpu_utils.py
import pytest
from gpu_utils import is_cuda_available, get_gpu_info, test_tensor_operation

def test_cuda_availability():
    """🔍 Test if CUDA is detected correctly."""
    assert isinstance(is_cuda_available(), bool), "❌ CUDA availability should return a boolean."
    print("✅ CUDA availability check passed!")

def test_gpu_info_format():
    """💻 Test the structure of GPU info."""
    if is_cuda_available():
        gpu_info = get_gpu_info()
        assert isinstance(gpu_info, list), "❌ GPU info should be a list."
        assert all("id" in gpu and "name" in gpu for gpu in gpu_info), "❌ GPU entries must have 'id' and 'name'."
        print("📝 GPU info format check passed!")
    else:
        print("⚠️ CUDA not available, skipping GPU info format test.")

def test_tensor_gpu_operation():
    """🧮 Test tensor operation on GPU."""
    if is_cuda_available():
        result = test_tensor_operation()
        assert result.is_cuda, "❌ Result tensor should be on the GPU."
        assert result.shape == (3, 3), "❌ Result tensor should be of shape (3, 3)."
        print("🚀 Tensor operation on GPU succeeded!")
    else:
        pytest.skip("⚠️ CUDA not available, skipping tensor operation test.")
