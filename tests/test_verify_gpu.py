import pytest

from verify_gpu import verify_gpu


def test_gpu_available(mocker):
    """
    Test GPU availability and return GPU info if available.

    written by chatgpt4

    """
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.device_count", return_value=1)
    mocker.patch("torch.cuda.get_device_name", return_value="Test GPU")

    result = verify_gpu()
    assert "Number of GPUs available: 1" in result
    assert "GPU 0: Test GPU" in result


def test_no_gpu_available(mocker):
    """
    Test GPU availability and return GPU info if available.

    written by chatgpt4

    """
    mocker.patch("torch.cuda.is_available", return_value=False)

    with pytest.raises(SystemError) as exc:
        verify_gpu()
    assert str(exc.value) == "No GPU found. GPU is required"
