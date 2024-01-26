import torch


def verify_gpu():
    """
    Verify GPU availability and return GPU info if available.

    This script is only used in the Dockerfile to confirm the availability of GPU

    """

    output = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        output.append(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            output.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        raise SystemError("No GPU found. GPU is required")

    return "\n".join(output)


if __name__ == "__main__":
    print(verify_gpu())
