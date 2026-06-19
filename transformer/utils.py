import torch


def resolve_device(device_name: str | None = "cuda") -> torch.device:
    if not device_name:
        device_name = "cuda"

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; using CPU instead.")
        return torch.device("cpu")

    return device
