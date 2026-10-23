import torch


def get_device():
    """Return the best available device: "cuda", "mps" or "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
