"""Explicit hardware admission for the existing unchanged Hopper acceptance math."""


def add_hardware_argument(parser):
    parser.add_argument("--hardware-family", choices=("H100", "H200"), default="H100")


def require_hardware(family):
    import torch

    if family not in ("H100", "H200"):
        raise ValueError("Select explicit H100 or H200 acceptance hardware")
    assert torch.cuda.is_available(), "Requires CUDA"
    actual = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    assert family in actual and capability == (9, 0), (family, actual, capability)
    return dict(expected_family=family, actual=actual, compute_capability=list(capability))
