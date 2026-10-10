"""Type check target file for TensorClass pattern.

This file is checked by pyrefly, mypy, and pyright to verify that
the TensorClass inheritance pattern works correctly with type checkers.
"""

import torch
from tensordict.tensorclass import TensorClass


class MyData(TensorClass):
    x: torch.Tensor
    y: torch.Tensor


def main() -> None:
    B = 10
    # Test __init__ with batch_size kwarg
    data = MyData(x=torch.randn(B, 3), y=torch.randn(B, 3), batch_size=[B])

    # Test injected attributes
    _batch_size: torch.Size = data.batch_size
    _device: torch.device | None = data.device

    # Test injected methods
    clone = data.clone()
    _clone_batch_size: torch.Size = clone.batch_size

    # Test to() method with str device (PyTorch supports str)
    cpu_data = data.to(device="cpu")
    _cpu_device: torch.device | None = cpu_data.device

    # Test cpu() method
    cpu_data2 = data.cpu()
    _cpu_device2: torch.device | None = cpu_data2.device
