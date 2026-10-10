# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Type checking tests for tensorclass.

This module tests that TensorClass base class pattern works correctly
with type checkers. The TensorClass base class exposes injected members
like batch_size, device, clone(), etc. to the type checker.

Type checker compatibility:
- TensorClass inheritance pattern: Works with pyrefly, mypy, and pyright
- @tensorclass decorator pattern: Works at RUNTIME only
  (Type checkers cannot see dynamically injected members due to Python limitations)

Usage pattern:
    class MyData(TensorClass):
        x: torch.Tensor
        y: torch.Tensor

    # Type checkers can now see batch_size, device, clone(), etc.
    data = MyData(x=..., y=..., batch_size=[B])
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from tensordict import tensorclass
from tensordict.tensorclass import TensorClass


# Pattern 1: Using TensorClass base class (recommended for type checker compatibility)
# This pattern works with pyrefly, mypy, and pyright because the type checker
# can see the injected members from the base class.
class ExchangeState(TensorClass):
    t: torch.Tensor
    pos: torch.Tensor


# Pattern 2: Using @tensorclass decorator (original pattern)
# This pattern works at runtime but type checkers cannot recognize
# dynamically injected members like batch_size, device, clone(), etc.
@tensorclass
class ExchangeStateDecorator:
    t: torch.Tensor
    pos: torch.Tensor


# ============================================================================
# Runtime Tests
# ============================================================================


def test_tensorclass_base_runtime() -> None:
    """Test that TensorClass base pattern works at runtime."""
    B = 10
    state = ExchangeState(t=torch.randn(B, 3), pos=torch.randn(B, 3), batch_size=[B])

    # Access injected members
    assert state.batch_size == torch.Size([B])
    assert state.device is None

    # Test clone
    clone = state.clone()
    assert clone.batch_size == torch.Size([B])

    # Test to() method with str device (PyTorch supports str)
    cpu_state = state.to(device="cpu")
    assert cpu_state.device == torch.device("cpu")

    # Test cpu() method
    cpu_state2 = state.cpu()
    assert cpu_state2.device == torch.device("cpu")


def test_tensorclass_decorator_runtime() -> None:
    """Test that @tensorclass decorator pattern works at runtime."""
    B = 10
    state = ExchangeStateDecorator(
        t=torch.randn(B, 3),
        pos=torch.randn(B, 3),
        batch_size=[B],
    )

    # Access injected members (runtime works)
    assert state.batch_size == torch.Size([B])
    assert state.device is None

    # Test clone
    clone = state.clone()
    assert clone.batch_size == torch.Size([B])


# ============================================================================
# Type Checker Tests
# ============================================================================

# Type check target file: TensorClass inheritance pattern
# This pattern works with all type checkers (pyrefly, mypy, pyright)
TYPE_CHECK_FILE = Path(__file__).parent / "_type_check_tensorclass.py"


def test_pyrefly_type_check() -> None:
    """Test that pyrefly accepts TensorClass inheritance pattern."""
    result = subprocess.run(
        [sys.executable, "-m", "pyrefly", "check", str(TYPE_CHECK_FILE)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,  # Run from project root
    )

    # Check for errors
    output = result.stdout + result.stderr
    if "0 errors" in output:
        return  # Success

    if result.returncode != 0:
        pytest.fail(f"pyrefly type check failed:\n{output}")


def test_mypy_type_check() -> None:
    """Test that mypy accepts TensorClass inheritance pattern."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            str(TYPE_CHECK_FILE),
            "--ignore-missing-imports",
            "--follow-imports=skip",  # Only check the target file
            "--no-error-summary",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,  # Run from project root
    )

    # mypy returns 0 on success, or stdout contains "Success"
    if result.returncode == 0 or "Success" in result.stdout:
        return

    # Filter errors: only consider errors in the target file itself
    # (ignore errors in tensordict stubs which are pre-existing issues)
    target_file_name = TYPE_CHECK_FILE.name
    target_errors = [
        line
        for line in result.stdout.splitlines()
        if target_file_name in line and "error:" in line
    ]

    if not target_errors:
        return  # No errors in target file, success

    pytest.fail(
        f"mypy type check failed with errors in {target_file_name}:\n"
        + "\n".join(target_errors)
    )


def test_pyright_type_check() -> None:
    """Test that pyright accepts TensorClass inheritance pattern."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            str(TYPE_CHECK_FILE),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,  # Run from project root
    )

    # pyright returns 0 on success
    if result.returncode == 0:
        return

    pytest.fail(
        f"pyright type check failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
