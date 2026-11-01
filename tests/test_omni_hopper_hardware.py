import argparse
from unittest.mock import patch
import pytest
from tests.acceptance.omni_hopper_hardware import add_hardware_argument, require_hardware

def test_default_h100_and_explicit_h200():
    p=argparse.ArgumentParser();add_hardware_argument(p)
    assert p.parse_args([]).hardware_family=="H100"
    assert p.parse_args(["--hardware-family","H200"]).hardware_family=="H200"

@pytest.mark.parametrize("expected,actual,cap,allowed",[
    ("H100","NVIDIA H100 80GB HBM3",(9,0),True),
    ("H200","NVIDIA H200",(9,0),True),
    ("H100","NVIDIA H200",(9,0),False),
    ("H200","NVIDIA H100 80GB HBM3",(9,0),False),
    ("H200","NVIDIA A100",(8,0),False),
    ("H200","NVIDIA H200",(8,0),False),
])
def test_explicit_family_and_architecture_contract(expected,actual,cap,allowed):
    with patch("torch.cuda.is_available",return_value=True),patch("torch.cuda.get_device_name",return_value=actual),patch("torch.cuda.get_device_capability",return_value=cap):
        if allowed:
            assert require_hardware(expected)["actual"]==actual
        else:
            with pytest.raises(AssertionError):require_hardware(expected)

def test_no_cuda_is_not_a_gpu_pass():
    with patch("torch.cuda.is_available",return_value=False):
        with pytest.raises(AssertionError):require_hardware("H200")
