import pytest
import torch
from airadv.train import resolve_device

def test_cuda_default_and_explicit_index(monkeypatch):
    selected=[]
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"current_device",lambda:2)
    monkeypatch.setattr(torch.cuda,"set_device",selected.append)
    assert resolve_device("cuda")==torch.device("cuda:2")
    assert resolve_device("cuda:0")==torch.device("cuda:0")
    assert selected==[torch.device("cuda:2"),torch.device("cuda:0")]

def test_cpu_and_missing_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    assert resolve_device("cpu")==torch.device("cpu")
    with pytest.raises(RuntimeError,match="CUDA unavailable"):
        resolve_device("cuda")
