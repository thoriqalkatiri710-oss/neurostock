import torch
import gymnasium as gym
import pettingzoo
import pandas as pd
import numpy as np

def test_imports():
    assert torch.__version__ is not None
    assert pd.__version__ is not None
    assert np.__version__ is not None
    print("Semua library berhasil di-import.")

def test_torch_basic_op():
    x = torch.randn(3, 3)
    y = torch.randn(3, 3)
    z = x @ y
    assert z.shape == (3, 3)
    print("Operasi tensor dasar berhasil.")

if __name__ == "__main__":
    test_imports()
    test_torch_basic_op()