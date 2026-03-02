import os
import sys
import importlib.util
import torch
import torch.nn as nn
from multiprocessing import set_start_method

# Ensure we use fork on Linux for multiprocessing (matches runtime behavior)
try:
    set_start_method('fork')
except RuntimeError:
    pass

# Load the target module by path to avoid package import issues
MODULE_PATH = os.path.join(os.path.dirname(__file__), 'chessgamemultithread.py')
spec = importlib.util.spec_from_file_location('chessgamemultithread', MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Stub concat_fen_legal to a simple tensor with 77 channels: first 13 zeros, last 64 ones
import torch

def stub_concat_fen_legal(fen):
    # shape [77,8,8]
    parts = []
    parts.append(torch.zeros((13,8,8), dtype=torch.float32))
    parts.append(torch.ones((64,8,8), dtype=torch.float32))
    return torch.cat(parts, dim=0)

# Replace the module's concat_fen_legal with our stub
mod.concat_fen_legal = stub_concat_fen_legal

# Dummy model compatible with module expectations (nn.Module with forward)
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        B = x.size(0)
        return torch.zeros((B, 4096), dtype=torch.float32)

# Run the smoke test using CPU
dummy = DummyModel()
fen = "r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P3/2N2N2/PPPQ1PPP/R3K2R w KQkq - 0 1"
print('Starting smoke test...')
move = mod.chessmarro_mcts_predict_chess_move(fen, simulations=4, model=dummy, device='cpu')
print('Smoke test finished. Result move:', move)
