import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import numpy as np

import sys

sys.path.append("./chessintionlib")  

# --------------------------------------------------
# Modify this file if you want to use other models
model_PATH = "chessmarro_v9_final.pth"
# --------------------------------------------------


class Mish(nn.Module):
    """Activación Mish: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block para atención de canales"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """Bloque Residual con Pre-activación y SE"""
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.se = SEBlock(channels)
        self.activation = Mish()

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        
        out = self.se(out)
        return out + residual

class ChessNetPV_Optimized(nn.Module):
    def __init__(self, num_blocks=12): # 6 o 12 bloques es mucho más profundo que el original
        super(ChessNetPV_Optimized, self).__init__()

        in_channels = 77
        base_channels = 256 # Ancho constante profesional 
        head_bottleneck_channels = 32 # Para reducir parámetros en FC [6]
        

        # Entrada inicial
        self.conv_input = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False)
        
        # Torre Residual (Cuerpo de la red)
        self.res_tower = nn.Sequential(
            *[ResBlock(base_channels) for _ in range(num_blocks)]
        )
        
        # BN y Activación final de la torre (por arquitectura de pre-activación)
        self.final_bn = nn.BatchNorm2d(base_channels)
        self.final_act = Mish()

        # --- CABEZAL DE POLÍTICA (4096 salidas) ---
        self.policy_conv = nn.Conv2d(base_channels, head_bottleneck_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(head_bottleneck_channels)
        self.policy_fc = nn.Linear(head_bottleneck_channels * 8 * 8, 4096)

        # --- CABEZAL DE VALOR (1 salida) ---
        self.value_conv = nn.Conv2d(base_channels, head_bottleneck_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(head_bottleneck_channels)
        self.value_fc1 = nn.Linear(head_bottleneck_channels * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Cuerpo
        x = self.conv_input(x)
        x = self.res_tower(x)
        x = self.final_act(self.final_bn(x))

        # Política
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)

        # Valor
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        value = F.relu(self.value_fc1(v))
        value = self.tanh(self.value_fc2(value))

        return policy, value

def init_model():
    Chess_model = ChessNetPV_Optimized  # Define class
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    print(f"Using {device} device")

    # Create model ONCE (to retain progress)
    model = Chess_model().to(device)
    print(model)

    # Load the model
    model_path = model_PATH
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

#r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8
