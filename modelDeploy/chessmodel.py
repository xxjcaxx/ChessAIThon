import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import numpy as np

import sys

sys.path.append("./chessintionlib")  

#from chess_aux_c import uci_to_number, number_to_uci, concat_fen_legal, concat_fen_legal_bits, concat_fen_legal_ptr


"""
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()

        # Model parameters
        bit_layers = 77  
        in_channels = bit_layers
        base_channels = 128  # Base number of channels  # Increase!!
        kernel_size = 3
        padding = kernel_size // 2
        lineal_channels = 1024  

        # First convolution layer (no residual needed)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Second convolution with residual
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.res_conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1)  # 1x1 conv to match channels

        # Third convolution with residual
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.res_conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1)

        # Fourth convolution with residual
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)
        self.res_conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(base_channels * 8 * 8 * 8, lineal_channels)  # Retain spatial info
        self.drop1 = nn.Dropout(p=0.4)  # Lower dropout for better accuracy

        self.fc2 = nn.Linear(lineal_channels, lineal_channels)
        self.drop2 = nn.Dropout(p=0.4)

        self.fcf = nn.Linear(lineal_channels, 4096)  

    def forward(self, x):
        # First convolution (no residual)
        x = F.relu(self.bn1(self.conv1(x)))

        # Second layer with residual
        res = self.res_conv2(x)  
        x = F.relu(self.bn2(self.conv2(x))) + res  

        # Third layer with residual
        res = self.res_conv3(x)
        x = F.relu(self.bn3(self.conv3(x))) + res  

        # Fourth layer with residual
        res = self.res_conv4(x)
        x = F.relu(self.bn4(self.conv4(x))) + res  

        # Flatten while keeping spatial information
        x = x.view(x.size(0), -1)  

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = self.fcf(x)

        return x

class ChessNetPV(nn.Module):
    def __init__(self):
        super(ChessNetPV, self).__init__()

        # Model parameters
        bit_layers = 77
        in_channels = bit_layers
        base_channels = 128  # Base number of channels  # Increase!!
        kernel_size = 3
        padding = kernel_size // 2
        lineal_channels = 1024

        # First convolution layer (no residual needed)
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Second convolution with residual
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(base_channels * 2)
        self.res_conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=1)  # 1x1 conv to match channels

        # Third convolution with residual
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(base_channels * 4)
        self.res_conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=1)

        # Fourth convolution with residual
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(base_channels * 8)
        self.res_conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(base_channels * 8 * 8 * 8, lineal_channels)  # Retain spatial info
        self.drop1 = nn.Dropout(p=0.4)  # Lower dropout for better accuracy

        self.fc2 = nn.Linear(lineal_channels, lineal_channels)
        self.drop2 = nn.Dropout(p=0.4)

        # Política: Salida de 4096 movimientos
        self.fcf = nn.Linear(lineal_channels, 4096)
        
        # Valor: Salida de 1 escalar (Evaluación de la posición)
        self.fc_value_1 = nn.Linear(lineal_channels, 256)
        self.fc_value_2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh() # Para rango [-1, 1]

    def forward(self, x):
        # First convolution (no residual)
        x = F.relu(self.bn1(self.conv1(x)))

        # Second layer with residual
        res = self.res_conv2(x)
        x = F.relu(self.bn2(self.conv2(x))) + res

        # Third layer with residual
        res = self.res_conv3(x)
        x = F.relu(self.bn3(self.conv3(x))) + res

        # Fourth layer with residual
        res = self.res_conv4(x)
        x = F.relu(self.bn4(self.conv4(x))) + res

        # Flatten while keeping spatial information
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        # Policy
        policy = self.fcf(x)

        # Value
        value = F.relu(self.fc_value_1(x))
        value = self.tanh(self.fc_value_2(value))
        

        return policy, value
"""

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
    model_path = "chessmarro_v9_final.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

"""
def predict_chess_move(fen_position,model,device):

    board = concat_fen_legal(fen_position)
    board_matrix = torch.tensor(board, dtype=torch.float32)   
    board_matrix = board.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(board_matrix)
    # Get legal move mask
    legal_moves_mask =  board[-64:].reshape(4096).to(dtype=torch.bool)
    # Ensure illegal moves are -inf
    outputs = torch.where(legal_moves_mask.unsqueeze(0), outputs, -float('inf'))
    # Filter only valid move indices
    valid_indices = torch.nonzero(legal_moves_mask, as_tuple=True)[0]
    # Apply softmax only on legal moves
    temperature = 1.2
    legal_outputs = outputs[0, valid_indices]  # Extract valid logits
    probabilities = torch.softmax(legal_outputs / temperature, dim=0)
    # Sample a move index from only valid moves
    move_idx = torch.multinomial(probabilities, 1).item()
    # Convert back to the original move index
    selected_move_index = valid_indices[move_idx].item()
    return number_to_uci(selected_move_index)
"""
#r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8
