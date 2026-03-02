import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import chess
import numpy as np

import sys

sys.path.append("./chessintionlib")  

from chess_aux_c import uci_to_number, number_to_uci, concat_fen_legal, concat_fen_legal_bits, concat_fen_legal_ptr
from chess_aux import concat_fen_legal as cflp

labels = ["white pawns", "black pawns", "white knights", "black knights", "white bishops", "black bishops",
         "white rooks", "black rooks", "white queen", "black queen", "white king", "black king", "**Turn**",
         "1 North moves", "1 NE moves", "1 East moves", "1 SE moves", "1 South moves", "1 SW moves", "1 West moves", "1 NW moves",
         "2 North moves", "2 NE moves", "2 East moves", "2 SE moves", "2 South moves", "2 SW moves", "2 West moves", "2 NW moves",
         "3 North moves", "3 NE moves", "3 East moves", "3 SE moves", "3 South moves", "3 SW moves", "3 West moves", "3 NW moves",
         "4 North moves", "4 NE moves", "4 East moves", "4 SE moves", "4 South moves", "4 SW moves", "4 West moves", "4 NW moves",
         "5 North moves", "5 NE moves", "5 East moves", "5 SE moves", "5 South moves", "5 SW moves", "5 West moves", "5 NW moves",
         "6 North moves", "6 NE moves", "6 East moves", "6 SE moves", "6 South moves", "6 SW moves", "6 West moves", "6 NW moves",
         "7 North moves", "7 NE moves", "7 East moves", "7 SE moves", "7 South moves", "7 SW moves", "7 West moves", "7 NW moves",
         "E2N Knight", "2EN Knight", "2ES Knight", "E2S Knight", "W2S Knight", "2WS Knight", "2WN Knight", "W2N Knight",
         "none", "none", "none", "none", "none", "none", "none", "none",
         "none", "none", "none", "none", "none", "none", "none", "none",
         ]


fen = 'r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8'

p = cflp(fen)

c = concat_fen_legal_bits(fen)
C = concat_fen_legal(fen)

print("************** Bits ")
for i in range(77):
    print(f"Slice {i+1}", labels[i])
    for row_a, row_b in zip(p[i].astype(int), c[i].int().cpu().numpy()):
        print(" ".join(map(str, row_a)), "   |   ", " ".join(map(str, row_b)))  
    print("-" * 40)  # Separator for better visualization

print("*************** normal")
for i in range(77):
    print(f"Slice {i+1}", labels[i])
    for row_a, row_b in zip(p[i].astype(int), C[i].int().cpu().numpy()):
        print(" ".join(map(str, row_a)), "   |   ", " ".join(map(str, row_b)))  
    print("-" * 40)  # Separator for better visualization
 
board_matrix = torch.tensor(p, dtype=torch.float32) 
board_matrix = board_matrix.unsqueeze(0)
board_matrix = board_matrix.to("cuda")   
c = c.unsqueeze(0)

if board_matrix.shape == c.shape:
    print("Shapes iguales")
else:
    print(f"Shapes distintos: {board_matrix.shape} vs {c.shape}")

if board_matrix.dtype == c.dtype:
    print("Tipos iguales")
else:
    print(f"Tipos distintos: {board_matrix.dtype} vs {c.dtype}")

if board_matrix.device == c.device:
    print("Mismo device")
else:
    print(f"Devices distintos: {board_matrix.device} vs {c.device}")

if torch.equal(board_matrix, c):
    print("Son idénticos")
else:
    print("Difieren en algún valor")


diff_mask = board_matrix != c
print("Número de diferencias:", diff_mask.sum().item())

# Ejemplo: mostrar índices y valores distintos
indices = torch.nonzero(diff_mask, as_tuple=False)
for idx in indices:
    i = tuple(idx.tolist())
    print(f"En {i}: board_matrix={board_matrix[i]} vs c={c[i]}")