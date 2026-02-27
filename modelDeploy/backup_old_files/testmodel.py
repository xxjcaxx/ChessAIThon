import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from chessmodel import predict_chess_move, ChessNet, init_model
import sys
sys.path.append("./chessintionlib")  
from chess_aux_c import uci_to_number, number_to_uci, concat_fen_legal, concat_fen_legal_bits, concat_fen_legal_ptr

model, device = init_model()

def predict_chess_moves_vectorized(boards_tensor, temperature=1.2):
    """
    Predice movimientos para un batch de posiciones con Conv2D, usando
    movimientos legales codificados en los últimos 64 canales/elementos.
    """
    B = boards_tensor.size(0)
    device = boards_tensor.device

    with torch.no_grad():
        outputs = model(boards_tensor)  # [B, 4096], conv2d espera [B, C, H, W]

    # Extraer máscaras legales de los últimos 64 canales
    # Flatten del canal y el tablero a 4096
    legal_masks = boards_tensor[:, -64:, :, :].reshape(B, 4096).to(dtype=torch.bool)

    # Poner -inf en posiciones ilegales
    masked_logits = torch.where(legal_masks, outputs, -float('inf'))

    # Softmax con temperatura
    probs = torch.softmax(masked_logits / temperature, dim=1)  # [B, 4096]

    # Samplear movimiento
    move_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]

    # Convertir a UCI
    pred_moves = [number_to_uci(idx.item()) for idx in move_indices]

    return pred_moves



fens = [
    "r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8",
    "r1bqkbnr/pppppppp/n7/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 2",
    '5k2/R7/3K4/4p3/5P2/8/8/5r2 w - - 0 0',
'5k2/1R6/4p1p1/1pr3Pp/7P/1K6/8/8 w - - 0 0',
'5k2/8/p7/4K1P1/P4R2/6r1/8/8 b - - 0 0',
'8/8/8/p2r1k2/7p/PP1RK3/6P1/8 b - - 0 0',
'8/8/8/1P4p1/5k2/5p2/P6K/8 b - - 0 0',
'3b2k1/1p3p2/p1p5/2P4p/1P2P1p1/5p2/5P2/4RK2 w - - 0 0',
'5k2/3R4/2K1p1p1/4P1P1/5P2/8/3r4/8 b - - 0 0',
'6k1/6pp/5p2/8/5P2/P7/2K4P/8 b - - 0 0',
'8/3R4/8/r3N2p/P1Pp1P2/2k2K1P/3r4/8 w - - 0 0',
'6k1/8/6r1/8/5b2/2PR4/4K3/8 w - - 0 0',
'8/1p3k2/3B4/8/3b2P1/1P6/6K1/8 b - - 0 0',
'8/8/8/2p1k3/P6R/1K6/6rP/8 w - - 0 0',
'6k1/5p1p/6p1/1P1n4/1K4P1/N6P/8/8 w - - 0 0',
'8/k5r1/2N5/PK6/2B5/8/8/8 b - - 0 0',
'6k1/8/5K2/8/5P1R/r6P/8/8 b - - 0 0',
'8/8/4k1KP/p5P1/r7/8/8/8 w - - 0 0',
'1R6/p2r4/2ppkp2/6p1/2PKP2p/P4P2/6PP/8 b - - 0 0',
'8/7p/6p1/8/k7/8/2K3P1/8 b - - 0 0',
'R7/8/8/6p1/4k3/3rPp1P/8/6K1 b - - 0 0',
'8/7p/1p1k2p1/p1p2p2/8/PP2P2P/4KPP1/8 w - - 0 0'
]

   # Preprocesar todas las posiciones
boards = [concat_fen_legal(fen) for fen in fens]  # cada uno es numpy
boards_tensor = torch.stack(
[b.detach().clone().float() if isinstance(b, torch.Tensor) else torch.from_numpy(b).float() 
    for b in boards]
).to(device)

moves = predict_chess_moves_vectorized(boards_tensor, 1.0)
print(moves)
# -> ['c5d4', 'g8f6']   (ejemplo)

import chess

def check_batch_legality(fens, moves):
    """
    Comprueba si los movimientos predichos son legales en sus respectivas posiciones FEN.

    Args:
        fens (list[str]): lista de posiciones en formato FEN.
        moves (list[str]): lista de movimientos en notación UCI (uno por FEN).

    Returns:
        list[bool]: lista de True/False indicando si cada movimiento es legal.
    """
    assert len(fens) == len(moves), "El número de FENs y movimientos debe coincidir"

    results = []
    for fen, move in zip(fens, moves):
        board = chess.Board(fen)
        results.append(move in {m.uci() for m in board.legal_moves})
    return results


print(check_batch_legality(fens, moves))