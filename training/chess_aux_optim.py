import chess
import numpy as np
import torch

# --- Precalcula códigos de movimientos ---
codes, i = {}, 0
for nSquares in range(1,8):
    for direction in [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]:
        codes[(nSquares*direction[0], nSquares*direction[1])] = i
        i += 1
# Knight moves
knight_moves = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]
for move in knight_moves:
    codes[move] = i
    i += 1

# --- Convierte un tablero a 13x8x8 boolean ---
def board2rep_fast(board):
    piece_map = board.piece_map()
    layers = np.zeros((12,8,8), dtype=np.uint8)
    for sq, p in piece_map.items():
        rank, file = 7 - chess.square_rank(sq), chess.square_file(sq)
        idx = (p.piece_type-1)*2 + (0 if p.color==chess.WHITE else 1)
        layers[idx, rank, file] = 1
    turn_layer = np.ones((8,8), dtype=np.uint8) if board.turn else np.zeros((8,8), dtype=np.uint8)
    board_rep = np.concatenate([layers, turn_layer[None]], axis=0)
    return board_rep.astype(bool)

# --- Convierte jugadas legales a 64x8x8 boolean ---
def legal_moves_to_64_8_8_fast(board):
    legal_moves = list(board.legal_moves)
    array6488 = np.zeros((64,8,8), dtype=bool)
    if not legal_moves:
        return array6488
    from_sq = np.array([m.from_square for m in legal_moves])
    to_sq   = np.array([m.to_square for m in legal_moves])
    from_ranks = 7 - np.array([chess.square_rank(s) for s in from_sq])
    from_files = np.array([chess.square_file(s) for s in from_sq])
    delta_r = np.array([chess.square_rank(t)-chess.square_rank(f) for f,t in zip(from_sq,to_sq)])
    delta_f = np.array([chess.square_file(t)-chess.square_file(f) for f,t in zip(from_sq,to_sq)])
    move_codes = np.array([codes[(df, dr)] for df, dr in zip(delta_f, delta_r)])
    array6488[move_codes, from_ranks, from_files] = 1
    return array6488

# --- Concatenar tablero y jugadas legales ---
def concat_fen_legal(fen):
    board = chess.Board(fen)
    board_rep = board2rep_fast(board)
    legal_rep = legal_moves_to_64_8_8_fast(board)
    return np.concatenate([board_rep, legal_rep], axis=0)

# --- Batch de FENs a tensor PyTorch listo para GPU ---
def batch_fens_to_tensor(fen_list, device="cuda"):
    boards_np = [concat_fen_legal(fen) for fen in fen_list]
    boards_tensor = torch.tensor(np.stack(boards_np), dtype=torch.float32, device=device)
    return boards_tensor
