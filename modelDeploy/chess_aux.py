
import chess
import numpy as np

# Receives board in chess-python format and the piece type
# Returns the matrix representation of this pieces in the board
def board_to_matrix(board, piece_type):
    piece_map = board.piece_map()
    matrix = np.zeros((8,8))

    for square, piece in piece_map.items():
        # chess.square_rank y chess.square_file devuelven la fila y columna respectivamente
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        if(piece.piece_type == piece_type):
            matrix[7 - rank, file] = (-1 if piece.color == chess.BLACK else 1) * 1 # piece.piece_type
    return matrix

# Receives board in chess-python type and returns the 13 layers board representation
def board2rep(board):
    pieces = [1,2,3,4,5,6]
    layers = []
    for piece in pieces:
        matrix = board_to_matrix(board,piece)
        white_matrix = np.where(matrix == 1, 1, 0)
        black_matrix = np.where(matrix == -1, 1, 0)

        layers.append(white_matrix)
        layers.append(black_matrix)
    if board.turn:
        color_matrix = np.ones((8,8))
        layers.append(color_matrix)
    else:
        color_matrix = np.zeros((8,8))
        layers.append(color_matrix)
    board_rep = np.stack(layers)
    board_rep = board_rep.astype(bool)
    #print(board_rep)
    return board_rep




codes, i = {}, 0
    # All 56 regular moves
for nSquares in range(1,8):
    for direction in [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:
            codes[(nSquares*direction[0],nSquares*direction[1])] = i
            i += 1
    # 8 Knight moves
codes[(1,2)], i = i,  i+1
codes[(2,1)], i = i,  i+1
codes[(2,-1)], i = i,  i+1
codes[(1,-2)], i = i,  i+1
codes[(-1,-2)], i = i,  i+1
codes[(-2,-1)], i = i,  i+1
codes[(-2,1)], i = i,  i+1
codes[(-1,2)], i = i,  i+1
    # We avoid pawn promotion because are the same moves and we are looking for 1 best move.
    # print(len(codes)) 64 moves


# Receives a board and returns all legal moves in 64x8x8 matrix
# It uses chess-python to calculate them
#
def legal_moves_to_64_8_8(board):
    legal_moves = list(board.legal_moves)
# Apply the function to each move in the list
    extracted_moves = [[
        [chess.square_rank(m.from_square),chess.square_file(m.from_square)],
         (chess.square_file(m.to_square) - chess.square_file(m.from_square), chess.square_rank(m.to_square) - chess.square_rank(m.from_square))
    ] for m in legal_moves]

    array6488 = np.zeros((64,8,8))
    for em in extracted_moves:
        array6488[codes[em[1]], 7-em[0][0], em[0][1]] = 1

    array6488 = array6488.astype(bool)
    return  array6488

# This function is to convert to a number 0 to 4096
def uci_to_number(uci_move):
    m = uci_move #chess.Move.from_uci(uci_move)
    move_code = codes[(chess.square_file(m.to_square) - chess.square_file(m.from_square),
                   chess.square_rank(m.to_square) - chess.square_rank(m.from_square))]
    pos = np.ravel_multi_index(
        multi_index=((move_code, 7-chess.square_rank(m.from_square), chess.square_file(m.from_square))),
        dims=(64,8,8)
    )
    return pos

def number_to_uci(number_move):
    move_code, from_row, from_col = np.unravel_index(number_move, (64, 8, 8))  # Rank == row, file== col
    code = list(codes.keys())[list(codes.values()).index(move_code)]
    row_a = str(8-from_row)
    col_a = chr(ord('a') + from_col)
    col_b = chr(ord('a') + from_col + code[0])
    row_b = str(8-from_row + code[1])
    uci_move = f"{col_a}{row_a}{col_b}{row_b}"
    return uci_move

## Contantenates 13x8x8 to postions and 64x8x8 legal moves
def concat_fen_legal(fen):
    board = chess.Board(fen)
    fen_matrix = board2rep(board)
    legal_moves = legal_moves_to_64_8_8(board)
    fen_matrix_legal_moves = np.concatenate((fen_matrix,legal_moves),0)
    return fen_matrix_legal_moves






#r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8
