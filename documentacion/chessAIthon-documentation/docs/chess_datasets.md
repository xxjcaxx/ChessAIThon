# Making a Chess Dataset

The goal of our project is to develop an AI model capable of determining the best move in a game of chess. Inspired by groundbreaking projects like **AlphaZero** and **Leela Chess Zero**, we aim to adopt a similar final architecture for our model. However, our approach to training will differ significantly due to practical constraints. Unlike AlphaZero and Leela, which rely on **reinforcement learning**—a process where the AI learns by playing millions of games against itself—we will train our model using **collected data** from existing chess games. This decision is driven by the fact that we do not have access to the immense computational resources required for self-play training, such as those available to organizations like Google.

To build our AI model, we need a **high-quality dataset** consisting of chess positions paired with the best moves and values, as determined by human expertise. 


To start, we will use a **base dataset** obtained from the **Lc0** webpage database, which contains a vast collection of chess games and Lc0 decisions. To prepare this data for our AI, we applied a methodology similar to **ETL (Extract, Transform, Load)**, a technique commonly used in big data processes. This process allowed us to clean, organize, and structure the raw data into a usable format, which is now stored in `parquet` files. These files serve as the foundation for training our AI model.

However, to make our AI stand out from others, we need to go beyond the base dataset and create **additional, carefully curated data**. This involves selecting unique and instructive chess positions and pairing them with the best moves, as determined by human judgment. By doing so, we can enrich the dataset with high-quality examples that reflect nuanced strategies and decision-making, helping our AI learn more effectively.

To facilitate this process, we have developed a **web application** that allows users to interact with the dataset. Using this tool, you can download the existing dataset, review chess positions, and contribute new examples by adding your own insights on the best moves. These contributions can then be integrated into the training pipeline, further enhancing the AI's ability to make intelligent and human-like decisions.


## Formats of the datasets

In the initial stages of our project, we will use **CSV files** due to their simplicity and tabular structure, which makes them easy to work with. Each line in the CSV file will contain two key pieces of information: the `FEN` representation of the chessboard and the `best move` in `UCI`  format. 

To facilitate collaboration and data sharing, our **web application** will enable users to download their contributed examples in CSV format. These files can then be uploaded to a **Git repository**, where they can be shared with the team and used in the preprocessing stage. This collaborative approach ensures that everyone can contribute to the dataset, enriching it with diverse and high-quality examples.

Once the CSV files are ready, we will use an **ETL (Extract, Transform, Load) tool** developed in a **Jupyter Notebook** to process the data further. This tool will transform the raw CSV data into a more advanced format suitable for training our AI model. Specifically, it will generate a **parquet file** with two additional columns: 

1. **77x8x8 Board Representation**: This is a numerical representation of the chessboard, formatted in a way that the AI model can easily process. It captures all the necessary information about the position, includ
ing piece locations, legal moves and other game states, in a structured 3D array.
2. **Move Representation (0 to 4096)**: Instead of using the UCI format, which is text-based, the move will be converted into a numerical value between 0 and 4096. This format is more efficient for the model to understand and process during training.

This **Notebook** will use our library to transform the data. This library is accessible in the project repository in Github.

We will share the **Notebook** with students in order to use it when they want and to explore the code or improve their own version of it. 

For the Web Application we need to convert to another format: **JSON** with FEN, Best move in UCI anf 4096 format and 77x8x8 Board in 77 x 64 bits integer format. This will be easy for the API of web application and easy for Javascript/Angular in the client. 

### CSV Dataset

This is the simplest way to share information. It's easy to view with any text editor or Excel. And it's easy to undestrand for humans and most Chess programs. 

Example:

```csv
fen,best
1rb5/4r3/3p1npb/3kp1P1/1P3P1P/5nR1/2Q1BK2/bN4NR w - - 3 61,c2c4
rn1q2n1/b3k1pr/pp1pB1Qp/2p1p1P1/2P1PP2/5R1P/P2P4/RNB1K3 w - - 1 24,g6f7
8/3r3k/NP1p4/p2QP1P1/1BB3Pp/1R4n1/6K1/5R2 w - - 5 82,d5g8
1nr1r3/n4Q2/P1kp2N1/2p3B1/1pp3P1/6P1/1R2P2R/K5N1 w - - 3 43,f7b7
7Q/3Bk3/2P1p3/4P2P/7b/5K2/B7/1b6 w - - 3 78,h8e8
8/4nB2/7k/3P2R1/p4p2/8/P4K2/6R1 w - - 3 78,g5h5
```

### Parquet Dataset

Parquet format is not plain text, so it's only for use in python programming. It's very compressed and optimal for massive data. Final datasets for AI training will be in this format. 

Inside parquet files we will have the same information than in CSV plus a 77x8x8 representation of board and legal moves; and a 4096 representation of the move. In training documentation is the explanation of this format and the transformation.


## Existing datasets

Datasets are public available at Kaggle. There are some:
* https://www.kaggle.com/datasets/xxjcaxx/chess-games-in-fen-and-best-move  All the data. It is too big to manage. We can find al raw and converted data.
* Selected data more than sufficient to train the CNN. Each file has more than 2 million of games converted and ready to train the AI model. We only used a fraction of first file to obtain a "good" model. We can use more data to fine tunning it. If we need even more data, we can use the other dataset. This dataset also includes the 25000 mate in one quality selected collection that can be used to fine tunning to reinforce mate performance of the model.  

## Making students Dataset

Web application of the project will help to create a store a dataset of selected games and best moves from our students. In order to work the other objectives of the project it will be available in CSV+FEN format. They will upload CSV to Github, Kaggle or Hugginface to import them in Colab and convert to Parquet format to train their own version of the model.


## Convert to 77x8x8 

This algorithm is used while converting to parquet and in inference during MCTS algorithms. We have tree versions: The first is the basic python and easy to understand. The second is an optimized version that reduces /2 the time during MCTS simulations:

### Basic algorithm:

```python
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

```

Optimized algorithms:

```python
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

```


The thirth version is a C++ super optimized for some MCTS. 