```python
from google.colab import drive
drive.mount('/content/drive')

!pip install python-chess
!pip install ipython

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pyarrow as pa
import pyarrow.parquet as pq

import torch
import random
from torch.utils.data import Dataset, DataLoader
import chess
import chess.svg
from IPython.display import display, SVG
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


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

```

```python

Chess_model = ChessNet  # Define class
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")
model = Chess_model().to(device)
model_route = '/content/drive/MyDrive/modelo_entrenado_chessintionv2.pth'
model.load_state_dict(torch.load(model_route,map_location=torch.device(device), weights_only=True))
```


### Auxiliar functions


First we need to declare some functions to convert from fen to matrix and to use the model:


```python
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
        layers.append(black_matrix)
    else:
        color_matrix = np.zeros((8,8))
        layers.append(black_matrix)

    board_rep = np.stack(layers)
    board_rep = board_rep.astype(bool)
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
print(codes)

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

# This function is to convert to a number, more simple.
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
    #print(move_code, code, from_row, from_col, uci_move )
    return uci_move

print(uci_to_number(chess.Move.from_uci('g1h3')))
print(number_to_uci(3646))

def concat_fen_legal(fen):
    board = chess.Board(fen)
    fen_matrix = board2rep(board)
    legal_moves = legal_moves_to_64_8_8(board)
    fen_matrix_legal_moves = np.concatenate((fen_matrix,legal_moves),0)
    fen_matrix_legal_moves = fen_matrix_legal_moves.reshape(-1)
    return fen_matrix_legal_moves


model.eval()
def get_best(board, mask=True):
    board_matrix = torch.tensor(board, dtype=torch.float32)
    board_matrix = board_matrix.unsqueeze(0)
    board_matrix = board_matrix.to(device)
    outputs = model(board_matrix)
    legal_moves_mask = board[-64:]
    legal_moves_mask = torch.tensor(legal_moves_mask.reshape(4096), dtype=torch.float32).to(device)
    if mask == True:
        outputs = outputs * legal_moves_mask
    pred = outputs.argmax(dim=1, keepdim=True)
    pred = pred.item()


    return {"pred": pred, "outputs":outputs}
```

    {(0, 1): 0, (1, 1): 1, (1, 0): 2, (1, -1): 3, (0, -1): 4, (-1, -1): 5, (-1, 0): 6, (-1, 1): 7, (0, 2): 8, (2, 2): 9, (2, 0): 10, (2, -2): 11, (0, -2): 12, (-2, -2): 13, (-2, 0): 14, (-2, 2): 15, (0, 3): 16, (3, 3): 17, (3, 0): 18, (3, -3): 19, (0, -3): 20, (-3, -3): 21, (-3, 0): 22, (-3, 3): 23, (0, 4): 24, (4, 4): 25, (4, 0): 26, (4, -4): 27, (0, -4): 28, (-4, -4): 29, (-4, 0): 30, (-4, 4): 31, (0, 5): 32, (5, 5): 33, (5, 0): 34, (5, -5): 35, (0, -5): 36, (-5, -5): 37, (-5, 0): 38, (-5, 5): 39, (0, 6): 40, (6, 6): 41, (6, 0): 42, (6, -6): 43, (0, -6): 44, (-6, -6): 45, (-6, 0): 46, (-6, 6): 47, (0, 7): 48, (7, 7): 49, (7, 0): 50, (7, -7): 51, (0, -7): 52, (-7, -7): 53, (-7, 0): 54, (-7, 7): 55, (1, 2): 56, (2, 1): 57, (2, -1): 58, (1, -2): 59, (-1, -2): 60, (-2, -1): 61, (-2, 1): 62, (-1, 2): 63}
    3646
    g1h3


# Visualizing results

We have trained a CNN based neural network AI to solve chess situations. But we don't know if it works well or not. We are going to analyze the results by various ways. It will be usefull to understand how this model works.

There are various way to analyze the prerformance:
- Quantitative analysis:
  - Compare with best move in Stockfish or real data
  - Top-k -> Is the best predicted move in the top 5 moves?
- Saliency & decision Interpretation
  - Heat maps from the layers to understand how they "see" the board
  - Saliency maps to understand important cells in decision.



## Visualizing initial board with legal moves:

> We are using a prepared dataset with 99 best moves by Stockfish.

The first exercise is analizing the algorithm of chess situation and legal moves codification to be undestanded by the CNN.

You can change `n_sample` value to see another example and see if you can detect in the black and white matrix below the meaing of the dots.

The dataset encodes the board into **84 channels**, where:  
- **Channels 0-11** represent **piece positions** (white and black pieces).  
- **Channel 12** represents **which player's turn it is**.  
- **Channels 13-76** represent **legal moves** (directional movement possibilities).  
- **Channels 77-84** are extra placeholders (set to `zeros` here).  

1. Can you identify where the pieces are in the first 12 matrices?  
2. Look at **Channel 12** (Turn Matrix). What value does it have?  
3. Look at **movement matrices** (13-76). What do the white and black dots represent?

Compare the **black and white matrices** to the chessboard (SVG representation) and answer:  
1. Do the **piece matrices (0-11)** match the FEN notation?  
2. Do the **movement matrices** align with how pieces move in chess?  
3. What do you notice in **Knight move matrices** compared to Bishop or Rook movements?  


```python

import matplotlib.pyplot as plt
import matplotlib

df_linchess = pd.read_parquet('/content/drive/MyDrive/linchess_converted_stockfish99.parquet.gzip', engine="pyarrow")
df_linchess['board'] = df_linchess['board'].apply(lambda board: board.reshape(77, 8, 8).astype(int))

n_sample = 95

matriz = df_linchess.loc[n_sample,'board']
fen = df_linchess.loc[n_sample,'fen_original']
print("Fen: ",fen)
print("Turn matrix: ")
print(matriz[12])
print(matriz.shape)

board = chess.Board(df_linchess.loc[n_sample, 'fen_original'])
svg_board = chess.svg.board(board=board, size=300)
display(SVG(svg_board))

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

filas = 7
columnas = 12
fig, axs = plt.subplots(filas, columnas, figsize=(10, 10))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
for i in range(84):
    fila = i // columnas
    columna = i % columnas
    canal = matriz[i] if i < 77 else np.zeros((8, 8))
    axs[fila, columna].imshow(canal, cmap='gray', vmin=0, vmax=1)
    axs[fila, columna].axis('off')  # Desactiva los ejes para una mejor visualización
    axs[fila, columna].text(4, 9, labels[i], fontsize=6, ha='center', va='top')
plt.show()
```

    Fen:  r3k2r/1bqp1pp1/p1nbpn1p/8/Pp2PP2/1B1QB2N/1PPN2PP/R3K2R b KQkq - 0 13
    Turn matrix: 
    [[0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0]]
    (77, 8, 8)



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_6_1.svg)
    



    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_6_2.png)
    


## Quantitative analysis

We have a **chess AI** that predicts the best move to play in a given position. However, we also have **Stockfish**, one of the strongest chess engines, which gives us a list of the **best possible moves**.  

This graph helps us understand:  
1. **How often the AI's predicted move matches Stockfish’s best moves.**  
2. **If the AI’s move is completely different from what Stockfish suggests.**  
3. **Whether the AI’s move matches what human players typically play.**  


Understanding the graph:
1. If most bars are high for **Rank 1**, the AI is picking very strong moves, just like Stockfish. ✅  
2. If the AI's moves often rank **2nd, 3rd, 4th, or 5th**, it means the AI is good but not always perfect. 🤔  
3. If many moves fall **outside Stockfish’s top choices**, the AI might be weak or playing a different style. ❌  
4. If the AI’s choices match **human best moves** (green line), it might be playing in a more human-like way rather than trying to be as perfect as Stockfish. 🧠  



```python
df_linchess
```





  <div id="df-58732166-49e3-4c73-ad2f-31112ac00f1a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>board</th>
      <th>best</th>
      <th>fen_original</th>
      <th>best_uci</th>
      <th>sf_best</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1,...</td>
      <td>140</td>
      <td>4rbk1/1pp1qPpn/p1n4p/3r4/P7/5N1P/1P2QPP1/R1B1R...</td>
      <td>e7f7</td>
      <td>[g8f7, e7f7, g8h8]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>4077</td>
      <td>r2qkbnr/1pp1pppp/p1b5/3p4/3P4/4PN2/PPP2PPP/RNB...</td>
      <td>f3e5</td>
      <td>[f3e5, b2b3, e1g1, d1e2, a2a4]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>776</td>
      <td>rn1qkbnr/ppp2ppp/3p4/4p3/4P3/1PN5/1PPP1PPP/R1B...</td>
      <td>a7a5</td>
      <td>[g8f6, b8d7, c7c6, f8e7, b8c6]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>20</td>
      <td>r7/1p2q2p/p1n1Qbk1/5p2/5B1P/PN6/1P4P1/R6K w - ...</td>
      <td>e6e7</td>
      <td>[h4h5, e6e7, b3c5, a1e1, e6d5]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>331</td>
      <td>r3k2r/1p1bnp1p/p3p1p1/3q4/3P4/8/PP3PPP/RNBQR1K...</td>
      <td>d7c6</td>
      <td>[d7c6, e7f5, d5h5, d5d6, d5a5]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>272</td>
      <td>r3k2r/1bqp1pp1/p1nbpn1p/8/Pp2PP2/1B1QB2N/1PPN2...</td>
      <td>a6a5</td>
      <td>[d6e7, d6f8, c6a5, e8g8, a8c8]</td>
    </tr>
    <tr>
      <th>96</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>3682</td>
      <td>3Q4/7p/8/7k/2N2pr1/1B6/1PP2K2/8 w - - 3 55</td>
      <td>c4e5</td>
      <td>[d8e7, c4e5, d8f6, d8d7, c2c3]</td>
    </tr>
    <tr>
      <th>97</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>778</td>
      <td>rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR ...</td>
      <td>c7c5</td>
      <td>[c7c5, e7e5, c7c6, e7e6, b8c6]</td>
    </tr>
    <tr>
      <th>98</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>957</td>
      <td>r2r2k1/2pB1ppp/p1Pb4/1p6/6Q1/qP2B2P/P1R2PP1/5R...</td>
      <td>f1d1</td>
      <td>[f1d1, e3c1, c2d2, c2e2, g2g3]</td>
    </tr>
    <tr>
      <th>99</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>55</td>
      <td>5rn1/pp4k1/2p1p3/5n1R/3PN3/2P5/PP4PP/5RK1 w - ...</td>
      <td>h2h3</td>
      <td>[e4c5, e4g5, h5g5, f1f4, f1f3]</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-58732166-49e3-4c73-ad2f-31112ac00f1a')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-58732166-49e3-4c73-ad2f-31112ac00f1a button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-58732166-49e3-4c73-ad2f-31112ac00f1a');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-629b49a4-e399-43a6-917a-e913e840795f">
  <button class="colab-df-quickchart" onclick="quickchart('df-629b49a4-e399-43a6-917a-e913e840795f')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-629b49a4-e399-43a6-917a-e913e840795f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_0c49cd49-a5dc-4033-8fd4-d6f0a2dfa897">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_linchess')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_0c49cd49-a5dc-4033-8fd4-d6f0a2dfa897 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_linchess');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
df_linchess['predicted_best_move'] = None

# Iterate through the rows and predict best move for each chess position
for index, row in df_linchess.iterrows():
    try:
        board = row['board']
        predicted_move = get_best(board)
        df_linchess.loc[index, 'predicted_best_move'] = number_to_uci(predicted_move['outputs'].argmax(dim=1, keepdim=True).item())
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        df_linchess.loc[index, 'predicted_best_move'] = "Error" # Or handle the error as you see fit

#Example of how to use the new column
df_linchess[['fen_original', 'predicted_best_move','sf_best']]

```





  <div id="df-9d9dc774-4701-4a28-a759-5cea434a6d83" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fen_original</th>
      <th>predicted_best_move</th>
      <th>sf_best</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4rbk1/1pp1qPpn/p1n4p/3r4/P7/5N1P/1P2QPP1/R1B1R...</td>
      <td>g8h8</td>
      <td>[g8f7, e7f7, g8h8]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>r2qkbnr/1pp1pppp/p1b5/3p4/3P4/4PN2/PPP2PPP/RNB...</td>
      <td>f3e5</td>
      <td>[f3e5, b2b3, e1g1, d1e2, a2a4]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>rn1qkbnr/ppp2ppp/3p4/4p3/4P3/1PN5/1PPP1PPP/R1B...</td>
      <td>g8f6</td>
      <td>[g8f6, b8d7, c7c6, f8e7, b8c6]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>r7/1p2q2p/p1n1Qbk1/5p2/5B1P/PN6/1P4P1/R6K w - ...</td>
      <td>e6e7</td>
      <td>[h4h5, e6e7, b3c5, a1e1, e6d5]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>r3k2r/1p1bnp1p/p3p1p1/3q4/3P4/8/PP3PPP/RNBQR1K...</td>
      <td>e8g8</td>
      <td>[d7c6, e7f5, d5h5, d5d6, d5a5]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>r3k2r/1bqp1pp1/p1nbpn1p/8/Pp2PP2/1B1QB2N/1PPN2...</td>
      <td>e8g8</td>
      <td>[d6e7, d6f8, c6a5, e8g8, a8c8]</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3Q4/7p/8/7k/2N2pr1/1B6/1PP2K2/8 w - - 3 55</td>
      <td>d8d4</td>
      <td>[d8e7, c4e5, d8f6, d8d7, c2c3]</td>
    </tr>
    <tr>
      <th>97</th>
      <td>rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR ...</td>
      <td>b7b6</td>
      <td>[c7c5, e7e5, c7c6, e7e6, b8c6]</td>
    </tr>
    <tr>
      <th>98</th>
      <td>r2r2k1/2pB1ppp/p1Pb4/1p6/6Q1/qP2B2P/P1R2PP1/5R...</td>
      <td>f1c1</td>
      <td>[f1d1, e3c1, c2d2, c2e2, g2g3]</td>
    </tr>
    <tr>
      <th>99</th>
      <td>5rn1/pp4k1/2p1p3/5n1R/3PN3/2P5/PP4PP/5RK1 w - ...</td>
      <td>h5g5</td>
      <td>[e4c5, e4g5, h5g5, f1f4, f1f3]</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9d9dc774-4701-4a28-a759-5cea434a6d83')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9d9dc774-4701-4a28-a759-5cea434a6d83 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9d9dc774-4701-4a28-a759-5cea434a6d83');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-3f83b239-f518-4294-9ae1-4cb0b7533c1a">
  <button class="colab-df-quickchart" onclick="quickchart('df-3f83b239-f518-4294-9ae1-4cb0b7533c1a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-3f83b239-f518-4294-9ae1-4cb0b7533c1a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import chess
import chess.svg
from IPython.display import SVG

def plot_move_rank_distribution(rank_counts, not_in_sf_best, in_human_best, in_human_and_sf_best, total_fail, total_sf_best):
    ranks = list(rank_counts.keys())
    counts = list(rank_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(ranks, counts, color='royalblue', label='Predicted Move Rank in sf_best')

    # Add data labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, str(int(yval)), ha='center', fontsize=10)

    # Stacked bars for Human Best & Human + SF Best
    plt.bar(max(ranks) + 1, in_human_best, color='green', label='In Human Best')
    plt.bar(max(ranks) + 1, total_sf_best - in_human_best, bottom=in_human_best, color='purple', label='total sf')
    plt.bar(max(ranks) + 1, in_human_and_sf_best - total_sf_best , bottom=total_sf_best, color='orange', label='Human +sf')


    plt.bar(max(ranks) + 2, not_in_sf_best, color='red', label='Not in sf_best')
    plt.bar(max(ranks) + 2, total_fail, color='black', label='Total Fail')

    plt.xlabel('Rank in Stockfish Best Moves')
    plt.ylabel('Number of Predictions')
    plt.title('Distribution of AI-Predicted Moves Compared to Stockfish')
    plt.xticks(ranks + [max(ranks) + 1, max(ranks) + 2], labels=ranks + ['Human and SF', '!FS, total fail'])
    plt.legend()
    plt.show()



# Create a dictionary to store the counts of predicted moves in each position
rank_counts = {}
for i in range(1, 6):  # Assuming a maximum of 99 positions
    rank_counts[i] = 0

not_in_sf_best = 0
in_human_best = 0
in_human_and_sf_best = 0
total_fail = 0

for index, row in df_linchess.iterrows():
    predicted_move = row['predicted_best_move']
    sf_best_moves = row['sf_best']
    human_best_move = row['best']

    if predicted_move != "Error":
      try:
        # Convert the NumPy array to a list to use the index() method
        rank = sf_best_moves.tolist().index(predicted_move) + 1
        if rank in rank_counts:
          rank_counts[rank] += 1
        else:
          print("Move not found in sf_best at a valid position!")
      except ValueError:
          not_in_sf_best += 1
    else:
      not_in_sf_best += 1
    if predicted_move == number_to_uci(human_best_move):
      in_human_best += 1
    if predicted_move == number_to_uci(human_best_move) or predicted_move in sf_best_moves:
      in_human_and_sf_best += 1
    if predicted_move != number_to_uci(human_best_move) and not predicted_move in sf_best_moves:
      total_fail += 1

#rank_counts[6] = not_in_sf_best
#rank_counts[7] = in_human_best

total_sf_best = sum(rank_counts.values())
#print(f"Total positions in sf_best: {total_sf_best}")
print(total_sf_best,in_human_best,in_human_and_sf_best)

# Plotting
ranks = list(rank_counts.keys())
counts = list(rank_counts.values())


plot_move_rank_distribution(rank_counts, not_in_sf_best, in_human_best, in_human_and_sf_best, total_fail, total_sf_best)

```

    59 31 60



    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_10_1.png)
    


The heatmap of mistakes visually represents where the AI makes the most errors on the chessboard. Each square's color intensity corresponds to the number of times the AI chose a move starting from that square that did not match Stockfish's best moves. Darker or more intense red squares indicate positions where the AI frequently makes mistakes, while lighter or blank squares suggest areas with fewer errors. This helps identify patterns in the AI’s decision-making, such as whether it struggles more in the opening, middlegame, or specific piece movements. By analyzing this heatmap, we can better understand the model's weaknesses and areas for improvement.


```python
# Heatmap Visualization
def plot_heatmap_from_mistakes(df_linchess):
    """
    Extracts mistakes from DataFrame and plots a heatmap.
    """
    mistake_counts = {}

    for index, row in df_linchess.iterrows():
        predicted_move = row['predicted_best_move']
        sf_best_moves = row['sf_best']

        if predicted_move != "Error" and predicted_move not in sf_best_moves:
            try:
                start_square = chess.parse_square(predicted_move[:2])  # Extract start square
                if start_square in mistake_counts:
                    mistake_counts[start_square] += 1
                else:
                    mistake_counts[start_square] = 1
            except:
                pass  # Handle invalid move formats

    board_array = np.zeros((8, 8))  # Initialize 8x8 board representation

    # Fill in mistake data
    for square, count in mistake_counts.items():
        row, col = divmod(square, 8)  # Convert chess square index to row/col
        board_array[7 - row, col] = count  # Flip row for correct board orientation

    plt.figure(figsize=(8, 8))
    sns.heatmap(board_array, annot=True, fmt='g', cmap='Reds', linewidths=0.5, square=True, cbar=True)
    plt.title("AI Move Mistakes Heatmap")
    plt.xlabel("File (a-h)")
    plt.ylabel("Rank (1-8)")
    plt.xticks(ticks=np.arange(8) + 0.5, labels=list("abcdefgh"))
    plt.yticks(ticks=np.arange(8) + 0.5, labels=list("87654321"))
    plt.show()


plot_heatmap_from_mistakes(df_linchess)

```


    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_12_0.png)
    


This 100 examples by stockfish is insufficient to make a good mistake heat map, we will use a bigger dataset with human best:


```python
def read_data(file,page,size):
    with pq.ParquetFile(file) as pf:
        print("reading",file, page, size, pf.metadata)

        iterb = pf.iter_batches(batch_size = size)
        for i in range(page):
            next(iterb)

        batches = next(iterb)

        # Construir el DataFrame
        df_chess = pa.Table.from_batches([batches]).to_pandas()

        batches = None
        iterb = None

        # reshape
        df_chess['board'] = df_chess['board'].apply(lambda board: board.reshape(77, 8, 8).astype(int))
        df_chess['predicted_best_move'] = df_chess['board'].apply(lambda board: number_to_uci(get_best(board)['outputs'].argmax(dim=1, keepdim=True).item()))
        df_chess.info(memory_usage='deep')
        df_chess.memory_usage(deep=True)
        return df_chess

df_chess = read_data('/content/drive/MyDrive/linchesgamesconverted0.parquet.gz',0,20000)
```

    reading /content/drive/MyDrive/linchesgamesconverted0.parquet.gz 0 20000 <pyarrow._parquet.FileMetaData object at 0x7e2555174310>
      created_by: parquet-cpp-arrow version 14.0.2
      num_columns: 4
      num_rows: 2000000
      num_row_groups: 2
      format_version: 2.6
      serialized_size: 3526
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20000 entries, 0 to 19999
    Data columns (total 5 columns):
     #   Column               Non-Null Count  Dtype 
    ---  ------               --------------  ----- 
     0   board                20000 non-null  object
     1   best                 20000 non-null  int64 
     2   fen_original         20000 non-null  object
     3   best_uci             20000 non-null  object
     4   predicted_best_move  20000 non-null  object
    dtypes: int64(1), object(4)
    memory usage: 759.5 MB



```python
df_chess
```





  <div id="df-6635990a-5620-47a6-bee4-b9ef11febab1" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>board</th>
      <th>best</th>
      <th>fen_original</th>
      <th>best_uci</th>
      <th>predicted_best_move</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>563</td>
      <td>rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w ...</td>
      <td>d2d4</td>
      <td>b2b3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>779</td>
      <td>rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR ...</td>
      <td>d7d5</td>
      <td>b7b6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>52</td>
      <td>rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBN...</td>
      <td>e2e3</td>
      <td>e2e3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>268</td>
      <td>rnbqkbnr/ppp1pppp/8/3p4/3P4/4P3/PPP2PPP/RNBQKB...</td>
      <td>e7e6</td>
      <td>b8c6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>55</td>
      <td>rnbqkbnr/ppp2ppp/4p3/3p4/3P4/4P3/PPP2PPP/RNBQK...</td>
      <td>h2h3</td>
      <td>c2c4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>3853</td>
      <td>8/5n2/8/2Q3pk/P2P1p2/8/1P4P1/6K1 b - - 4 63</td>
      <td>f7e5</td>
      <td>h5g4</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>666</td>
      <td>8/8/8/2Q1n1pk/P2P1p2/8/1P4P1/6K1 w - - 5 64</td>
      <td>c5e5</td>
      <td>c5e5</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>351</td>
      <td>8/8/8/4Q1pk/P2P1p2/8/1P4P1/6K1 b - - 0 64</td>
      <td>h5g4</td>
      <td>h5g4</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>156</td>
      <td>8/8/8/4Q1p1/P2P1pk1/8/1P4P1/6K1 w - - 1 65</td>
      <td>e5f5</td>
      <td>e5e7</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0,...</td>
      <td>102</td>
      <td>8/8/8/5Qp1/P2P1pk1/8/1P4P1/6K1 b - - 2 65</td>
      <td>g4h5</td>
      <td>g4f5</td>
    </tr>
  </tbody>
</table>
<p>2000 rows × 5 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-6635990a-5620-47a6-bee4-b9ef11febab1')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-6635990a-5620-47a6-bee4-b9ef11febab1 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-6635990a-5620-47a6-bee4-b9ef11febab1');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-72ec1578-35bd-4855-8857-097d8536745a">
  <button class="colab-df-quickchart" onclick="quickchart('df-72ec1578-35bd-4855-8857-097d8536745a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-72ec1578-35bd-4855-8857-097d8536745a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_d2d1138e-e36b-444d-9e9b-7e82da4b83d3">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('df_chess')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_d2d1138e-e36b-444d-9e9b-7e82da4b83d3 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('df_chess');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python

def plot_heatmap_from_mistakes_human(df_linchess):
    """
    Extracts mistakes from DataFrame and plots a heatmap.
    """
    mistake_counts = {}

    for index, row in df_linchess.iterrows():
        predicted_move = row['predicted_best_move']
        best_moves = number_to_uci(row['best'])

        if predicted_move != "Error" and predicted_move != best_moves:
            try:
                start_square = chess.parse_square(predicted_move[:2])  # Extract start square
                if start_square in mistake_counts:
                    mistake_counts[start_square] += 1
                else:
                    mistake_counts[start_square] = 1
            except:
                pass  # Handle invalid move formats

    board_array = np.zeros((8, 8))  # Initialize 8x8 board representation

    # Fill in mistake data
    for square, count in mistake_counts.items():
        row, col = divmod(square, 8)  # Convert chess square index to row/col
        board_array[7 - row, col] = count  # Flip row for correct board orientation

    plt.figure(figsize=(8, 8))
    sns.heatmap(board_array, annot=True, fmt='g', cmap='Reds', linewidths=0.5, square=True, cbar=True)
    plt.title("AI Move Mistakes Heatmap")
    plt.xlabel("File (a-h)")
    plt.ylabel("Rank (1-8)")
    plt.xticks(ticks=np.arange(8) + 0.5, labels=list("abcdefgh"))
    plt.yticks(ticks=np.arange(8) + 0.5, labels=list("87654321"))
    plt.show()


plot_heatmap_from_mistakes_human(df_chess)




```


    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_16_0.png)
    


## Visualizing pieces heat map

The code provided in the next cells provides an **interactive way** to understand how **Convolutional Neural Networks (CNNs)** can evaluate and predict chess moves. It allows to **see the decision-making process** of a chess-playing AI, bridging the gap between **deep learning** and **game strategy**.


- The neural network used here **does not see a chessboard as humans do**.  
- Instead, it processes **matrices (images)** that represent **pieces, turns, and possible moves**.  
- Each channel in the `matriz` represents a **different feature**, such as:  
  - Which pieces are on the board (12 channels for each piece type).  
  - Whether it’s **White’s or Black’s turn** (1 channel).  
  - The **possible legal moves** (many channels).  

  By visualizing **these matrices as heat maps**, we can understand **where the CNN is focusing**.
- A **heat map** is a **visual representation of how much the CNN values a move**.  
- The **brighter areas** show squares the CNN thinks are **strong candidates** for the next move.  
- This helps us see **what the AI considers important** in a chess position.  

You can compare:
1. The **raw CNN output** (`example_best`).
2. The **filtered legal move CNN output** (`example_best_legal`).
3. The **Stockfish move** (classical chess engine).

You can analyze:
- Does the CNN **understand legal moves**, or does it suggest **illegal** ones?  
- How does the CNN’s move choice **compare to Stockfish**, a traditional chess engine?  
- When do the **best predicted moves** match Stockfish’s **optimal** move?

You can detect if the CNN doesn't works well:
- If `"No coincidence!"` appears often, it means the **CNN predicts illegal moves** too frequently.  
- If the CNN often **disagrees with Stockfish**, it may not be **strong enough** to play at a high level.  
- The heat map shows **whether the CNN understands piece activity correctly**.


```python
def visualize_heat_maps(sample_n):

    sample_list = df_linchess.sample(n=sample_n).index

    for sample_idx, sample in enumerate(sample_list):
        matriz = df_linchess.loc[sample, 'board']
        board = chess.Board(df_linchess.loc[sample, 'fen_original'])

        example_best = get_best(matriz,mask=False)  # Without legal moves mask to show all results in heat map
        example_best_legal = get_best(matriz,mask=True)  # With mask to extract real legal move

        print("NN Best",number_to_uci(example_best['outputs'].argmax(dim=1, keepdim=True).item()),
              "NN Best Legal",number_to_uci(example_best_legal['outputs'].argmax(dim=1, keepdim=True).item()),
              "Stockfish Best",  df_linchess.loc[sample,'sf_best']
             )
        imagen_final = np.zeros((8,8))
        for idx,i in  enumerate(example_best['outputs'].cpu().detach().numpy()[0]):
            if(abs(i) > 0.2):
                (code, x, y) = np.unravel_index(idx, (64, 8, 8))
                imagen_final[x,y] = imagen_final[x,y] if imagen_final[x,y] > i else i

        imagen_final_normalized = (imagen_final - imagen_final.min()) / (imagen_final.max() - imagen_final.min())

        cmap = matplotlib.colormaps['viridis']
        fill = {}
        for i in range(8):
            for j in range(8):
                fill[(7-i)*8 + j] = '#%02x%02x%02x%02x' % tuple([round(255*x) for x in cmap(imagen_final_normalized[i,j])])

        final_position_best = chess.Move.from_uci(number_to_uci(example_best['outputs'].argmax(dim=1, keepdim=True).item()))
        final_position_best_legal = chess.Move.from_uci(number_to_uci(example_best_legal['outputs'].argmax(dim=1, keepdim=True).item()))
        final_position_best_sf = chess.Move.from_uci(df_linchess.loc[sample,'sf_best'][0])

        arrows=[
            chess.svg.Arrow(final_position_best.from_square, final_position_best.to_square, color="#ffcccc"),
            chess.svg.Arrow(final_position_best_legal.from_square, final_position_best_legal.to_square, color="#ff5555"),
            chess.svg.Arrow(final_position_best_sf.from_square, final_position_best_sf.to_square, color="#ccffcc")
        ]

        if final_position_best.to_square != final_position_best_legal.to_square:
            print("No coincidence!")

        svg_board = chess.svg.board(board=board, fill=fill, arrows=arrows, size=300)
        display(SVG(svg_board))

visualize_heat_maps(10)
```

    NN Best f3d4 NN Best Legal f3d4 Stockfish Best ['f3d4' 'c4c5' 'c4b5' 'b3a2' 'b3d1']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_1.svg)
    


    NN Best h6f7 NN Best Legal h6f7 Stockfish Best ['b8c7' 'h6g4' 'a6a5' 'h8h7' 'b8a7']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_3.svg)
    


    NN Best f7d5 NN Best Legal f7d5 Stockfish Best ['f7b3' 'f7d5' 'f7c4' 'f7h5' 'a2a3']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_5.svg)
    


    NN Best f2f3 NN Best Legal f2f3 Stockfish Best ['f2f3' 'f2f1' 'f2e1' 'f2g1']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_7.svg)
    


    NN Best e8c8 NN Best Legal e8c8 Stockfish Best ['c6d4' 'f8f7' 'd7g4' 'h7h6' 'a8c8']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_9.svg)
    


    NN Best d2d3 NN Best Legal d2d3 Stockfish Best ['g1e2' 'd2d3' 'c2c3' 'b1c3' 'c4b3']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_11.svg)
    


    NN Best e2e3 NN Best Legal e2e3 Stockfish Best ['c4f7' 'e2e1' 'e2e4' 'e2f3' 'e2e3']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_13.svg)
    


    NN Best d2d4 NN Best Legal d2d4 Stockfish Best ['d2d4' 'e3e4' 'h2h4' 'f1c4' 'a2a4']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_15.svg)
    


    NN Best c3b4 NN Best Legal c3b4 Stockfish Best ['g3g4' 'c3d3' 'h3h5' 'f2g2' 'f2g1']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_17.svg)
    


    NN Best f3f4 NN Best Legal f3f4 Stockfish Best ['d3a3' 'd3c3' 'd3d5' 'f3e3' 'd3b3']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_19.svg)
    


## Visualising NN best moves heat map

This code is an **advanced visualization tool** that helps us **analyze how a Convolutional Neural Network (CNN) processes chess positions**. It allows us to **see inside the neural network** and understand how it extracts information at different layers.  

- Each convolutional layer captures **different levels of abstraction**:  
  1. **Early layers (`conv1`)** → Detect **edges, shapes, and textures**.  
  2. **Mid layers (`conv2` and `conv3`)** → Recognize **piece positions and movement patterns**.  
  3. **Deeper layers (`conv4`)** → Identify **strategic factors like threats and control over squares**.  
- By visualizing these layers, we see **how the CNN gradually builds its understanding** of a chess position.

- The heat maps show **where the CNN is focusing** when analyzing the chessboard.  
- Brighter areas in the heat maps **indicate important regions** for the AI’s decision-making.  
- This can help us **interpret why the AI chooses certain moves**.  
- Unlike humans, CNNs don’t know **explicit chess rules**.  
- Instead, they **learn patterns** from data, recognizing important squares and piece activity.  
- The heat maps **reveal what the model considers important** in a given position.  
- **Move Predictions Can Be Explained** → If the CNN picks a bad move, heat maps help diagnose **why it made that mistake**.  
- **Helps Debug and Improve the Model** → If heat maps don’t match human intuition, the model may need **better training data**.  

You can play with this code:
1. **Compare heat maps for different positions.** Does the AI focus more on **attacks**, **defense**, or **central control**?  
2. **Modify the CNN architecture.** Does adding more layers improve the move predictions?  
3. **Test different chess positions.** How do heat maps change in **endgames vs. middlegames**?  




```python
# https://kozodoi.me/blog/20210527/extracting-features
import matplotlib.pyplot as plt
import matplotlib
sample = 4  # 3 are interesting
features = {}

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook

model.conv1.register_forward_hook(get_features("feats1"))
model.conv2.register_forward_hook(get_features("feats2"))
model.conv3.register_forward_hook(get_features("feats3"))
model.conv4.register_forward_hook(get_features("feats4"))

matriz =  df_linchess.loc[sample,'board']
board = chess.Board(df_linchess.loc[sample,'fen_original'])

example_best_legal = get_best(matriz,mask=True)  # With mask to extract real legal move
best = number_to_uci(example_best_legal['outputs'].argmax(dim=1, keepdim=True).item())

print("NN Best",best, "Stockfish Best",  df_linchess.loc[sample,'sf_best'])
svg_board = chess.svg.board(board=board, size=300)
display(SVG(svg_board))
#features['feats'].shape
print(features['feats1'].shape)
print(features['feats2'].shape)
print(features['feats3'].shape)
print(features['feats4'].shape)

matriz1 = np.sum(features['feats1'][0].cpu().numpy(),axis=0)
matriz2 = np.sum(features['feats2'][0].cpu().numpy(),axis=0)
matriz3 = np.sum(features['feats3'][0].cpu().numpy(),axis=0)
matriz4 = np.sum(features['feats4'][0].cpu().numpy(),axis=0)
print(matriz1.shape)


files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

def plot_heatmap(matriz, title):
    plt.figure(figsize=(3.5, 3.5))
    plt.imshow(matriz, cmap='inferno', origin='upper')  # 'upper' so rank 1 is at the bottom
    #plt.colorbar()

    # Set axis labels
    plt.xticks(ticks=np.arange(8), labels=files, fontsize=12)  # a-h
    plt.yticks(ticks=np.arange(8), labels=reversed(ranks), fontsize=12)  # 1-8 (reversed)

    plt.title(title, fontsize=14)
    plt.show()

# Plot all heatmaps with chessboard labels
plot_heatmap(matriz1, "Feature Map - Conv1")
plot_heatmap(matriz2, "Feature Map - Conv2")
plot_heatmap(matriz3, "Feature Map - Conv3")
plot_heatmap(matriz4, "Feature Map - Conv4")
```

    NN Best e8g8 Stockfish Best ['d7c6' 'e7f5' 'd5h5' 'd5d6' 'd5a5']



    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_21_1.svg)
    


    torch.Size([1, 128, 8, 8])
    torch.Size([1, 256, 8, 8])
    torch.Size([1, 512, 8, 8])
    torch.Size([1, 1024, 8, 8])
    (8, 8)



    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_21_3.png)
    



    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_21_4.png)
    



    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_21_5.png)
    



    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_21_6.png)
    



```python
import matplotlib.pyplot as plt
import numpy as np
import chess
import chess.svg
from IPython.display import display, SVG

sample = 4
features = {}

def get_features(name):
    """Hook function to extract features from each layer of the CNN"""
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

# Register hooks to each layer
model.conv1.register_forward_hook(get_features("feats1"))
model.conv2.register_forward_hook(get_features("feats2"))
model.conv3.register_forward_hook(get_features("feats3"))
model.conv4.register_forward_hook(get_features("feats4"))
model.fc1.register_forward_hook(get_features("fc1"))
model.fc2.register_forward_hook(get_features("fc2"))

# Get the board and the neural network prediction
matriz = df_linchess.loc[sample, 'board']
board = chess.Board(df_linchess.loc[sample, 'fen_original'])

example_best_legal = get_best(matriz, mask=True)  # With mask to extract real legal move
best_move = number_to_uci(example_best_legal['outputs'].argmax(dim=1, keepdim=True).item())

svg_board = chess.svg.board(board=board, size=300)
display(SVG(svg_board))

print("NN Best Move:", best_move)

# Get feature maps from the CNN layers
matriz1 = features['feats1'][0].cpu().numpy()  # First convolutional layer feature map
matriz2 = features['feats2'][0].cpu().numpy()  # Second convolutional layer feature map
matriz3 = features['feats3'][0].cpu().numpy()  # Third convolutional layer feature map
matriz4 = features['feats4'][0].cpu().numpy()  # Fourth convolutional layer feature map
fc1_activations = features['fc1'].cpu().numpy()
fc2_activations = features['fc2'].cpu().numpy()

# Function to reduce 3D feature map (channels x height x width) to 2D by averaging across channels
def reduce_feature_map(feature_map):
    return np.mean(feature_map, axis=0)

# Reduce feature maps to 2D (average over channels)
matriz1_reduced = reduce_feature_map(matriz1)
matriz2_reduced = reduce_feature_map(matriz2)
matriz3_reduced = reduce_feature_map(matriz3)
matriz4_reduced = reduce_feature_map(matriz4)


# Chessboard coordinates (files and ranks for axis labeling)
files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

# Function to plot the feature map with inverted colormap for better visualization of important activations
def plot_feature_map_with_top_k_inverted(feature_map, title, k=5, cmap='viridis'):
    """Plot the feature map and highlight the top activations with inverted colormap."""
    plt.figure(figsize=(3, 3))

    # Normalize the feature map
    feature_map_normalized = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))

    # Invert the feature map for clearer visual representation (important areas become warm colors)
    feature_map_normalized = 1 - feature_map_normalized  # Invert the colors

    # Identify the top-k coordinates (highest activations) after normalization and inversion
    top_k_coords = np.unravel_index(np.argsort(feature_map_normalized.flatten())[-k:], feature_map_normalized.shape)

    # Plot the heatmap with inverted colormap
    plt.imshow(feature_map_normalized, cmap=cmap, origin='upper', vmin=0, vmax=1)

    # Set axis labels
    plt.xticks(ticks=np.arange(8), labels=files, fontsize=12)
    plt.yticks(ticks=np.arange(8), labels=reversed(ranks), fontsize=12)

    # Highlight top-k activations with circles
    for i in range(len(top_k_coords[0])):
        plt.scatter(top_k_coords[1][i], top_k_coords[0][i], s=150, edgecolor='red', facecolor='none', lw=2)

    plt.show()

# Plot feature maps with top-k activations after normalization and inversion
plot_feature_map_with_top_k_inverted(matriz1_reduced, "Feature Map - Conv1", k=5)
plot_feature_map_with_top_k_inverted(matriz2_reduced, "Feature Map - Conv2", k=5)
plot_feature_map_with_top_k_inverted(matriz3_reduced, "Feature Map - Conv3", k=5)
plot_feature_map_with_top_k_inverted(matriz4_reduced, "Feature Map - Conv4", k=5)


output_predictions = example_best_legal['outputs'][0].cpu().detach().numpy()
# Get top-k moves (highest predicted moves)
k=5
top_k = output_predictions.argsort()[-k:][::-1]

# Print the top-k moves and their corresponding probabilities/scores
for idx in top_k:
    move = number_to_uci(idx)
    score = output_predictions[idx]
    print(f"Number: {idx} Move: {move}, Score: {score}")
```


    
![svg](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_22_0.svg)
    


    NN Best Move: e8g8



    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_22_2.png)
    



    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_22_3.png)
    



    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_22_4.png)
    



    
![png](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_22_5.png)
    


    Number: 644 Move: e8g8, Score: 18.2082576751709
    Number: 3788 Move: e7f5, Score: 16.302738189697266
    Number: 3916 Move: e7c6, Score: 15.7904691696167
    Number: 777 Move: b7b5, Score: 15.535648345947266
    Number: 640 Move: a8c8, Score: 15.155900001525879


## **Understanding Saliency Maps**
Saliency maps are a technique used in deep learning to highlight the most important features of an input that contribute to a model's decision. They are widely used in **computer vision, natural language processing, and interpretability in AI**. In this case, they help to understand which **squares** on a chessboard influence the neural network's decision when choosing a move.

A **saliency map** is a visualization of how **sensitive** a model's output is to changes in the input.

In the following function, the saliency map highlights which squares were most influential when the model decided on a move. This helps in:
- **Understanding model decisions:** Why did the NN choose a move?
- **Comparing NN and Stockfish:** Does the NN focus on similar squares as Stockfish?
- **Debugging the Model:** Are there irrelevant squares with high saliency? If so, the model might be overfitting.
- **Saliency maps only show sensitivity, not causality.**  
  - Just because a square is highlighted does not mean it "caused" the decision.
- **No distinction between positive or negative influence.**  
  - The map shows **importance**, not whether it is a positive or negative factor.
- **They are not always reliable.**  
  - Some pixels (squares) might show high importance due to noise.


```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import chess
import chess.svg
from IPython.display import SVG
import cv2

# Function to compute saliency map
def compute_saliency_map(model, input_tensor, move_index):
    """
    Computes the saliency map for a given chess position.

    Args:
    - model: The trained CNN model
    - input_tensor: The chess position input tensor (1, 77, 8, 8)
    - move_index: The predicted move index

    Returns:
    - saliency_map: 8x8 array with importance values for each square
    """
    model.eval()  # Set model to evaluation mode

    # Ensure input_tensor is a leaf tensor
    input_tensor = input_tensor.detach().clone().requires_grad_(True)
    input_tensor.retain_grad()  # Retain gradients for non-leaf tensors

    # Move to model's device
    input_tensor = input_tensor.to(next(model.parameters()).device)

    # Forward pass
    output = model(input_tensor)

    # Ensure move_index is an integer tensor
    if isinstance(move_index, torch.Tensor):
        move_index = move_index.item()

    score = output[0, move_index]  # Select the output corresponding to the move

    # Backpropagation to get gradients
    model.zero_grad()
    score.backward()

    # Ensure gradients exist
    if input_tensor.grad is None:
        raise RuntimeError("Gradients are not being computed. Check model architecture and requires_grad settings.")

    # Get absolute gradients (importance of each pixel)
    saliency = input_tensor.grad.abs().detach().cpu().numpy()[0]  # (77, 8, 8)

    # Aggregate across all feature planes
    saliency_map = np.sum(saliency, axis=0)  # (8, 8)

    # Normalize between 0 and 1
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-10)

    return saliency_map


# Function to overlay saliency map onto chessboard
def overlay_saliency_on_chessboard(board_fen, saliency_map, nn_move, sf_move):
    """
    Overlay a saliency map onto the chessboard.

    Args:
    - board_fen: Chess position in FEN notation
    - saliency_map: 8x8 saliency heatmap

    Returns:
    - SVG image of the chessboard with saliency overlay
    """
    cmap = plt.get_cmap("Blues")  # Blue heatmap for saliency
    fill_colors = {}

    for i in range(8):
        for j in range(8):
            color = cmap(saliency_map[i, j])  # Get color
            fill_colors[(7 - i) * 8 + j] = '#%02x%02x%02x%02x' % tuple([int(255 * x) for x in color])

    board = chess.Board(board_fen)

    final_position_best = chess.Move.from_uci(nn_move)
    final_position_best_sf = chess.Move.from_uci(sf_move)

    arrows=[
        chess.svg.Arrow(final_position_best.from_square, final_position_best.to_square, color="#ff5555"),  # Red for NN
        chess.svg.Arrow(final_position_best_sf.from_square, final_position_best_sf.to_square, color="#55ff55")   # Green for Stockfish
        ]

    return SVG(chess.svg.board(board=board, fill=fill_colors, arrows=arrows, size=350))

# Example usage
sample_position = df_linchess.sample(1)  # Get random chess position
input_tensor = torch.tensor(sample_position['board'].values[0]).unsqueeze(0).float()
move_index = 0  # Index of move to analyze (can be changed)

# Generate Saliency Map
saliency_map = compute_saliency_map(model, input_tensor, move_index)
example_best_legal = get_best(sample_position['board'].values[0],mask=True)  # With mask to extract real legal move
best_nn = number_to_uci(example_best_legal['outputs'].argmax(dim=1, keepdim=True).item())
sf_move = sample_position['sf_best'].values[0]
print("NN Best Move:", best_nn)
print("Stockfish Best Move:", sf_move)
# Display chessboard with Saliency Map
display(overlay_saliency_on_chessboard(sample_position['fen_original'].values[0], saliency_map,best_nn,sf_move[0]))

print("FEN",sample_position['fen_original'].values[0])
print("Best uci",sample_position['best_uci'].values[0])
print("Best move NN",best_nn)
print("Best move Stockfish",sample_position['sf_best'].values[0])

```

    <ipython-input-20-2eb5425421aa>:45: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
      if input_tensor.grad is None:



    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-20-2eb5425421aa> in <cell line: 0>()
         96 
         97 # Generate Saliency Map
    ---> 98 saliency_map = compute_saliency_map(model, input_tensor, move_index)
         99 example_best_legal = get_best(sample_position['board'].values[0],mask=True)  # With mask to extract real legal move
        100 best_nn = number_to_uci(example_best_legal['outputs'].argmax(dim=1, keepdim=True).item())


    <ipython-input-20-2eb5425421aa> in compute_saliency_map(model, input_tensor, move_index)
         44     # Ensure gradients exist
         45     if input_tensor.grad is None:
    ---> 46         raise RuntimeError("Gradients are not being computed. Check model architecture and requires_grad settings.")
         47 
         48     # Get absolute gradients (importance of each pixel)


    RuntimeError: Gradients are not being computed. Check model architecture and requires_grad settings.



```python
!ls /content
```

    drive  sample_data



```python
!jupyter nbconvert --to markdown "/content/Exploring Chessmaro AI Model.ipynb"

```

    [NbConvertApp] WARNING | pattern '/content/Exploring Chessmaro AI Model.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    

