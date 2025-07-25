from chess_aux import concat_fen_legal as cflp
import sys
import os
import chess
import numpy as np

ruta_modulo = os.path.abspath("c/chessintionlib/")  # Ajusta la ruta según tu estructura
sys.path.append(ruta_modulo)

from chess_aux_c import concat_fen_legal as cflc

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


p = cflp('r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8')

c = cflc('r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8')

for i in range(77):
    print(f"Slice {i+1}", labels[i])
    for row_a, row_b in zip(p[i].astype(int), c[i].astype(int)):  
        print(" ".join(map(str, row_a)), "   |   ", " ".join(map(str, row_b)))  
    print("-" * 40)  # Separator for better visualization



print(np.array_equal(p, c))
print(np.argwhere(p != c))

board = chess.Board('r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8')
print([m.uci() for m in list(board.legal_moves)])

"""
['g8e7', 'g8h6', 'e8f8', 'e8d8', 'e8e7', 'e8d7', 'a8d8', 'a8c8', 'a8b8', 'f6d8', 'f6e7', 'f6h6', 'f6g6', 'f6e6', 'f6g5', 'f6f5', 'f6e5', 'f6h4', 'f6f4', 'f6d4', 'f6c3', 'f6b2', 'c6d8', 'c6b8', 'c6e7', 'c6e5', 'c6a5', 'c6d4', 'c6b4', 'c5b6', 'c5d4', 'c5b4', 'c5e3', 'c5a3', 'c5f2', 'e8c8', 'h7h6', 'g7g6', 'b7b6', 'a7a6', 'd6d5', 'h7h5', 'g7g5', 'b7b5', 'a7a5']
e8d7
e8e7
e8d8
e8f8
e8a8
d6d5
a7a6
b7b6
g7g6
h7h6
a7a5
b7b5
g7g5
h7h5
c6b4
c6d4
c6a5
c6e5
c6e7
c6b8
c6d8
g8h6
g8e7
c5f2
c5a3
c5e3
c5b4
c5d4
c5b6
a8b8
a8c8
a8d8
f6b2
f6c3
f6d4
f6f4
f6h4
f6e5
f6f5
f6g5
f6e6
f6g6
f6h6
f6e7
f6d8"""