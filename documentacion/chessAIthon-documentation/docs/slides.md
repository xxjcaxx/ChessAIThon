# ChessAIThon

* How computers represents Chess scenarios and moves?
* How can I store scenarios and moves?
* How Chess AI works?
* How to train an AI for Chess?
* How to use this AI to get the best move?
* How can I use my inteligence to improve AI?

# How computers represents Chess scenarios and moves?

## Computers
* Bitboard
* Board Arrays and Piece-Centric Data Structures

## Humans
* Visual
* Notation (FEN, PGN) (UCI, SAN) 

Human can comunicate with computers in FEN and UCI

# How can I store scenarios and moves?

* PGN and FEN 
* CSV
* JSON
* Parquet

## Version Control

* Git

# How Chess AI works?

* rule-based systems
* Alpha-beta search
* MCTS
* Neural Networks
* Neural Networks + MCTS


# How to train an AI for Chess?

* Our choice: CNN 
* Our choice: based in a Dataset
* Self-reinforcement: Alphazero / Leela Chess Zero
* Prediction: Policies + Value
    * Policies: Sorted best moves and priority
    * Value: Hope that actual situation will lead to win the game (non trivial)

# CNN

* Bitboard representation: 77x8x8
* 77 Layers input
* 256 Layers in some residual networks
* Lineal outputs for policies and value

# Dataset

* From Leela Chess Zero simulations (400 nodes)
* Need to be transformed to 77x8x8 in Parquet format
* ~1500000 scenarios in fen, best move, and value

# Results

* ~28% of accuracy
* ~70% of accuracy in top best moves
* Sufficient for MCTS in theory. 

# How to use this AI to get the best move?

* CNN best move sometimes is wrong 
* It doesn't have the simulation experience
* MCTS will choose best policies and explore them thanks to values.
* Sometimes MCTS chooses a better move
* Sometimes is more conservative. 
* We need some random:
    * Dirichlet Noise
    * Softmax decisions


# How can I use my inteligence to improve AI?

* ChessAIThon provides a web platform to play and store best moves
* You can store best move and value in your opinion
* ChessAIThon provides a Notebook to transform your dataset to 77x8x8 parquet format
* You can use this dataset to fine-tuning out CNN with another Notebook
* You can change the weiths of the CNN in deploy and play with it.