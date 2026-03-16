---
marp: true
theme: default
style: '@import "slides.css";'
paginate: true

---

<!-- _class: lead -->

# ChessAIThon Slides
### Strategic Gaming, Coding, and AI in VET Education

![bg left:40% width:450px](Exploring%20Chessmaro%20AI%20Model_files/logoblanc.png)



*   How computers represent Chess scenarios and moves?
*   How can I store scenarios and moves?
*   How Chess AI works?
*   How to train an AI for Chess?
*   How to use this AI to get the best move?
*   How can I use my intelligence to improve AI?

![pos-bottom-left  width:200px](Exploring%20Chessmaro%20AI%20Model_files/erasmus_plus_ok.jpg)


---

# How computers represents Chess scenarios and moves?

![bg right:40% width:450px](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_6_2.png)

-   **Bitboards**: 64-bit integers representing squares. Highly efficient for bitwise operations.
-   **Board Arrays**: Simple 8x8 grids for piece tracking.
-   **Human Formats**: **FEN** (snapshot) and **UCI/SAN** (move notation).
-   **AI Input (77x8x8)**: 
    - 12 layers for piece positions.
    - 1 layer for turn indicator.
    - 64 layers for **legal move masks**.

---

# How can I store scenarios and moves?

-   **Standard Formats**: PGN (Games), FEN (Positions).
    rnb1k1nr/pppp1ppp/5q2/2bP4/8/5N2/PPP1PPPP/RNBQKB1R w KQkq - 3 6
-   **Data Science Stack**: CSV for simplicity, JSON for web APIs.
-   **Parquet**: Columnar storage with **int64 compression**, reducing 10GB datasets to 1.5GB.
-   **Version Control**: Using **Git/GitHub** to manage dataset iterations and collaborative student contributions.


---

# How Chess AI works?

![bg left:40% width:500px](Exploring%20Chessmaro%20AI%20Model_files/Minimax-alpha-bera-pruning-cnn.png)
-   **Rule-based Systems**: Brute-force calculation (Deep Blue).
-   **Alpha-beta Search**: Heuristic evaluation (Stockfish).
-   **Modern Neural Networks**: Intuition-based move prediction (AlphaZero).
-   **Hybrid Model (ChessAIThon)**: 
    - **NN**: Provides "intuition" (Policy + Value).
    - **MCTS**: Provides "calculation" (Look-ahead search).

---

# How to train an AI for Chess?

![bg right:45% width:500px](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_1.svg)

-   **Our Choice**: Supervised learning from a curated dataset of human and simulated games.
-   **Learning from Data**: Comparing NN predictions with the "best move" targets.
-   **Human-Centric**: Trained on student moves to model human-like playstyles and complexity.

---

# Deep Learning

![bg left:45% width:500px](Exploring%20Chessmaro%20AI%20Model_files/F1-03.large.jpg)

-   Each problem is a mathematical function with many parameters. We try parameters and check if returns the appropiate Y for the X
-   **Loss and Gradient Descent** We try to minimize Loss and we try better parameters in the way to descent.
-   **Learning Rate**: If we learn too fast we can miss the minimum. 
-   **Architecture**: We choose Convolutional Networks because chess has the same architecture as an image: 8x8 pixels of some color depth.

---

# CNN

![bg left:45% width:500px](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_21_6.png)

-   **Bitboard representation**: 77x8x8
-   **Output**:
    - **Policies**: Probability distribution over 4096 possible moves.
    - **Value**: Predicting the win/loss outcome ([-1, 1]).
`
---

# ChessNet
`

- **CNN layers**`
- **Full Connected layers**
- **Residual Tower**
![width:100%](Exploring%20Chessmaro%20AI%20Model_files/chessnet.png)



---

# Dataset

![bg right:45% width:500px](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_16_0.png)

-   **Source**: Leela Chess Zero simulations (400 nodes) and Lichess puzzles.
-   **Volume**: ~4,500,000 scenarios in Parquet format.
-   **Diversity**: Includes balanced positions, tactical puzzles, and mate-specific scenarios.
-   **Refinement**: Filtering out "decided" positions to prioritize informative learning signals (0.1 < |Value| < 0.7).

---

# Results

![bg left:45% width:550px](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_10_1.png)

-   **Top-1 Accuracy**: **~30%** (Matching Stockfish's perfect move).
-   **Top-5 Accuracy**: **~70%** (Sufficient to prune MCTS search space).
-   **Human-like Behavior**: Matches human "mistakes" in complex positions, making it an ideal pedagogical tool.
-   **Success**: Even modest accuracy leads to strong strategic play when paired with MCTS.

---

# How to get the best move?

![bg right:45% width:500px](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_19_11.svg)

-   **MCTS Guided Search**: Simulation explores branches prioritized by the CNN Policy.
-   **Exploration (PUCT)**: Balancing known good moves with less-explored alternatives.
-   **Noise**: Dirichlet noise at the root node ensures variety and prevents deterministic traps.
-   **Mixture of experts**: Different MCTS in parallel vote the best move. The fist trust in the CNN, the last more in exploration.

---

# How can I use my inteligence to improve AI?

![bg left:45% width:500px](Exploring%20Chessmaro%20AI%20Model_files/Exploring%20Chessmaro%20AI%20Model_22_3.png)

-   **ChessMinds Web App**: Play against ChessMarro and contribute your best moves to the dataset.
-   **Fine-tuning**: Use provided Notebooks to retrain the CNN on your personal games.
-   **Community**: Modify architecture parameters or weights and share your version on HuggingFace.

---

<!-- _class: lead -->

# Thank You
### Questions & Collaborative AI Development
**GitHub**: xxjcaxx/ChessAIThon

