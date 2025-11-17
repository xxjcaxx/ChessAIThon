
# Chess and Computers

## Introduction

Chess is one of the oldest and most popular games in the world, known for its deep strategy and endless possibilities. But did you know that chess has also played a huge role in the development of computers and artificial intelligence (AI)? For decades, chess has been a testing ground for scientists and programmers to push the limits of what machines can do. From simple mechanical devices in the 18th century to super-smart AI programs like AlphaZero, the journey of chess and computers is a fascinating story of human creativity and technological progress.

In the 1950s, scientists and mathematicians began to wonder: *Can a computer play chess?* This question led to some of the earliest and most exciting experiments in computer science. Two brilliant minds, **Claude Shannon** and **Alan Turing**, were at the forefront of this exploration. Their work laid the foundation for how computers could think and play games like chess.

Claude Shannon, often called the "father of information theory," was one of the first people to seriously study how a computer could play chess. In 1950, he wrote a groundbreaking paper titled *"Programming a Computer for Playing Chess."* In it, he explained two main ways a computer could approach chess:  

1. **Brute-Force Strategy**: The computer would look at every possible move and outcome, calculating the best move step by step. This method is like exploring every single path in a maze to find the exit.  
2. **Selective Strategy**: Instead of calculating every possible move, the computer would focus only on the most promising moves, using rules and logic to narrow down its choices. This is like taking educated guesses to solve the maze faster.  

Around the same time, **Alan Turing**, the famous mathematician and codebreaker, was also thinking about chess and computers. Turing created one of the first chess algorithms.

The work of Shannon, Turing, and early programmers wasn’t just about teaching computers to play a game. It was about exploring how machines could think, solve problems, and make decisions. Chess became a way to test and improve the capabilities of computers, paving the way for the AI technologies we use today.  

So, the next time you play chess against a computer or even a chess app on your phone, remember: it all started with these pioneers who dared to ask, *"Can a machine play chess?"* And their answer changed the world!

## Chess Representation

When humans play chess, we see the board as a grid of squares with pieces on them. We visualize moves, plan strategies, and use our intuition to decide what to do next. But computers don’t "see" or "think" like humans. They need a way to represent the chessboard and pieces in a format they can understand and process quickly. Over the years, programmers have developed clever ways to store and manipulate chess positions, making it possible for computers to play the game efficiently.

### **1. Bitboard Representation: A Compact and Powerful Tool**  
One of the most efficient ways computers represent a chessboard is using something called a **bitboard**. A bitboard is a 64-bit number (since a chessboard has 64 squares) where each bit (a 0 or 1) represents a square on the board.  

Each type of piece (pawns, knights, bishops, etc.) gets its own bitboard. For example, one bitboard might represent all the white pawns, and another might represent all the black rooks. If a bit is set to 1, it means a piece is on that square; if it’s 0, the square is empty.  

Bitboards are incredibly fast for computers to process. They allow the computer to perform complex calculations (like checking for attacks or generating moves) using simple bitwise operations (like AND, OR, and XOR). This makes them perfect for chess engines that need to analyze millions of positions per second.  

### **2. Board Arrays and Piece-Centric Data Structures**  
Another way computers represent the chessboard is using **arrays**. An array is like a list or grid that stores information about each square on the board.  

- **Board Arrays**: A simple 8x8 array can represent the chessboard, with each cell in the array storing information about the piece on that square (or if it’s empty). For example, the computer might use numbers or letters to represent each piece (like "P" for pawn, "N" for knight, etc.).  
- **Piece-Centric Data Structures**: Instead of focusing on the board, the computer can also store information about each piece individually. For example, it might keep a list of all the pieces and their positions, making it easier to track where everything is. NNUE from Stockfish is an example of it.  

These methods are simpler than bitboards but are still useful for certain tasks, like displaying the board or checking the position of a specific piece.

### **3. Move Generation and Legality Checking**  
Once the computer has a way to represent the board, it needs to figure out what moves are possible and which ones are legal. This is called **move generation** and **legality checking**.  

- **Move Generation**: The computer looks at each piece and calculates where it can move based on the rules of chess. For example, a knight can move in an L-shape, while a rook can move in straight lines. The computer uses its representation of the board (like bitboards or arrays) to quickly find all possible moves.  
- **Legality Checking**: After generating moves, the computer needs to make sure they’re legal. For example, a move isn’t legal if it puts the player’s own king in check. The computer checks this by simulating the move and seeing if the king is safe afterward.  

We will rely for legaly checking in libraries like Python Chess: https://pypi.org/project/chess/ so we don't have to worry about it. 

### **How Humans Represent the Board and Moves**  
Humans don’t use bitboards or arrays, we rely on our brains to visualize the board and plan moves. Here’s how we do it:  

- **Visualization**: When humans look at a chessboard, we see the entire position at once. We can quickly recognize patterns, like a fork (where one piece attacks two at the same time) or a pin (where a piece is stuck because moving it would expose a more valuable piece).  
- **Notation**: Humans use chess notation (like "e4" or "Nf3") to record and describe moves. This helps us communicate and analyze games without needing to see the board.  
- **Intuition and Strategy**: Humans rely on experience and intuition to decide which moves are good or bad. We think about long-term plans, like controlling the center or setting up a checkmate, rather than calculating every possible move.  

While computers are much faster at calculating moves, humans excel at creativity and understanding the "big picture" of the game. This is why human-computer chess matches are so fascinating—they combine the best of both worlds.

When humans play chess, we use visual boards and notation like "e4" or "Nf3" to describe moves. But computers need a more structured way to represent the game so they can process and analyze it efficiently. Two of the most important formats for this are **FEN (Forsyth-Edwards Notation)** and **UCI (Universal Chess Interface)**.  

> These two formats are easy to understand for a human and a machine.

#### **1. FEN: Representing the Chess Position**  
FEN is a compact way to describe the current state of a chessboard. It’s like a snapshot of the game at any moment, including the positions of the pieces, whose turn it is, and other important details. A FEN string consists of six parts, separated by spaces. Here’s an example of a FEN string for the starting position:

```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

Let’s break this down:

1. **Piece Placement**: The first part describes where the pieces are on the board. Each row (rank) is separated by a `/`. Uppercase letters represent white pieces, and lowercase letters represent black pieces. For example:
   - `r` = black rook
   - `n` = black knight
   - `k` = black king
   - `P` = white pawn
   - Numbers (like `8`) represent empty squares.

   So, `rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR` represents the starting position.

2. **Active Color**: The second part is a single letter (`w` or `b`) to show whose turn it is. `w` means White’s turn, and `b` means Black’s turn.

3. **Castling Rights**: The third part shows which castling options are still available. `K` = White can castle kingside, `Q` = White can castle queenside, `k` = Black can castle kingside, `q` = Black can castle queenside. A `-` means no castling is available.

4. **En Passant Target**: The fourth part shows if an en passant capture is possible. If a pawn has just moved two squares, this is the square where it can be captured (e.g., `e3`). Otherwise, it’s `-`.

5. **Halfmove Clock**: The fifth part counts the number of half-moves (plies) since the last pawn move or capture. This is used for the 50-move rule.

6. **Fullmove Number**: The sixth part is the total number of full moves in the game. It starts at 1 and increases after Black’s turn.

---

#### **2. UCI: Representing Moves and Communicating with Engines**  
UCI (Universal Chess Interface) is a protocol that allows chess engines (like Stockfish or Leela Chess Zero) to communicate with graphical interfaces (like chess GUIs or apps). It’s how the engine receives moves and sends back its analysis or best move. Here’s how UCI represents moves and positions:

- **Move Representation**: In UCI, moves are represented using the **algebraic notation** of the starting and ending squares. For example:
  - `e2e4` means a pawn moves from e2 to e4.
  - `g1f3` means a knight moves from g1 to f3.
  - `e7e8q` means a pawn moves from e7 to e8 and promotes to a queen.

- **Position Input**: To tell the engine about the current position, UCI uses the `position` command. For example:
  - `position startpos` tells the engine to set up the starting position.
  - `position startpos moves e2e4 e7e5` sets up the starting position and then plays the moves e2e4 and e7e5.
  - `position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1` sets up a position using a FEN string.

- **Engine Output**: When the engine calculates a move, it responds with the `bestmove` command. For example:
  - `bestmove e2e4` means the engine recommends moving a pawn from e2 to e4.

## Chess and AI

The journey of chess AI has been a fascinating evolution from rule-based systems to advanced machine learning approaches. In the early days, chess engines relied on **rule-based systems**, where programmers manually coded the rules of chess and strategies into the computer. These engines, like IBM’s Deep Blue, used brute-force calculations to evaluate millions of positions per second, combined with human-crafted evaluation functions to decide the best moves. While effective, these systems were limited by their reliance on human knowledge and the sheer computational power required to analyze complex positions.

The introduction of **neural networks** marked a significant shift in how computers approached chess. Unlike rule-based systems, neural networks learn patterns and strategies from data, mimicking the way the human brain works. Instead of being explicitly programmed with chess knowledge, these systems are trained on millions of chess games, allowing them to develop their own understanding of the game. This approach led to the creation of engines like **AlphaZero** and **Leela Chess Zero**, which use self-learning techniques to master chess without any human guidance. These engines don’t just calculate moves—they *intuit* the best strategies, often discovering creative and unconventional ideas that even grandmasters find surprising.

One of the most iconic moments in the history of chess AI was the **Deep Blue vs. Garry Kasparov match in 1997**. Deep Blue, a rule-based supercomputer developed by IBM, defeated the reigning world champion, Garry Kasparov, in a six-game match. This victory was a milestone in AI history, proving that machines could outperform even the best human players in a game long considered the pinnacle of human intellect. However, Deep Blue’s success was largely due to its immense computational power and human-programmed evaluation functions, rather than any form of learning or intuition.

The rise of **Monte Carlo Tree Search (MCTS)** further revolutionized chess engines. MCTS is a decision-making algorithm that simulates thousands of random games from a given position to estimate the best move. Unlike brute-force methods, MCTS focuses on the most promising lines of play, making it more efficient and effective. When combined with neural networks, as seen in AlphaZero, MCTS allows engines to explore positions deeply while leveraging learned patterns and strategies. This hybrid approach has become a cornerstone of modern chess AI.

Monte Carlo Tree Search (MCTS) is more efficient than brute-force search, but its performance can be greatly enhanced when combined with neural networks. While a neural network cannot simulate all future moves, it has an intuitive ability to identify the most promising candidate moves at the outset. By guiding MCTS toward these moves, the search algorithm explores high-potential branches more effectively, rather than relying on random simulations. This synergy allows the engine to focus computational resources on the most relevant lines of play, leading to deeper and more accurate analysis.

In the era of **modern chess AI**, engines like AlphaZero and Leela Chess Zero have taken center stage. These self-learning systems use **reinforcement learning**, a technique where the engine plays millions of games against itself, improving over time by learning from its mistakes and successes. Unlike classical engines, which rely on pre-programmed evaluation functions, neural network-based engines develop their own understanding of chess, often uncovering new strategies and ideas that challenge traditional human understanding of the game.

Stockfish has implemented NNUE in order to generate more "Human" moves: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html#a-simple-nnue-network . NNUE is what allowed Stockfish to combine classical brute-force search with neural evaluation—and that’s why Stockfish remains at the top of computer chess today. The architecture is very different to Alphazero neural network because it doesn't store board matrix, it instead represent piece–square pairs (e.g., “white knight on f3”). 

Stockfish’s strategy represents a hybrid milestone in the history of chess AI: unlike early engines such as Deep Blue that relied purely on brute-force search and handcrafted evaluation, and unlike modern neural systems such as AlphaZero that use deep reinforcement learning and stochastic search, Stockfish combines its powerful deterministic **alpha-beta** search with **NNUE (Efficiently Updatable Neural Networks)**, a lightweight neural evaluator trained on massive datasets. This approach keeps the speed and depth of classical engines while integrating the pattern-recognition strengths of neural networks, showing how the evolution of chess AI has moved from pure calculation, through stochastic learning, toward a synthesis of both traditions.



### Training Chess AI

In summary, there are several approaches to modeling and training a chess AI. For our project, we will not adopt brute-force search or hard-coded heuristic techniques to determine the best move. Instead, our approach is more **stochastic** and less **deterministic**. 

By integrating **Monte Carlo Tree Search (MCTS)** with a neural network, we can introduce controlled randomness or “personality” into the AI’s playstyle. 

> We acknowledge that, given our limited computational resources, this experiment will not achieve performance comparable to traditional engines, let alone systems such as AlphaZero. However, it will serve as a valuable learning experience. This approach directly aligns with the **stochastic nature** of our project, where the integration of the **Chessmaro CNN** with **Monte Carlo Tree Search (MCTS)** is not just a technical choice, but a **pedagogical and philosophical one**.

The MCTS-enhanced CNN allows us to move beyond a pure evaluation function and incorporate the essential human elements of chess captured in the earlier points: **complexity, difficulty, and player personality.**

The inherent **randomness** introduced by MCTS—specifically in its exploration phase and its reliance on the neural network's **probability distribution** over moves (rather than a single deterministic evaluation)—serves as the technical analog for **human personality**.

* **Controlled "Error":** The model won't always choose the move with the highest engine evaluation, but rather a move with a **high probability** of being played by a human with similar training data. This controlled divergence from the "perfect" move is key to modeling the fact that **chess players are likely to make errors in difficult positions, unlike engines**.
* **Modeling Difficulty:** By using the CNN to guide the MCTS, positions that are intrinsically **difficult for human players** will likely result in a flatter, less decisive probability distribution from the Chessmaro model. This uncertainty in the model directly corresponds to the **complexity metric** we want to develop.

Our choice of a data-driven model, trained on student-generated moves, directly enables the application of player profiling and customized opening theory:

* **Complexity Tendencies** By analyzing which nodes the **MCTS explores** most frequently and the **depth/breadth** of the search required before a decision is made, we can generate real-time metrics that reflect the position's perceived difficulty for the AI. This data is essential for devising **opening systems** around an opponent’s **tendency to seek or avoid complexity**.
* **Targeted Training and Diagnostics:** The human-centric nature of the **Chessmaro CNN** allows us to focus our **VET training dataset** on specific types of positions. We can:
    * Feed the model with data filtered by **player rating and time control**.
    * Retrain or fine-tune models on positions that are empirically **difficult for low-rated players and easy for high-rated players**. The resulting differences in the move probabilities between the two models would form the basis of a **diagnostic chess exam** to identify a player's knowledge gaps.

Integrating the MCTS/CNN output provides a more robust, **data-driven method** for rating and generating challenges:

* **Advanced Puzzle Generation:** We can use the MCTS to find moves that are not tactically decisive but represent a **positional challenge**. A puzzle solution could be defined not by a vast evaluation difference, but by a move that significantly **increases the MCTS win rate** while having a relatively small difference in the **CNN's predicted move probability**. This creates the desired **non-tactical puzzles**.
* **Faster Rating Establishment:** By using the **confidence/uncertainty** of the Chessmaro model’s output for a given position, we can assign an **initial complexity/difficulty score** with far greater speed and accuracy than the current "elo"-based trial-and-error method, thus requiring **fewer attempts to establish an initial puzzle rating**.

In essence, the MCTS/CNN framework is the **technical vehicle** for transforming theoretical ideas about **human cognitive difficulty** into practical, measurable, and educational tools.