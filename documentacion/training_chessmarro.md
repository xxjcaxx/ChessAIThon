# Introduction

Chess is a great example for learning AI model creation because it has well-defined rules, a finite and discrete state space, and a deterministic nature, making it easier to model computationally. It has been a historical benchmark for AI, from Deep Blue to AlphaZero, offering rich literature and frameworks. The game’s complexity scales from simple moves to deep strategic planning, providing a natural learning curve. Additionally, vast databases of chess games make training AI models more accessible.

We are developing an AI model to determine the best next move in a given chess position, focusing on deep learning methods that have proven competitive against traditional stochastic algorithms. In concrete we are going to make a generic CNN for chess that:

* Will be able to play with white or black moves.
* Will be able to play at any point of the game. But:
    * There will be more similar starts (logicaly) and we will have special care in mates, so we will put part of our mates database.
* Will predict a legal move before testing
* Will improve as we can the CNN and hyperparameters.

## Lectures

* https://www.freecodecamp.org/news/create-a-self-playing-ai-chess-engine-from-scratch/
* http://cs230.stanford.edu/projects_winter_2019/reports/15808948.pdf
* https://ai.stackexchange.com/questions/27336/how-does-the-alpha-zeros-move-encoding-work
* http://www.diva-portal.se/smash/get/diva2:1366229/FULLTEXT01.pdf
* https://github.com/asdfjkl/neural_network_chess/releases

## What is a CNN?

A **CNN (Convolutional Neural Network)** is a type of computer program that helps machines recognize patterns in pictures and videos. It's like a super-smart eye for computers!

Imagine you are looking at a picture of a dog. Your brain automatically sees the fur, ears, and eyes and knows it's a dog. A CNN does something similar but in a digital way. It scans the picture in small parts, finds important details (like shapes, edges, and colors), and then puts everything together to figure out what’s in the image.

CNNs are used in things like facial recognition (unlocking phones with your face), self-driving cars (detecting people and objects), and even medical scans (helping doctors find diseases). So, they help computers "see" and understand the world just like humans do.

### Why we will use CNN for Chess moves prediction?

Using a **Convolutional Neural Network (CNN)** for **chess move prediction** makes sense because a chessboard is like an image—a **grid-based structure** where pieces have specific positions, similar to pixels in an image. Here’s why CNNs work well for this task:

1. **Chessboard is a Spatial Grid**
   - A chessboard is an **8x8 grid**, just like an image has pixels arranged in rows and columns.
   - CNNs are great at detecting patterns in spatial data, so they can analyze piece positions and relationships effectively.

2. **Pattern Recognition in Chess**
   - Chess strategies involve recognizing common patterns, like **forks, pins, checkmate threats**, and openings.
   - CNNs learn from thousands of games, identifying these patterns and predicting the best moves.

3. **Local Feature Extraction**
   - In an image, CNNs detect features like **edges and textures**.
   - In chess, CNNs detect **piece clusters, threats, and control over squares**, which helps in move prediction.

4. **Efficient Processing**
   - Instead of treating the board as just numbers, CNNs process it like an image, **reducing complexity** and making training faster.
   - They can focus on **important regions** (like areas around the king in check situations) rather than considering all moves equally.

5. **Used in AI Chess Engines**
   - AI systems like **AlphaZero** (by DeepMind) use deep learning (including CNNs) to predict chess moves without human-made rules, learning purely from playing games.
   - CNNs help AI **evaluate positions** and **choose the best move** just like grandmasters do.


We are going to create a **CNN (Convolutional Neural Network)** to help predict chess moves, similar to how **AlphaZero** and **Leela Chess Zero** work.

The main difference is that these powerful AIs learn by **playing against themselves** over and over, improving with each game. However, in our case, we will **train our CNN using games played by you** as examples. This will help the AI learn from real human moves!

But this won’t be easy!

We need to figure out how to **represent the chessboard** and the **best predicted move** so the AI can understand them. Then, we’ll need to **create a dataset** with different chess positions to train our CNN.

On top of that, we must make sure the moves the AI suggests are **valid**, meaning they follow the rules of chess, and also check if they are actually **good moves**.

# Preparing the data.

We can represent a chess game in varios formats: FEN, FEN moves, SAN moves, UCI moves... and with matrix. Each piece has a letter and we can make a matrix like this:

       +------------------------+
     8 | r  n  b  q  k  b  n  r |
     7 | p  p  p  p  .  p  p  p |
     6 | .  .  .  .  .  .  .  . |
     5 | .  .  .  .  p  .  .  . |
     4 | .  .  .  .  P  P  .  . |
     3 | .  .  .  .  .  .  .  . |
     2 | P  P  P  P  .  .  P  P |
     1 | R  N  B  Q  K  B  N  R |
       +------------------------+
         a  b  c  d  e  f  g  h'


To train a **Convolutional Neural Network (CNN)** to predict chess moves, we need to represent the chessboard in a way the AI can understand.

At first, we might think of using a **single matrix (grid of numbers)** like this:

```
[
  [ 4,  2,  3,  5,  6,  3,  2,  4],
  [ 1,  1,  1,  1,  0,  1,  1,  1],
  [ 0,  0,  0,  0,  0,  0,  0,  0],
  [ 0,  0,  0,  0, -1,  0,  0,  0],
  [ 0,  0,  0,  0, -1, -1,  0,  0],
  [ 0,  0,  0,  0,  0,  0,  0,  0],
  [-1, -1, -1, -1,  0, -1, -1, -1],
  [-4, -2, -3, -5, -6, -3, -2, -4]
]
```

Here:
- **Positive numbers** represent white pieces.
- **Negative numbers** represent black pieces.
- Different numbers represent different pieces (pawns, knights, bishops, etc.).

**Why is This a Problem?**

The CNN might **misinterpret** the numbers because it learns based on patterns and weights. For example:
- The number **6 (king)** is much bigger than **1 (pawn)**, but in chess, every piece is important!
- A CNN might think higher numbers mean **more important pieces**, which is **not true** in all situations.

**A Better Solution: One Matrix for Each Piece Type**

Instead of using a single matrix, we use **separate matrices** for each type of piece. For example:

- One matrix for **pawns**
- One matrix for **knights**
- One matrix for **bishops**
- One matrix for **rooks**
- One matrix for **queens**
- One matrix for **kings**

Each matrix is **filled with 1s and 0s**. This way, all pieces are treated **equally**, and the CNN can better understand their positions.
- The AI **sees chess like an image**, where each type of piece has its own layer, just like different colors in a picture.
- It prevents the AI from making mistakes based on number size.
- It improves **move predictions** by focusing on piece **positions**, not their values.

This is the same approach used by **AlphaZero** and other strong chess AIs.

       [[[0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 1., 0.],
         [1., 0., 0., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 1., 0., 0., 1.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.]],
         ...

This format is inspired in Alphazero way to code data. Alphazero does it more complex, but it has information about previous moves. But in theory, getting the best move is not related to historic data. Leela zero uses similar aproach.


## Preparing Moves

AlphaZero doesn’t just look at the chessboard; it also **tracks all possible moves** for each piece. It does this using a **72-channel matrix**, meaning there are **72 separate layers** of information about potential moves.

We can use a **similar approach** in two important ways:

**1. Representing the Best Move (Target)**
- Instead of just predicting a move as "piece A moves to square B," we can use a **matrix representation** to show the best move in a way the AI understands.
- This helps the AI **learn patterns** of strong moves instead of just memorizing past games.

**2. Representing All Legal Moves**
- We can **expand this idea** by also representing **all possible legal moves** for every piece in a game position.
- This means the AI will not only know where pieces are but also where they **can go**, making it smarter in predicting the best move.


## Format of the data

When training a chess AI, we need to store **a lot of chess games** in a way that the computer can understand. However, if we don’t use a smart format, the data can become **too large** and slow down the training process.
The data is saved in **JSON format**, but instead of using a simple 8x8 matrix for the chessboard, we use a special method to **save space**:

1. **Each chess situation is stored in a 77x8x8 matrix**
   - This means there are **77 layers of 8x8 grids**, each storing different details about the board.
   - Instead of saving regular numbers, we use a compressed format where each number is a **64-bit integer (int64)**.

2. **Why Use int64 Instead of a Regular 8x8 Grid?**
   - A normal 8x8 matrix in **CSV or JSON** takes **at least 8 bits per number**, plus extra characters like commas, brackets, etc.
   - Using **int64 (64-bit integer)** allows us to store an entire **8x8 matrix using just 1 bit per number**—saving a lot of space!
- If the dataset has **100,000 complete chess games**, and each game has **many moves** (each move is a new board position).
- In **FEN format** (a way to describe chess positions), the dataset is **about 500MB**.
- In the **77x int64 format**, it could be **1.5GB**.
- If we stored it in a **regular 77x8x8 format**, it could be **more than 10GB**, which is too large!

**How Do We Prepare the Data for AI Training?**
Before training, we need to **convert** this data into a format the AI can use:
✅ **Convert the 77x int64 data into a 76x8x8 NumPy matrix** (NumPy is a fast way to handle numbers in Python).
✅ **Reduce the dataset size** so we can train with **more chess games** without using too much storage.


This 77 array of matrix represents 76 boards with different type of information each one:
* The **12 first** represents the position of the 6 types of pieces x2 colors:

```python
    [
     white pawns, black pawns, white knights, black knights, white bishops, black bishops, white rooks, black rooks, white queen, black queen, white king, black king
    ]
```

One 1 in a place means this type of piece is on it.

* Then, we have a entire binary matrix that informs about turn. It means a 8x8 1s matrix is Black turn and 8x8 0s matrix is white turn
```
* The other 64 matrix are for all the possible moves of a piece. Since a queen can do every move except the knights moves, we take as a reference the queen. A queen can potentialy move in 8 directions and can move until 7 positions. This possible moves are the first 56 matrix. The other 8 are the 8 possible moves of a knights

*
```python
    [
    ...North moves x 7, ...NE x 7, ...E x7, ...SE x7, ...S x7, ...SW x7, ...W x7, ...NW x7  (Queen moves)
    ...knights x 8
    ]
```
* The other 64 matrix are for all the possible moves of a piece.

In a chess game, different pieces have different movement patterns. For example, a **Queen** can move in multiple directions, while a **Knight** has a very unique movement style. We can use a matrix to represent all these possible moves efficiently.

The Queen is one of the most powerful pieces in chess. It can move:
- **Vertically (up and down the board)**
- **Horizontally (left and right)**
- **Diagonally in all four directions (NE, SE, SW, NW)**

The Queen can potentially move up to **7 squares** in any of these 8 directions (since the board is 8x8). So, in total, there are **56 possible moves** for the Queen:
- **7 moves to the North (Up)**
- **7 moves to the North-East (NE)**
- **7 moves to the East (Right)**
- **7 moves to the South-East (SE)**
- **7 moves to the South (Down)**
- **7 moves to the South-West (SW)**
- **7 moves to the West (Left)**
- **7 moves to the North-West (NW)**

Each of these 8 directions is represented as a matrix of size 8x8, and the **first 56 matrices** (out of 64) will be used to describe all possible **Queen's moves** in these directions.

The **Knight** has a very unique movement: it moves in an "L" shape, either:
- Two squares in one direction (up/down/left/right), then one square perpendicular to that direction (left/right/up/down).
- Or, one square in one direction (up/down/left/right), then two squares perpendicular to that direction.

There are exactly **8 possible moves** for a knight from any given position. These are:
- Two squares up and one square left.
- Two squares up and one square right.
- Two squares down and one square left.
- Two squares down and one square right.
- One square up and two squares left.
- One square up and two squares right.
- One square down and two squares left.
- One square down and two squares right.

The **remaining 8 matrices** in the representation are used to describe these 8 possible moves for the **Knight**.

So, if we organize the data:
- The first **56 matrices** represent the **Queen's possible moves** in the 8 directions (North, NE, East, SE, South, SW, West, NW).
- The last **8 matrices** represent the **Knight's 8 possible moves**.

Each matrix is an **8x8 grid** where:
- **1** represents a potential move to that square (a valid move for the piece).
- **0** represents a square where the piece cannot move (based on its movement type).

We can visualize the structure of the full 64 matrices like this:

```python
[
  ... 7 North moves x 8,  # First 7 matrices for Queen's North moves
  ... 7 NE moves x 8,     # Next 7 matrices for Queen's NE moves
  ... 7 East moves x 8,   # Next 7 matrices for Queen's East moves
  ... 7 SE moves x 8,     # Next 7 matrices for Queen's SE moves
  ... 7 South moves x 8,  # Next 7 matrices for Queen's South moves
  ... 7 SW moves x 8,     # Next 7 matrices for Queen's SW moves
  ... 7 West moves x 8,   # Next 7 matrices for Queen's West moves
  ... 7 NW moves x 8,     # Next 7 matrices for Queen's NW moves

  ... 8 Knight's possible moves (8 matrices)
]
```

By using this matrix structure:
1. The AI can **easily learn** the movement patterns of each piece.
2. It can identify which positions are valid moves for a **Queen** or a **Knight** at any given point on the board.
3. The matrix format is easy to process and allows the AI to quickly determine all possible moves for any piece at any time.


**Thinking of the Matrix Like a Pixel in an Image**

Imagine you have an image on a screen. This image is made up of tiny **dots** called **pixels**. Each pixel has a color, and when you combine all these pixels, they form a complete picture. In the same way, we can think of the matrix for chess moves as a **grid of tiny "pixels"**.

In our case, the grid is an **8x8 matrix** (like a chessboard). Each **square** on this chessboard can either be a **1** (a move is possible) or a **0** (no move possible). So, this matrix is like an image where each square can be **on** (1) or **off** (0).

Now, think about how we use this grid to represent a piece's movement. Each piece (like a Queen or Knight) has its own **pattern of possible moves**.

- If a piece can move to a certain square, we mark it with a **1**.
- If it can't move there, we mark it with a **0**.

For example:
- The **Queen** can move in many directions (up, down, diagonally), so in the **matrix**, many squares will have **1s** where it can go, and **0s** everywhere else.
- The **Knight** moves in an "L" shape, so only a few squares will have **1s** for its possible moves.

The idea of a **"hot" position** means that the piece is very active or has a lot of potential moves. So, in the matrix:
- The position where the piece is located (like the Queen’s starting position) will have **more 1s** around it, because the piece can move to more squares.
- The AI will notice that these positions are important because the piece has **more options**, and that makes the position **"hotter"**.

For example:
- If the **Queen** is at the center of the board, the matrix around it will be **"hotter"** because the Queen can move in many directions. The AI will recognize that this position is important.
- A piece with fewer moves will have fewer **1s** around it, making its position **"colder"** in the matrix.

Now that the AI sees these patterns, it can:
- **Understand the game**: The AI can recognize where each piece is and how many squares it can move to. The more **1s** in the matrix, the more powerful the piece is in that position.
- **Make better decisions**: When the AI is trying to choose the best move, it looks for the **"hottest"** positions (where there are lots of **1s**) because those are the most valuable moves. It knows that those positions give it more **options** for winning the game.

It’s like the AI is looking at a **map of possible moves** for each piece, and it can use this to play smarter chess.

**Example**

So if we have a White Queen like this:




The queen can do lots of moves to: c7, c8, d7, e8, d6, e6, f6, d5, e4, f3, g2, c5, b5, a4, b6, b7, a8. We can take all this moves and interpret that, for example, the c6c7 move is 1 position to North, so the 12(board)+0*7(direction displacement)+1(squares quantity) = 13 matrix will have a 1 in the c6 current position of the queen.

If we take the c6g2 move of the queen, the 12 + 3*7 (SE) + 4 (square qty) = 37. This matrix will have a 1 in the c6 ([2][2]) that is the current position of the queen.

In summary, the 64 channels capture the possible moves for each of the 64 positions on the chessboard, covering various directions and types of moves for each piece. This representation is designed to provide comprehensive information about the legal moves associated with each piece and its position on the chessboard.

> There is a fundamental difference with Alphazero Alphazero stores last k moves in k*14 layers more. We are going to use 12 + 1 + 64 matrix. We will avoid historical information to make a more "simple" game where we don't know repetitions, castling or pawn promotes. It will save us many layers, disk space, CNN layers, RAM and GPU time.

> Our strategy is to feed the CNN with game status and possible moves. These 2 different data can reinforce learning during CNN phase. We can, also, use the 64x8x8 possible moves matrix as a "Boolean Mask" to avoid illegal moves during forward phase and to train only with legal and the best move. If we avoid to enter this legal moves in the CNN, we can spare 64 layers and have only a 13 layer input.


### Why this structure?

Representing a chessboard as 12 matrices, each capturing the positions of different types and colors of pieces, is advantageous for a Convolutional Neural Network (CNN) due to the nature of the information being processed. Here are some reasons why this representation might be preferred over a single matrix of numerical values for a CNN:

1. **Hierarchical Features:** The individual matrices for each piece type allow the CNN to learn hierarchical features. Each matrix can represent a specific piece (e.g., pawn, knight) and its position, enabling the network to focus on learning distinctive features for each type separately.

2. **Spatial Information:** Having separate matrices for different pieces helps preserve the spatial information of the board. This is crucial for chess, where the positions and movements of pieces relative to each other are essential for understanding the game state.

3. **Learning Discriminative Features:** CNNs are effective at learning hierarchical and discriminative features from images. By representing the chessboard as separate matrices, the network can learn to recognize unique patterns associated with each piece, contributing to better feature extraction.

4. **Flexibility:** The modular representation of the board provides flexibility in adapting to changes in the game state. Adding or removing a piece corresponds to modifying a specific matrix, making it easier for the network to adapt to variations in the board configuration.

5. **Interpretability:** The 12 matrices provide a more interpretable representation. Each matrix corresponds to a specific aspect of the game, making it easier to understand what features the network is focusing on during the learning process.


Using 64 matrices of size 8x8 to represent all possible moves for any piece on a given chessboard can be useful for training a neural network due to its ability to capture spatial relationships and patterns. Each 8x8 matrix corresponds to a potential move, and the presence of a '1' at a specific position in a matrix signifies the location of the piece that can make that particular move.

Here are some key points explaining the utility of this representation, all the previous arguments plus **Alignment with Board Representation:** Since chess boards are inherently represented as 8x8 grids, using matrices of the same size naturally aligns with the board's structure. This makes it easier for the network to relate its learned features to the actual layout of the chessboard.


## Best move representation

Lets talk about best move representation:

Best move is represented in a 0 to 4096 format to simplify. And it 's calculated with

    np.ravel_multi_index(
            multi_index=((from_rank, from_file, move_type)),
            dims=(64,8,8)
        )


In chess, each move a piece makes can be represented in a special way. Instead of writing out the move in a regular format (like **e2 to e4**), we use a system that turns the move into a number between **0 and 4096**. This number helps the computer understand all possible moves in a simplified way.

To create this number, we use a function called **`np.ravel_multi_index`**, which combines three pieces of information:
1. **`from_rank`**: The row where the piece starts.
2. **`from_file`**: The column where the piece starts.
3. **`move_type`**: The type of move being made (like a regular move, a capture, or castling).

This function takes these three pieces of information and turns them into a number between 0 and 4095, which corresponds to one of the 4096 possible legal moves in chess. This way, the computer can easily track all possible moves.

We choose this simple way to represent moves to make easy to the CNN to return a prediction because prediction is an array of 4096 numbers that represent the provability of each one to be the best move.

Now, when we give this number to the neural network (AI), it understands it by looking at a **matrix** (a big grid) of **legal moves** for every piece on the board. The AI knows that there will always be one **active** (valid) move to choose, and it can pick the right one based on the piece's position and the available moves.

Additionally, we need to **convert between different formats** of moves. For example, the moves might be in a **UCI format** (like **e2e4**) and we need to turn it into this special number format, or the AI might need to convert the number back into a regular move. This is called **coding and decoding** the move formats.


# Implement in a Notebook

LINK TO COLAB

As all AI projects in Python, we need to import lots of libraries. These are the most common: numpy for numerical jobs, pandas to data management and parquet for data loading and storage.
We need to install python-chess and ipython to code chess scenarios and to improve feeback in this notebook.



To develop a chess-playing AI, we can explore several strategies:

1. **Neural Network Learning from Historical Data:**
   - Utilize a neural network trained on a database of chess moves to predict optimal plays based on a given board position. The network takes the current board state as input and selects a move. This move is then compared with the best move from historical examples, and the accuracy is calculated to adjust the neural network for the next example. To enhance this approach, include legal moves as part of the input and properly encode the output in a format compatible with these legal moves. Finally, filter the network's selected moves using a mask to retain only legal ones and choose the best among them.

2. **Competing Against Another Chess AI (e.g., Stockfish):**
   - Engage the neural network in competition against another AI, such as Stockfish. Each decision made by the neural network is scored based on its success. Subsequently, retrain the network using these scores and pit it against the opponent again. Success can be measured by either winning the game or evaluating its moves using a scoring algorithm like Stockfish's.

3. **Self-Competition Learning (Inspired by AlphaZero):**
   - Allow the neural network to compete against itself, with the better versions consistently outperforming the weaker ones. This approach, akin to AlphaZero, is more complex and resource-intensive but offers adaptability to various games and isn't constrained by the opinions of other AIs.


### **Summary of Our Chess AI Discussion**

1️⃣ **Chess AI Model Structure**
   - Using a **CNN (Convolutional Neural Network)** to predict the best move from a chess position.
   - Input is a **77-layer 8×8 matrix** representing the board.
   - The model has **4 convolutional layers, batch normalization, and fully connected layers** leading to **4096 possible move outputs**.

2️⃣ **Move Representation**
   - Moves are encoded in a **0-4096 format** using `np.ravel_multi_index()`.
   - This format aligns with the **8×8×64 legal move matrix**, making it easier for AI to learn.
   - We need to **convert UCI chess notation to this format and back**.

3️⃣ **Monte Carlo Tree Search (MCTS)**
   - A **CNN alone is not enough** to find the best move because it can’t look ahead.
   - **MCTS helps by simulating future moves**, with AI guiding which moves to explore.
   - MCTS can be implemented **either in Python (server-side) or JavaScript (client-side)**.

4️⃣ **Deploying the Model as an API**
   - **Options:** Streamlit or Gradio for UI, Hugging Face Spaces for hosting.
   - **API:** If using Gradio, you need to expose an endpoint and call it from a webpage.
   - Need to **properly save and load the model (`.pth` file)** to avoid issues.

5️⃣ **Saving and Loading the Model**
   - **Only saving `state_dict()` is recommended** for portability.
   - `torch.save(model.state_dict(), path)` and then reload with `model.load_state_dict(torch.load(path))`.
   - If saving the full model, define the class globally to avoid `pickle` errors.

6️⃣ **Training Strategy**
   - **Dataset:** Started with **200k chess positions**, but **1M available**.
   - **Batch Size:** 64-128 (balance between speed and stability).
   - **Train/Test Split:** 80% training, 20% testing.
   - **Epochs:** Start with **10-30 epochs**, use **early stopping** to prevent overfitting.

---

### **Next Steps**
✅ Finalize training parameters.
✅ Implement **MCTS** for better move selection.
✅ Deploy API and test from a webpage.

Let me know what you want to focus on next! 🚀♟️

You're absolutely right! In chess, there are often **many good moves** in a given position, so even a relatively low accuracy, like 28%, could still be significant. Here's why:

### **Chess Context:**
- **Multiple Good Moves**: Unlike other tasks where there's typically a single correct answer, chess positions can have **several good moves** depending on strategy, tactics, or player style. A model predicting one of those good moves can still be quite valuable, even if it's not "perfect."
- **Complexity of Chess**: Chess has a **massive search space**, meaning even a small improvement in accuracy can be significant. It's harder to achieve high accuracy in predicting moves because the number of valid moves grows exponentially with each game state.

### **Why 28% Could Be Okay:**
- **Competitive Environment**: Even a relatively modest level of accuracy can be useful when applied through methods like **Monte Carlo Tree Search (MCTS)** or **other decision-making algorithms**. These methods don't rely on 100% accuracy but instead use the predictions to guide the search, selecting moves based on probabilities.
- **Better than Random**: If the model is consistently identifying useful moves and not just random guesses, even a lower accuracy can still make the model a very competent player, especially when paired with techniques like **MCTS**.

### **What You Can Do:**
- **Track Improvements**: If the accuracy continues to improve over time, that’s a positive sign, even if it's a slow rate of change. Sometimes **small improvements** can lead to significant strategic advances.
- **Tuning**: You can try increasing the **model's complexity** (adding more layers, channels, etc.), but be mindful of the **risk of overfitting** or overcomplicating the model.
- **Enhance Search Algorithms**: You can leverage your model's output as a **guide** rather than the sole determinant of the best move. Using **MCTS** or similar algorithms allows the AI to simulate more moves and refine its decision-making.

### Final Thoughts:
**28% accuracy** might sound low, but in chess, even a model with modest success at predicting good moves can lead to **strong performance** when combined with additional strategies like MCTS. Keep testing and refining, and you could see steady improvements as the model and the search methods evolve!
