## Montecarlo Tree Search

Monte Carlo Tree Search (MCTS) is a way for a computer to play chess (or other strategy games) by looking ahead at possible moves and choosing the best one.
Let’s break it down in simple steps:

### 1. **Making a Move Tree**
   - Imagine a tree where each branch is a different move the player can make.
   - Each branch splits into more branches based on how the opponent could respond.
   - The tree keeps growing as more moves are considered.

### 2. **Running Simulations (Playing Random Games)**
   - Instead of analyzing every move perfectly, MCTS picks a move, then plays random moves until the game ends.
   - It does this thousands or even millions of times.

### 3. **Checking the Results**
   - The computer counts how often each move leads to a win, loss, or draw.

### 4. **Choosing the Best Move**
   - It picks the move that has the best chance of leading to a win based on the simulations.


- It doesn’t need a huge database of chess knowledge.
- It gets better the longer it runs.
- It works well even when the game has too many possible moves to check them all.

A chess game tree is incredibly vast, with too many possible moves to explore fully. Even the most powerful computers cannot analyze every single possible game path, so instead of attempting an exhaustive search, we must focus on exploring the most promising moves. This is exactly what Monte Carlo Tree Search (MCTS) does. It doesn’t try to evaluate every move equally but instead finds the best ones through continuous exploration and learning.  

At the beginning of the search, MCTS selects moves randomly from all legal options. Since it doesn’t have any prior knowledge about which moves are good or bad, it treats them all as potential choices. As the simulation continues and more games are played, the search begins to favor moves that have led to successful outcomes more often. This means that over time, stronger moves are explored more deeply, forming a refined understanding of which strategies are most effective.  

However, MCTS does not completely forget about other moves. Instead, it occasionally selects less-explored options to ensure that no potentially strong move is overlooked. This balance between focusing on known strong moves (exploitation) and testing new possibilities (exploration) is what makes MCTS powerful. By maintaining this balance, the algorithm avoids getting stuck in local patterns and instead continues searching for optimal strategies.  

To manage this balance, MCTS uses a mathematical formula, such as the Upper Confidence Bound (UCB1), to decide whether to explore a new move or reinforce an already successful one. This helps MCTS refine its search intelligently, improving the quality of decisions over time without requiring exhaustive calculations. By repeatedly simulating games and adjusting its choices, MCTS efficiently finds strong moves, even in complex games like chess where brute-force search alone would be impractical.

Our approach improves Monte Carlo Tree Search (MCTS) by combining it with a neural network, making the simulations much more efficient. In standard MCTS, early moves are chosen randomly, and only through many simulations does the algorithm start favoring the best ones. This randomness means MCTS needs a large number of simulations to reach strong conclusions. By integrating a neural network, we guide MCTS toward better moves from the start, reducing the number of random choices and making each simulation more meaningful.  

The neural network acts as a smart evaluator, predicting which moves are most promising based on patterns it has learned from previous games. Instead of exploring all legal moves equally, MCTS now prioritizes those that the neural network suggests as strong candidates. This means that even with fewer simulations, the search process becomes more efficient, as it focuses on high-quality moves rather than wasting time on obviously weak options.  

By reducing the randomness in the early stages and ensuring that each simulation carries more weight, our solution allows MCTS to reach better decisions much faster. 

In a typical chess position, there are around 50 legal moves available. If we were to explore every possible move at each step, the game tree would expand extremely quickly. For example, after just one move, we would have 50 possibilities, and after two moves, the tree would already contain 2,500 (50 × 50) different positions.   

To address this, our approach uses an AI to select the most promising moves, significantly reducing the number of branches MCTS needs to explore. Instead of considering all 50 legal moves, the AI filters them down to a smaller subset—typically between 3 and 10 of the best moves. This allows us to focus only on the most relevant options, making the simulations much faster and more efficient.  

However, this approach comes with a risk. If the AI makes a mistake and fails to include the best move in its selection, MCTS will never consider it, potentially leading to a weaker decision. Despite this risk, narrowing down the choices allows the system to focus more on high-quality moves and conduct deeper simulations within the available time. By shifting more of the decision-making responsibility to the AI rather than relying entirely on MCTS, we create a system particularly useful for this project.

### Our MCTS

We provide the code to deploy it. You can change some input variables to make it better, but be careful because you can make it slower or impossible to run in the machine. 


## The deploy

Our AI is able to choose the best move, but we need to use it from the outside of our notebook, so we need to deploy to the outside. Gradio makes a web service that can be reached by web browser and as an API for other applications. We provide the code and we don't recommend to touch it. It should work and you will copy the URL and paste it in the our web platform to try it. 

## Services

We will use Huggingface to upload the model. In the examples you have a base to start, but your AI will have to replace it in the code. Your trained AI, in huggingface will be downloaded by a Google Colab Notebook to deploy it and you will use the deploy as an API for your AI player in competition. Once you choose your player and the other player, they will play with limited time some games to get the best. 