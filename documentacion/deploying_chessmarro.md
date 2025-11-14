# Deploy ChessMarro

The neural network behind ChessMarro can identify moves that appear promising, but it lacks long-term strategic understanding. To address this, we integrate Monte Carlo Tree Search, allowing the engine to run guided simulations on the most promising moves and select the one that performs best over time.

We also need to deploy the system on a machine equipped with a GPU—either in the cloud or locally. To simplify this process, we provide a ready-to-use Docker environment.

## Docker

To build a GPU-enabled Docker setup, we include the standard files: `requirements.txt`, `Dockerfile`, and `docker-compose.yml`. You can find their exact contents in the repository. Our image is based on `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`, which provides full support for NVIDIA GPUs.

The container exposes both a Gradio API and a web interface through `app.py`.

For quick local testing, you can simply run `predict.py`.

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

Using an AI-guided policy inside MCTS improves performance because it gives the search a meaningful direction from the very first simulation. Pure Monte-Carlo rollouts are noisy: they treat all legal moves equally and rely on random playouts to discover which branches are promising. This means the algorithm wastes many simulations exploring moves that a minimally competent player would never consider. When the branching factor is large, as in chess, random rollouts require tens or hundreds of thousands of simulations before the signal emerges clearly from the noise.

By contrast, injecting AI evaluations provides the tree with an initial bias toward moves that are already known to be reasonable. The policy estimates from the AI act as priors, shaping the search so that MCTS spends more time on moves with higher predicted quality and less time on obviously weak ones. The result is a much faster convergence of node values: the algorithm does not need to “rediscover” good moves from scratch through random sampling but can refine and correct the AI’s suggestions through proper exploration.

This combination preserves what makes MCTS strong—balanced exploration and exploitation—while dramatically accelerating the learning curve of the tree. Even if the AI is imperfect, a moderately accurate prior reduces variance in the results, improves stability between runs, and allows the search to focus its computational budget on realistic candidates. In practice, this leads to higher playing strength with far fewer simulations than pure random MCTS could ever achieve.


## Optimizing MCTS

To understand how a system can efficiently coordinate threads and processes so that many small inference requests are merged into large batches, enabling **maximum GPU throughput** during neural network evaluation.

Monte Carlo Tree Search (MCTS) generates **thousands of independent inference requests**, each corresponding to evaluating a chess position. A naive system would send each request one-by-one to the GPU—this is extremely inefficient:

- Each GPU inference has overhead: kernel launch, memory copies, synchronization.
- MCTS threads produce requests at irregular, unpredictable intervals.
- The GPU remains mostly **idle**, processing tiny batches (size = 1).
- CPU threads block waiting for the model, slowing the entire search.

A GPU is most efficient when processing **large, well-formed batches**.
If you feed the model with batch sizes of 16, 32, 64, etc., throughput increases dramatically.

**The batcher’s job is to collect scattered requests and merge them into large batches automatically.**


### Responsibilities of the `ChessBatcher` Class

The batcher acts as a **traffic controller** for all requests generated by MCTS threads.

It is built around **three communication channels**:


| Component            | Purpose                                           |
| -------------------- | ------------------------------------------------- |
| **input_queue**      | Sends batches to the GPU worker process           |
| **output_queue**     | Receives results from the GPU worker              |
| **pending[task_id]** | Stores a dedicated response queue for each caller |

---

### How Batching Happens Automatically

The batcher accumulates requests in `current_batch` until one of two conditions triggers a flush:


- If the batch reaches a predefined target (e.g., 32 positions) This guarantees large, optimal batches whenever demand is high.
- A background thread wakes up periodically (e.g., every 100 ms) and checks if the batch is non-empty. This prevents starvation:
  - Even if demand is low, no request waits too long.
  - Ensures reasonable latency for single or sparse requests.

---

### Processes and Queues

The architecture mixes **threads** and **processes** to separate concerns:

**`Process`: GPU Worker**

- Runs isolated from the main process.
- Holds the model on the GPU.
- Waits on `input_queue`, performs inference, returns results via `output_queue`.

This improves:

- Parallelism (CPU threads run MCTS while GPU worker blocks on inference)
- Stability (segfaults or CUDA errors stay isolated)

**`Manager.Queue`**

Used for:

- Response queues for each MCTS task
- Interprocess-safe shared dictionary `pending`

**`dispatch_loop` Process**

The dispatcher is a dedicated process that:

- Continuously reads `output_queue`
- Looks up the correct response queue using `task_id`
- Sends predictions back to the right MCTS thread

This creates an **asynchronous but reliable return path**.

---

### Threads Inside the Batcher

The batcher itself uses threads for responsiveness:

- Runs on a fixed interval.
- Sends small batches when timing out.
- Avoids blocking the main application.

Main Application Threads (MCTS):

- Many MCTS simulations run concurrently.
- Each simulation calls `add_board()` to request a prediction.
- Each waits asynchronously on its personal response queue.

GPU Worker Process:

- Technically separate from threads.
- Executes model inference using CUDA.


### Performance Optimizations

The batcher implements several advanced techniques to ensure the GPU is used optimally and efficiently.

When moving tensors from CPU → GPU, pinned (page-locked) memory enables:

- Faster DMA transfers
- Overlap between data transfer and kernel execution
- Enables the GPU to begin computation while copying from CPU memory.
- This reduces latency and improves concurrency.
