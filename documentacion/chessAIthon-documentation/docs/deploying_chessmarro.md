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


### Responsibilities of the `ChessBatcher` functions

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

## ChessAIthon structure

### 1. Hierarchical Concurrency Model

A two-level structure was proposed to maximize resource utilization:

* **Process Level (Workers):** Utilization of multiple independent processes, ideally mapped to the physical cores of the CPU, to bypass the limitations of the Global Interpreter Lock (GIL) and enable true parallelism.
* **Thread Level (Threads):** Within each process, the execution of multiple simulation threads. These threads allow for the overlapping of GPU inference latency with the processing of the tree data structure on the CPU.

### 2. Communication and Synchronization Pattern

To manage the interaction between CPU threads and the GPU, a **Dispatcher/Collector** pattern was implemented using message queues (`multiprocessing.Queue` vs. `queue.Queue`):

* **Request Multiplexing:** A sender thread centralizes requests from all local simulation agents and redirects them to a centralizing component (the *Batcher*).
* **Response Demultiplexing:** A receiver thread listens to a return channel and distributes results to specific threads using private queues and unique identifiers. This mechanism ensures that the sequential nature of an MCTS simulation (where step  depends on the result of step ) is strictly respected without blocking other agents.

### 3. Throughput Optimization via Dynamic Batching

The central component of the system is the **Batcher**, whose function is to resolve the trade-off between latency and throughput:

* **Temporal and Quantitative Aggregation:** The system accumulates requests until a critical batch size () is reached or a safety timer () expires. This ensures that the GPU processes large volumes of data simultaneously, where its SIMT (*Single Instruction, Multiple Threads*) architecture is most efficient.
* **System Scalability:** It was determined that the optimal number of threads per physical core depends on the GPU's capacity to process batches. An excess of threads increases queue latency without improving performance, while a shortage underutilizes the specialized hardware.



### Design Synthesis

| Component | Technical Implementation | Academic Objective |
| --- | --- | --- |
| **Worker** | `multiprocessing.Process` | Memory isolation and true parallelism. |
| **Simulation** | `threading.Thread` | Reactive concurrency and latency hiding. |
| **Inter-process Communication** | `mp.Queue` (Serialized) | Message passing in distributed memory environments. |
| **Intra-process Communication** | `queue.Queue` (Referential) | Low-latency data exchange in shared memory. |

This design represents a robust architecture for search problems in massive state spaces, where efficiency depends on precise orchestration between asynchronous control logic and synchronous data processing.


### 1. Root Parallelism and Determinism

Root parallelism consists of running multiple independent instances of the MCTS tree. The central dilemma is preventing all processes from exploring the exact same lines (determinism), which would negate the advantage of parallelism.

* **Dirichlet Noise:** Identified as the standard for injecting entropy into the root node, forcing each process to prioritize different candidate moves.
* **Hyperparameter Variation ():** A heterogeneous search strategy was implemented where each worker operates with a distinct exploration coefficient. Low values encourage **exploitation** (depth in main lines), while high values encourage **exploration** (breadth in secondary lines).

### 2. Throughput Optimization: The Batcher and GPU

A recurring problem in neural network inference is GPU underutilization.

* **Batch Efficiency:** It was analyzed that processing small amounts of positions is inefficient. The proposed solution was to increase the number of simultaneous requests via internal threads within each process.
* **Asynchronous Communication:** A system of queues and specialized processes (`sender`/`receiver`) was established to manage data flow between the search engines and the inference server, using unique identifiers to ensure each response returns to the correct thread.

### 3. Tree Parallelism and Virtual Loss

When introducing multiple threads exploring the same tree, there is a risk that all threads will rush toward the move currently considered "best," even before receiving the network's evaluation.

* **Virtual Loss:** A fundamental technique that temporarily penalizes a node's score while it is being evaluated by a thread. This "disincentivizes" other threads, forcing them to explore alternative nodes and ensuring effective parallel expansion of the tree.
* **Concurrency Management:** The need for granular locks (`Locks`) was discussed to protect the integrity of node data (visits, value) without stalling chess logic or inference execution.

### 4. Consolidation of Results

The final phase of the algorithm translates the individual intelligence of each worker into a single global decision.

* **Aggregation by Visits:** Academically, summing the visits from all processes is more robust than averaging their values. By consolidating search statistics from all 24 processes, the statistical "consensus" acts as a filter against neural network evaluation errors.


## ChessAIthon vs Stockfish

The behavior of a **Monte Carlo Tree Search (MCTS) system guided by a neural network** applied to chess was analyzed, systematically comparing its decisions with those of Stockfish and studying the effect of the number of simulations on the quality of the selected moves.

Empirical analysis of complete games showed that, in a significant proportion of positions, the move ultimately chosen by the MCTS coincides with the move initially prioritized by the neural network. Although this initial policy has limited accuracy (around 25% depending on the training), the MCTS fails to consistently reverse these preferences when the network makes mistakes, even when increasing the number of simulations from 200 to 4000. This phenomenon was clearly observed in direct comparisons between three columns: Stockfish, MCTS with few simulations, and MCTS with many simulations.

Detailed analysis of specific positions, especially endgames, revealed that the system tends to consider as very bad moves that Stockfish identifies as optimal or necessary to prolong resistance. This was explained by the weakness of the evaluation function used: a heuristic based primarily on material balance, simply normalized, which produces values ​​close to zero for many strategically very different positions. As a result, multiple leaves of the tree return similar evaluations, drastically reducing the MCTS's ability to discriminate between alternatives.

From an algorithmic perspective, it was concluded that the MCTS is dominated by the network's initial policy. The search process focuses early on moves with the highest preprobability, and the evaluation is not informative enough to quickly penalize incorrect moves. In this context, the search term (PUCT) amplifies the network's initial preferences instead of correcting them, which explains the small difference between running few or many simulations.

The implicit comparison with AlphaZero clarified this limitation. Unlike the analyzed system, AlphaZero relies on a trained value function to predict the final game outcome, allowing suboptimal moves to be quickly discarded during the search. Furthermore, it uses softer policies and explicit search mechanisms, preventing the search from becoming prematurely anchored to initial preferences.

The main conclusion is that the system has reached the information limit of its current evaluation function. The MCTS operates stably and consistently, and the neural network effectively influences decisions, but the search cannot substantially improve the quality of moves without a richer value signal. Consequently, increasing the number of simulations does not produce appreciable improvements except in positions where several moves are approximately equivalent.

In general terms, the analysis confirms a fundamental principle: the MCTS does not generate knowledge on its own, but rather amplifies the quality of the policy and the evaluation it receives. To overcome the observed limitations, it would be necessary to introduce a learned value function, reduce the excessive dependence on the initial policy, or explicitly strengthen the exploration mechanisms, especially in endgames.

Below is a structured, academic summary of the entire discussion, focused exclusively on the strategic and conceptual ideas, not on specific technical errors or bugs.


### 1. Fundamental Difference Between Classical MCTS and AlphaZero-Style MCTS

**Traditional MCTS**

* Uses long playouts until the end of the game (or a fixed depth).
* Simulations are usually fast and inexpensive.
* The evaluation is obtained from the final result of the simulated game.

**AlphaZero-Style MCTS**

* Does not perform long playouts.
* In each expansion:
  * The neural network is called only once.
  * The network returns:
    * **Policy (P):** probability distribution of moves.
    * **Value (V):** static evaluation of the position.
* The simulation ends immediately after the expansion.
* The value `V` is backpropagated through the tree.

**Strategic Conclusion:**
The goal is not to simulate complete games, but to **build a probability distribution of strong moves** using searches guided by prior knowledge.


### 2. Main Bottleneck: Neural Network Inference

* Calling a CNN at each step of the playout destroys performance.
* AlphaZero avoids this:
* One inference **per expanded node**, not per simulated move.
* The performance of modern MCTS depends more on:
* prior quality
* correct value propagation
* exploration/exploitation balance

**Correct Strategy:**
Reduce inferences and increase selection quality (PUCT).


### 3. Using PUCT instead of Classical UCT

The strategic formula used is:


  PUCT(s,a) = Q(s,a) + c_{puct} \cdot P(s,a)\frac{\sqrt{N(s)}}{1 + N(s,a)}


* `Q`: average learned value
* `P`: network prior (or equivalent heuristic)
* `N`: visits

**Key idea:**
Exploration is not uniform; promising moves are prioritized based on prior knowledge.


### 4. Handling Value and Perspective Alternation

* Value should always be interpreted from the perspective of the player who moves.
* In backpropagation:
* the sign of the value is reversed at each level of the tree.
* This eliminates the need to explicitly distinguish between max and min.

**Strategic Principle:**
A single scalar value is sufficient if turn alternation is handled correctly.


### 5. Number of Simulations: Orders of Magnitude

There is no universal fixed number.
* AlphaZero Engines:
* 800–1600 simulations per move (chess)
* Lightweight Projects:
* 100–400 → reasonable decisions
* 1000+ → strategic structure begins to emerge

**Key Idea:**
The quality of the prior is more important than the raw number of simulations.


### 6. Correct Interpretation of Evaluations Close to Zero

In openings and early middlegames:

* Values ​​close to 0.0 are normal
* They indicate equilibrium, not system failure


### 7. Conceptual Difference between Stockfish and AlphaZero-type Engines

**Stockfish**

* Alpha-beta search
* Deterministic NNUE evaluation
* Produces a concrete best move

**AlphaZero / MCTS**

* Produces a **probability distribution**
* The best move is defined by:
* number of visits
* not necessarily the highest immediate value

**Strategic Implication:**
Comparing engines requires looking at more than just the “best move”.


### 8. Fair Comparison between MCTS and Stockfish

For the comparison to be scientifically valid:

* Same time per move
* Non-trivial positions (no immediate checkmates)
* Metrics:
* match with Stockfish's top-N
* stability of choice
* strategic consistency

**Central idea:**
The goal is not to "beat Stockfish," but to **validate the quality of guided search**.


### General Conclusion

The strategy followed in this project is consistent with the modern philosophy of chess engines:
* abandonment of long playouts
* use of informed priors
* static evaluation and backpropagation
* comparison by distribution and not just by best move
* real-time benchmarks


# **ChessAIThon – Deployment Manual (Docker + NVIDIA GPU Support)**

This guide explains how to deploy the *ChessAIThon* model on an Ubuntu system using Docker, Docker Compose, and the NVIDIA Container Toolkit for GPU acceleration.

---

## **1. Update System and Install Required Packages**

Update package lists and install certificates + curl:

```bash
sudo apt update
sudo apt install ca-certificates curl
```

---

## **2. Add Docker’s Official GPG Key and Repository**

Create directory for Docker’s keyrings:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
```

Download Docker’s GPG key:

```bash
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

Add Docker repository:

```bash
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF
```

Update package lists:

```bash
sudo apt update
```

---

## **3. Install Docker Engine and Plugins**

```bash
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify Docker works:

```bash
sudo docker run hello-world
```

Check Docker Compose:

```bash
docker compose
```

---

# **4. Clone the ChessAIThon Repository**

```bash
git clone https://github.com/xxjcaxx/ChessAIThon.git
cd ChessAIThon/
ls
```

Navigate to the deployment directory:

```bash
cd modelDeploy/
ls
```

---

## **5. Install NVIDIA Drivers (for GPU Support)**

Check for NVIDIA GPU:

```bash
lspci | grep -i nvidia
```

Automatically install the recommended drivers:

```bash
sudo ubuntu-drivers autoinstall
```

Reboot:

```bash
sudo reboot
```

After reboot, confirm GPU is detected:

```bash
nvidia-smi
```

---

## **6. Place the Trained Model in the Deployment Folder**

Move your trained model file into *modelDeploy/*:

```bash
mv modelo_entrenado_chessintionv2.pth ChessAIThon/modelDeploy/
cd ChessAIThon/modelDeploy/
ls
```

---

## **7. (Initial Attempt) Start Docker Compose**

```bash
sudo docker compose up -d
sudo docker compose logs -f
```

If the container fails due to missing GPU runtime, continue with the next section.

---

## **8. Install NVIDIA Container Toolkit (GPU Support for Docker)**

Add NVIDIA GPG key:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

Add the NVIDIA container repository:

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Update package lists:

```bash
sudo apt update
```

Install NVIDIA Docker toolkit:

```bash
sudo apt install -y nvidia-container-toolkit
```

Configure Docker to use the NVIDIA runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

Restart Docker:

```bash
sudo systemctl restart docker
```

Verify GPU works inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.0-base nvidia-smi
sudo docker run --rm --gpus all nvidia/cuda:12.3.0-base nvidia-smi
```

---

## **9. Start the ChessAIThon Deployment with GPU Support**

Navigate to the deployment directory:

```bash
cd ChessAIThon/modelDeploy/
```

Run Docker Compose:

```bash
sudo docker compose up -d
sudo docker compose logs -f
```

This should successfully start the model with GPU acceleration.


