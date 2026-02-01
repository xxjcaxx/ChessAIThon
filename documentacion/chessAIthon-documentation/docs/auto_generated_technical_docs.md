# Chessaithon Engine: Technical Documentation

## 1. System Overview

This system implements a high-performance Chess engine combining **Monte Carlo Tree Search (MCTS)** with a **Convolutional Neural Network (CNN)** for move prioritization. The architecture is designed for high concurrency, utilizing a Producer-Consumer pattern to separate CPU-bound search tasks from GPU-bound inference tasks.

The system is split into several distinct processes communicating via `multiprocessing.Queue`:

1. **API Server:** Handles external requests.
2. **Task Listener:** Orchestrates MCTS workers.
3. **MCTS Workers:** Perform parallel tree searches.
4. **Batcher:** Aggregates inference requests to optimize GPU throughput.
5. **Inference Server:** Executes tensor operations on the GPU.

---

## 2. Component Analysis

### 2.1. Entry Point & Orchestration (`main.py`, `api.py`)

The system initializes via `main.py`, which spawns the necessary child processes.

* **Lazy Model Loading:** The neural network model is loaded lazily using a thread-safe `Lock` mechanism to ensure it is initialized only once per process context.
* **Task Listener:** This process acts as a load balancer. It continuously polls the `task_q` for incoming FEN strings (board states). Upon receiving a task, it spawns `mp.cpu_count()` worker processes. It aggregates the visit counts from all workers to determine the final best move.
* **API Layer:** Built with FastAPI, it exposes a `/predict` endpoint. It uses a UUID to track request lifecycles, submitting tasks to the queue and blocking until the specific result ID appears in `tasks_result_q`.

### 2.2. The Inference Pipeline (`inference_server.py`, `batcher.py`)

To maximize GPU utilization, the system avoids processing single board states. Instead, it employs a dynamic batching mechanism.

**The Batcher (`batcher.py`):**
This component sits between the MCTS workers and the GPU. It collects individual inference requests and groups them into batches.

* **Trigger Conditions:** A batch is flushed to the GPU if it reaches `BATCH_SIZE` (32) or if a `TIMEOUT` (20ms) elapses. This ensures low latency for low-load periods while maintaining high throughput under load.
* **Routing:** The batcher maintains the `(id_worker, id_thread)` context for each request, ensuring that predictions are routed back to the exact thread that requested them.

**The Inference Server (`inference_server.py`):**

* **Vectorization:** Converts a batch of FEN strings into a tensor.
* **Legal Masking:** The model output (logits) is masked against legal moves. Illegal moves are assigned a large negative value (`negbig`) to prevent selection.
* **Temperature Scaling:** The logits are divided by a `temperature` parameter before Softmax, allowing control over the "sharpness" of the probability distribution.
* **Output:** Returns a list of legal moves sorted by their predicted probability.

### 2.3. Monte Carlo Tree Search (`mcts.py`)

The core logic resides here. Unlike traditional Minimax, MCTS builds an asymmetric tree based on simulations.

**The Node (`MCTSNode`):**
Each node represents a board state. It stores:

* `visits` (): How many times this node was explored.
* `value` (): The accumulated evaluation score.
* `prior_p` (): The probability of this move assigned by the CNN.
* `vloss`: Virtual Loss, used for threading synchronization.

**The Algorithm:**

1. **Selection:** The engine traverses from the root to a leaf node using the PUCT algorithm.
2. **Expansion:** If the leaf is not terminal, it is expanded. The Neural Network provides the *policy* (priors) for the new children.
3. **Evaluation:** Instead of a random rollout, this engine uses a hybrid approach. It uses a static heuristic function `_evaluate_position` (considering material, development, and King safety) to assign a value  to the leaf.
4. **Backpropagation:** The value  is propagated up the tree, updating  and .

---

## 3. Key Algorithmic Concepts

### 3.1. PUCT Formula (Selection Phase)

To balance **Exploitation** (visiting high-value nodes) and **Exploration** (visiting rarely seen nodes), the system uses the Predictor + UCT (PUCT) variant. The score for a child node is calculated as:

Where  represents the exploitation term:


And  represents the exploration term, weighted by the Neural Network's prior :


* **Virtual Loss (`vloss`):** In a multithreaded environment, multiple threads might select the same node before one finishes expanding it. To prevent this, a "virtual loss" is applied immediately upon selection, temporarily lowering the node's Q-value to discourage other threads from selecting the same path simultaneously.

### 3.2. Multithreaded Worker Execution

Each MCTS worker runs a `ThreadPoolExecutor` (default 4 threads).

* **Concurrency:** Threads share the same MCTS tree (`root`).
* **Locking:** A `threading.Lock` is used within `MCTSNode` to protect critical sections (updating visit counts or expanding children).
* **Blocking on Inference:** When a thread needs to expand a node, it sends a request to the `batcher_q` via a local queue and blocks. This releases the Global Interpreter Lock (GIL) effectively, allowing other threads to continue traversing the tree while waiting for the GPU response.

---

## 4. Data Flow & Concurrency Model

The system utilizes a complex queuing topology to manage data flow across process boundaries.

| Queue Name | Type | Producer | Consumer | Data Payload |
| --- | --- | --- | --- | --- |
| `task_q` | MP Queue | API | Task Listener | `(task_id, fen, simulations)` |
| `batcher_q` | MP Queue | MCTS Workers | Batcher | `(worker_id, (thread_id, fen))` |
| `inference_q` | MP Queue | Batcher | Inference Server | List of `(id_context, fen)` |
| `worker_response_queues` | List[MP Queue] | Batcher | MCTS Workers | Move Probabilities |

### 4.1. The "Worker-Thread" Routing Problem

A challenge in this architecture is routing the GPU result back to the specific *thread* that requested it, given that multiple processes and threads are active.

1. The worker generates a request containing `(t_id, board.fen())` where `t_id` is the thread identifier.
2. The Batcher receives this but treats it as opaque data. It wraps it with the worker ID: `(id_worker, (t_id, fen))`.
3. The GPU processes the batch and returns results paired with the opaque ID.
4. The Batcher puts the result into `worker_response_queues[id_worker]`.
5. The Worker process reads the queue, extracts `t_id`, and places the result into a thread-local `queue.Queue` (`thread_responses`), unblocking the specific thread waiting for that evaluation.

