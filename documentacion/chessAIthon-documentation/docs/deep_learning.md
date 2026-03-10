# Deep Learning

This chapter explains the deep learning foundations used in **ChessAIThon** and how they connect to the **Chessmarro** engine.

## 1. Core Concepts

These are the minimum concepts needed to understand how the model works.

| Concept | Description | Why it matters in ChessAIThon |
| :--- | :--- | :--- |
| **Neural Networks** | Models that learn patterns from data through stacked layers of neurons. | `ChessNet` learns how board patterns relate to strong candidate moves. |
| **Convolutional Neural Networks (CNNs)** | Neural networks specialized for grid-like data. | A chessboard is an $8 \times 8$ grid, so CNNs are well-suited to detect spatial patterns. |
| **Tensors and Encoding** | Numeric representation of board states for model input. | The model operates on encoded board tensors, not on raw text notation. |
| **Backpropagation + Optimizer** | Process that updates weights to reduce prediction error. | Training improves move prediction quality over many iterations. |
| **Activation Functions** | Non-linear functions that let deep models learn complex relations. | ReLU-like activations in hidden layers help stable and efficient training. |

---

## 2. From Prediction to Play: Policy + MCTS

A neural network alone is a predictor. A chess engine needs search.

- The network outputs a **policy**: probabilities over legal candidate moves.
- **MCTS (Monte Carlo Tree Search)** uses that policy to prioritize which branches to explore.
- This combination improves practical playing strength compared with using either component alone.
- The objective is not perfect top-1 prediction on every position, but useful guidance for search.

---

## 3. Training and Generalization

Model quality depends on both data quality and training discipline.

- **Dataset quality:** broad and clean chess positions produce more robust behavior.
- **Generalization:** the model must perform on unseen positions, not only memorized examples.
- **Overfitting risk:** increasing complexity without control can reduce real match performance.
- **Evaluation:** monitor loss and move prediction metrics to verify that training is improving.

---

## 4. Practical Stack in the Project

The deep learning workflow is integrated into a full deployment pipeline.

| Component | Role in the system |
| :--- | :--- |
| **PyTorch (`torch`)** | Defines and runs the neural model for training and inference. |
| **FEN / UCI** | Standard input/output interfaces for chess positions and moves. |
| **Batching + Queues** | Groups inference requests to use compute resources efficiently. |
| **Docker + GPU** | Provides reproducible environments and faster inference at deployment time. |

---

## 5. Neural Network Architecture (Project View)

At a high level, Chessmarro follows the standard neural pipeline:

1. **Input layer:** encoded board state as tensor channels.
2. **Hidden CNN blocks:** extraction of tactical and positional features.
3. **Output head:** move probabilities used by the search module.

Some optimized variants may include additional architectural blocks (for example residual connections, channel reweighting, or alternative activations), but the core interpretation remains the same: the network estimates move quality, and MCTS turns those estimates into stronger decisions.

---

## 6. Activation Functions (Concise Comparison)

- **Sigmoid:** useful for bounded outputs in specific cases, less common in deep hidden stacks.
- **Tanh:** zero-centered output in $[-1, 1]$, sometimes useful in value-style outputs.
- **ReLU family:** default choice for hidden layers due to simplicity and training stability.
- **Advanced options (e.g., Mish):** can improve convergence in some settings, at higher complexity.

In this project context, hidden layers prioritize stable and efficient activations, while bounded activations are reserved for output behaviors that require explicit numeric ranges.

---

## 7. Summary

Deep learning in ChessAIThon is best understood as a **two-part system**:

1. A CNN-based model that evaluates candidate moves from encoded board states.
2. A search procedure (MCTS) that uses those predictions to choose stronger moves.

This architecture gives students practical exposure to modern game AI while keeping the conceptual path clear: **representation → prediction → search → decision**.
