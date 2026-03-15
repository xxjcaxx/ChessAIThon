# Pedagogical Manual

## Chess and Human Intelligence

Chess is far more than a game; it is a structured environment for exercising the highest functions of human cognition. For VET students, it serves as a gym for the mind, where every move requires:

* **Critical Thinking and Pattern Recognition:** Identifying threats and opportunities within an $8 \times 8$ grid.
* **Decision-Making under Pressure:** Evaluating multiple variables to select a single optimal path.
* **Spatial Perception and Memory:** Visualizing the board state several moves ahead and recalling strategic motifs.

## Chess with Computers as IT Alphabetization

The history of computing is inextricably linked to chess. Using chess as a gateway to IT literacy allows students to understand that computers do not "see" or "think" like humans, but rather process data through representation.

* **Data Representation:** Learning how a physical board is translated into digital formats like **FEN** (Forsyth-Edwards Notation) or **UCI** (Universal Chess Interface).
* **Algorithmic Thinking:** Understanding the difference between **Brute-Force** (calculating every move) and **Selective Strategies** (using logic to narrow choices), a foundation laid by pioneers like Claude Shannon and Alan Turing.

## Chess AI: The First AI for Our Students

The **Chessmarro** model provides an accessible entry point into the world of Artificial Intelligence.

* **Learning by Doing:** Unlike high-resource models like AlphaZero, Chessmarro is trained on human-collected data, making it "cheap" and practical for students to train on standard hardware.
* **Predictive Logic:** Students witness firsthand how a Neural Network learns to predict the next best move by comparing its guesses against historical datasets.

## Soft Skills

The synergy between coding and chess fosters essential transversal skills necessary for the modern workforce:

* **Perseverance and Concentration:** Sustaining focus during long matches and debugging complex code.
* **Problem-Solving and Creativity:** Finding "lateral thinking" solutions to positional challenges on the board and logic errors in scripts.
* **Planning and Organization:** Managing time and resources, both in-game and during project development.

## Code Skills

Through the ChessAIthon project, students master fundamental programming concepts using chess logic as their "problem-solving" engine:

* **Functions and Logic:** Implementing the rules of chess (e.g., move validation) requires clean, functional programming.
* **Version Control:** Using **Git** and **GitHub** is integrated into the curriculum to manage datasets and collaborate on code.
* **Data Structures:** Working with diverse formats such as **JSON, CSV, and Parquet** files to store and process millions of moves.

## AI Skills

Students transition from consumers to creators of AI by engaging with deep learning core concepts:

* **Convolutional Neural Networks (CNNs):** Understanding how CNNs detect "spatial patterns" like King safety or pawn structures.
* **Tensors and Feature Engineering:** Encoding a board into a $77 \times 8 \times 8$ tensor that an AI can process.
* **Training and Optimization:** Learning how **Backpropagation** and **Optimizers** (like Adam) adjust a network's "weights" to minimize error.

## Systems Skills

The deployment of Chessmarro introduces students to modern systems administration and infrastructure:

* **Containerization with Docker:** Using `Dockerfile` and `docker-compose` to create portable environments.
* **GPU Computing:** Configuring the NVIDIA Container Toolkit to leverage hardware acceleration for model inference.
* **API and UI Integration:** Deploying a **Gradio** web interface and **FastAPI** backend to make the AI playable over a network.