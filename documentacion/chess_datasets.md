# Making a Chess Dataset

The goal of our project is to develop an AI model capable of determining the best move in a game of chess. Inspired by groundbreaking projects like **AlphaZero** and **Leela Chess Zero**, we aim to adopt a similar final architecture for our model. However, our approach to training will differ significantly due to practical constraints. Unlike AlphaZero and Leela, which rely on **reinforcement learning**—a process where the AI learns by playing millions of games against itself—we will train our model using **collected data** from existing chess games. This decision is driven by the fact that we do not have access to the immense computational resources required for self-play training, such as those available to organizations like Google.

To build our AI model, we need a **high-quality dataset** consisting of chess positions paired with the best moves, as determined by human expertise. To start, we will use a **base dataset** obtained from the **Lichess** webpage database, which contains a vast collection of chess games and human decisions. To prepare this data for our AI, we applied a methodology similar to **ETL (Extract, Transform, Load)**, a technique commonly used in big data processes. This process allowed us to clean, organize, and structure the raw data into a usable format, which is now stored in `parquet` files. These files serve as the foundation for training our AI model.

However, to make our AI stand out from others, we need to go beyond the base dataset and create **additional, carefully curated data**. This involves selecting unique and instructive chess positions and pairing them with the best moves, as determined by human judgment. By doing so, we can enrich the dataset with high-quality examples that reflect nuanced strategies and decision-making, helping our AI learn more effectively.

To facilitate this process, we have developed a **web application** that allows users to interact with the dataset. Using this tool, you can download the existing dataset, review chess positions, and contribute new examples by adding your own insights on the best moves. These contributions can then be integrated into the training pipeline, further enhancing the AI's ability to make intelligent and human-like decisions.


## Format of the dataset

In the initial stages of our project, we will use **CSV files** due to their simplicity and tabular structure, which makes them easy to work with. Each line in the CSV file will contain two key pieces of information: the `FEN` representation of the chessboard and the `best move` in `UCI`  format. 

To facilitate collaboration and data sharing, our **web application** will enable users to download their contributed examples in CSV format. These files can then be uploaded to a **Git repository**, where they can be shared with the team and used in the preprocessing stage. This collaborative approach ensures that everyone can contribute to the dataset, enriching it with diverse and high-quality examples.

Once the CSV files are ready, we will use an **ETL (Extract, Transform, Load) tool** developed in a **Google Colab Notebook** to process the data further. This tool will transform the raw CSV data into a more advanced format suitable for training our AI model. Specifically, it will generate a **parquet file** with two additional columns: 

1. **77x8x8 Board Representation**: This is a numerical representation of the chessboard, formatted in a way that the AI model can easily process. It captures all the necessary information about the position, including piece locations, legal moves and other game states, in a structured 3D array.
2. **Move Representation (0 to 4096)**: Instead of using the UCI format, which is text-based, the move will be converted into a numerical value between 0 and 4096. This format is more efficient for the model to understand and process during training.

