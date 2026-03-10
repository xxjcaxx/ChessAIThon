# ChessAIthon Competition

## Play Chess Scenarios

In this phase, teams should organize themselves to play as many scenarios as possible, including full games and selected positions, in order to build the largest possible human-generated dataset.

For each position, the team must define both:
- the best move (policy), and
- the value, which represents the expected chance of winning from the current position, independently of the selected policy move.


## Prepare data

After playing, if multiple browsers were used, download all CSV files, merge them carefully without duplicating headers, and upload the result to GitHub.

You can do this daily as a backup process, or once at the end of the collection phase.

This CSV must then be converted into the Parquet format required for AI training.

To make this step easier, a user-friendly Jupyter Notebook will be provided so each team can load its CSV and export the corresponding Parquet file.

## Train your AI Chess Model

Using a Jupyter Notebook and a GPU-enabled environment (local machine, Colab, or Kaggle), each team will fine-tune the provided base model with its own Parquet dataset.

The result will be a trained model that can be shared on Kaggle and Hugging Face.

## Deploy your model


On a GPU machine, a Docker setup with the MCTS engine adapted to our project will be available.

Each team must place its fine-tuned model in the designated folder and run the following command to start the service:

```bash
docker compose down && docker compose up -d && docker compose logs -f
```

Take note of the machine IP and port (`8000` by default). This Docker service exposes an API that can be queried as follows:

```bash
curl -X POST "http://127.0.0.1:8000/predict"      -H "Content-Type: application/json"      -d '{"fen": "7r/p2q1pk1/1pp3p1/8/6P1/4Q3/PP1R1P1r/5KN1 b - - 0 38", "simulations": 400, "mcts_tree": false}' 
```

However, the best option is to configure this endpoint directly in the same web application used during the initial play phase, so you can test games against your AI.

In the Play section, you can choose the number of simulations, set the `puct` value (recommended: `1.4`, as the confidence balance between AI prior and search exploration), and decide whether the engine should play automatically or suggest best moves.