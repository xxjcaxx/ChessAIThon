# ChessAIthon Competition

## Play Chess Scenarios

In this phase, teams should organize themselves to play as many scenarios as possible, including full games and selected positions, in order to build the largest possible human-generated dataset.

For each position, the team must define both:
- the best move (policy), and
- the value, which represents the expected chance of winning from the current position, independently of the selected policy move.

In order to do this you can use the `Chess Minds` application available in: https://chess-ai-thon.vercel.app/ 

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

This is the output of the docker compose logs:

```bash
chessaithon-ai  | INFO:     172.19.0.1:59878 - "POST /predict HTTP/1.1" 200 OK
chessaithon-ai  | ♟  Request received | sims=10000
chessaithon-ai  | FEN: rnb1k1nr/pppp1ppp/5q2/2bP4/8/5N2/PPP1PPPP/RNBQKB1R w KQkq - 3 6
chessaithon-ai  | 
chessaithon-ai  | ♜ ♞ ♝ · ♚ · ♞ ♜
chessaithon-ai  | ♟ ♟ ♟ ♟ · ♟ ♟ ♟
chessaithon-ai  | · · · · · ♛ · ·
chessaithon-ai  | · · ♝ ♙ · · · ·
chessaithon-ai  | · · · · · · · ·
chessaithon-ai  | · · · · · ♘ · ·
chessaithon-ai  | ♙ ♙ ♙ · ♙ ♙ ♙ ♙
chessaithon-ai  | ♖ ♘ ♗ ♕ ♔ ♗ · ♖
chessaithon-ai  | 🌳 Root moves after noise (top 10):
chessaithon-ai  |  [('b1c3', '0.379'), ('e2e4', '0.154'), ('c1g5', '0.378'), ('e2e3', '0.089')]
chessaithon-ai  | Worker 0 [PUCT 1.4]: Max Depth: 25, Avg Depth: 10.86
chessaithon-ai  | Root visits: 9966
chessaithon-ai  | Sum children: 9966
chessaithon-ai  | 🔮 Root children predicted values:  b1c3:-0.707 | e2e4:-0.842 | c1g5:-0.658 | e2e3:-0.812
chessaithon-ai  | 🏆 Best move: b1c3  (7866 visits)
chessaithon-ai  | NN's Intuition: b1c3 (0.470) | e2e4 (0.183) | c1g5 (0.081) | e2e3 (0.074) | c1e3 (0.034) | c2c4 (0.032) | c1f4 (0.023) | b1d2 (0.022) | a2a3 (0.019) | d1d4 (0.014) | d5d6 (0.012) | c1d2 (0.008) | b1a3 (0.006) | f3d4 (0.003) | f3d2 (0.003) | c1h6 (0.002) | c2c3 (0.002) | h2h3 (0.002) | d1d3 (0.002) | d1d2 (0.001) | g2g3 (0.001) | a2a4 (0.001) | h1g1 (0.001) | f3e5 (0.000) | b2b3 (0.000) | e1d2 (0.000) | f3g1 (0.000) | f3g5 (0.000) | h2h4 (0.000) | b2b4 (0.000) | g2g4 (0.000) | f3h4 (0.000)
chessaithon-ai  | Other explored: e2e3 [1360v] ⮕  e2e4 [588v] ⮕  c1g5 [152v]
chessaithon-ai  | 📊 Batcher: avg=16.318045112781924 | incomplete=50.0 | total=1330.0
chessaithon-ai  | 🚀 Processing time: 9.54  s
```
It shows the model and simulations information:
- Request information.
- Chess board representation in Unicode
- The top 10 moves predicted by the NN with the priority of each one after apply Dirichlet Noise.
- The MCTS worker with puct and simulation depth information
- The visits of Root and children
- The predicted values of children by NN
- The best move with number of visits
- The complete raw prediction by NN to compare with noise and MCTS results.
- Other explored moves.
- The batcher of NN 
- The processing time. 

Sometimes it shows:

```bash
 ⚠  MCTS CORRECTION  Red preferred g8f6 but Search chose b8c6
```
This means that MCTS simulation has corrected the prediction of the NN. Most of the times is good, because simulation detected a better move. 


## Test model online

Go to https://chess-ai-thon.vercel.app/#play. 

You can configure AI in both players. We have deployed the AI server in Huggingface: https://jocasal-chessaithon.hf.space, so you have to put in AI API URL input https://jocasal-chessaithon.hf.space/predict and start playing. 

> This application is simple to use, but don't hide any technical question. You can deploy the model in your computer and put your URL in this input. You can see the output of the AI server in the console of the web navigator. 


