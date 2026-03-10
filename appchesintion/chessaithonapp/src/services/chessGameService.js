import { Chess } from "chess.js";
import { BehaviorSubject } from "rxjs";

const createGameState = (game, lastMove, suggestedMoves = []) => ({
  board: game.board(),
  currentPlayer: game.turn(),
  ended: game.isGameOver(),
  winner: game.isGameOver()
    ? game.isCheckmate()
      ? game.turn() === "w"
        ? "black"
        : "white"
      : "draw"
    : null,
  inCheck: game.inCheck(),
  legalMoves: game.moves({ verbose: true }).map((m) => m.lan),
  fen: game.fen(),
  lastMove: lastMove,
  suggestedMoves
});

export class GameState {
  constructor(fen) {
    this.game = new Chess(fen);
    this.state$ = new BehaviorSubject(createGameState(this.game));
    //this.b = 'Human';
    //this.w = 'Human';
    this.players = {
      w: "human",
      b: "human",
      wApi: "",
      bApi: "",
    };
    this.settings = {
      simulations: 400,
      puct: 1.0,
      suggestOnly: false
    };
  }
  set move(uci) {
    try {
      this.game.move(uci);
      this.lastMove = uci
    } catch (error) {
      console.log(error.message);
    }
    const newGameState = createGameState(this.game, uci, []);
    this.state$.next(newGameState);
    this.decideNextMove(newGameState);
  }
  decideNextMove(newGameState) {
    console.log(
      "decide" + this.players[newGameState.currentPlayer],
      newGameState.currentPlayer
    );

    if (this.players[newGameState.currentPlayer] == "ai") {
      console.log("decide AI move", this.players);
      // this.game.move(newGameState.legalMoves[Math.floor(Math.random() * newGameState.legalMoves.length)]);
      // Use the shared request function; do not send mcts_tree for automated next-move decisions
      requestAIMove({
        url: this.players[newGameState.currentPlayer + 'Api'],
        fen: newGameState.fen,
        simulations: this.settings?.simulations || 400,
        puct: this.settings?.puct || 1.0
      }).then(m => {
        if (this.settings?.suggestOnly) {
          const alternatives = Array.isArray(m?.alternatives) ? m.alternatives : [];
          const topFive = [m?.move, ...alternatives.map(a => a?.[0])]
            .filter(Boolean)
            .slice(0, 5);
          this.state$.next(createGameState(this.game, this.lastMove, topFive));
          return;
        }
        this.move = m.move;
      }).catch(err => console.error('AI request failed', err));

    }
  }
}


export const askAIMoveTree = async (url, fen, simulations, puct = 1.0) => {
  return await requestAIMove({ url, fen, simulations, puct, mcts_tree: true });
};

/**
 * Generic function to request AI move from a server.
 * Accepts optional `mcts_tree` that will be forwarded when provided.
 * Parameters: { url, fen, simulations=400, puct=1.0, mcts_tree }
 */
export const requestAIMove = async ({ url, fen, simulations = 400, puct = 1.0, mcts_tree } = {}) => {
  const body = { fen, simulations, puct };
  if (typeof mcts_tree !== 'undefined') body.mcts_tree = mcts_tree;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });

  const data = await response.json();
  return data;
};