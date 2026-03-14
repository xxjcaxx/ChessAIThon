import { Chess } from "chess.js";
import { BehaviorSubject } from "rxjs";

const createGameState = (game, lastMove, suggestedMoves = []) => {
  const ended = game.isGameOver();
  const checkmate = ended && game.isCheckmate();

  return {
    board: game.board(),
    currentPlayer: game.turn(),
    ended,
    winner: ended
      ? checkmate
        ? game.turn() === "w"
          ? "black"
          : "white"
        : "draw"
      : null,
    checkmate,
    endReason: ended ? (checkmate ? "checkmate" : "draw") : null,
    inCheck: game.inCheck(),
    legalMoves: game.moves({ verbose: true }).map((m) => m.lan),
    fen: game.fen(),
    lastMove: lastMove,
    suggestedMoves
  };
};

const createNoMoveEndState = (game, lastMove) => {
  const state = createGameState(game, lastMove, []);
  if (state.ended) {
    return state;
  }

  if (state.legalMoves.length === 0) {
    const checkmate = !!state.inCheck;
    return {
      ...state,
      ended: true,
      checkmate,
      endReason: checkmate ? "checkmate" : "stalemate",
      winner: checkmate
        ? state.currentPlayer === "w"
          ? "black"
          : "white"
        : "draw"
    };
  }

  return {
    ...state,
    ended: true,
    checkmate: false,
    endReason: "no_move",
    winner: "draw"
  };
};

const forceQueenPromotionIfNeeded = (game, uci) => {
  if (!uci || typeof uci !== "string" || uci.length !== 4) {
    return uci;
  }

  const matchingMove = game
    .moves({ verbose: true })
    .find((move) => `${move.from}${move.to}` === uci);

  if (matchingMove?.flags?.includes("p")) {
    return `${uci}q`;
  }

  return uci;
};

export class GameState {
  constructor(fen) {
    this.game = new Chess(fen);
    this.state$ = new BehaviorSubject(createGameState(this.game));
    this.aiStopped = false;
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
    if (this.aiStopped || newGameState.ended) {
      return;
    }

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

        if (!m?.move) {
          this.aiStopped = true;
          this.state$.next(createNoMoveEndState(this.game, this.lastMove));
          return;
        }

        const aiMove = forceQueenPromotionIfNeeded(this.game, m?.move);
        this.move = aiMove;
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