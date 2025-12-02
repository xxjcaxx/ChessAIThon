import { Chess } from "chess.js";
import { BehaviorSubject } from "rxjs";

const createGameState = (game) => ({
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
  }
  set move(uci) {
    try {
      this.game.move(uci);
    } catch (error) {
      console.log(error.message);
    }
    const newGameState = createGameState(this.game);
    this.state$.next(newGameState);
    this.decideNextMove(newGameState);
  }
  decideNextMove(newGameState) {
    console.log(
      "decide" + this[newGameState.currentPlayer],
      newGameState.currentPlayer
    );

    if (this.players[newGameState.currentPlayer] == "ai") {
      console.log("decide AI move", this.players);
      // this.game.move(newGameState.legalMoves[Math.floor(Math.random() * newGameState.legalMoves.length)]);
      fetch(this.players[newGameState.currentPlayer+"Api"], {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            fen: newGameState.fen,
            simulations: 10  // mejor como número
        })
    }).then(response => response.json()).then(m =>{
        this.game.move(m.move);
        this.state$.next(createGameState(this.game));
      });
      
    }
  }
}
