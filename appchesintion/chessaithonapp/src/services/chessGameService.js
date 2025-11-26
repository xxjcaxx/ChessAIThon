import { Chess } from "chess.js";
import { BehaviorSubject } from "rxjs";

const createGameState = (game) => ({
    board: game.board(),
    currentPlayer: game.turn(),
    ended: game.isGameOver(),
    winner: game.isGameOver() ? (game.isCheckmate() ? (game.turn() === 'w' ? 'black' : 'white') : 'draw') : null,
    inCheck: game.inCheck(),
    legalMoves: game.moves({ verbose: true }).map(m => m.lan),
    fen: game.fen(),
});


export class GameState {
    constructor(fen) {
        this.game = new Chess(fen);
        this.state$ = new BehaviorSubject(
            createGameState(this.game)
        );
        this.b = 'Human';
        this.w = 'Human';
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
        console.log("decide" + this[newGameState.currentPlayer]);
        
        if (this[newGameState.currentPlayer] == 'AI'){
            console.log("decide AI move");
            this.game.move(newGameState.legalMoves[Math.floor(Math.random() * newGameState.legalMoves.length)]);
             this.state$.next(createGameState(this.game));
        }
    }

}