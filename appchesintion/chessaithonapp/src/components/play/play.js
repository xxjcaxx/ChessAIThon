import { setFen, getFen } from "chessmarro-board";
import template from "./play.html?raw"
import style from "./play.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";
import { initStyle, initTemplate } from '../componentsUtils.js';
import { GameState } from "../../services/chessGameService.js";


class PlayComponent extends HTMLElement {

    state = new GameState("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
 
    async connectedCallback() {

        this.append(
            initStyle(style),
            initTemplate(template)
        );

        //Board
        const board = document.createElement("chess-board");

        this.state.state$.subscribe(gs => {
            board.state.movesHistory.next([...board.state.movesHistory.getValue(),
                { fen: board.dataset.fen, move: gs.lastMove}])
            board.dataset.fen = gs.fen;
            //board.makeMove(gs.lastMove);
            
            
            board.state.currentFen.next(gs.fen);
            board.state.displayFen.next(gs.fen);
            board.state.currentTurn.next(gs.currentPlayer);
            board.state.suggestedMoves.next(gs.suggestedMoves || []);
            //console.log(gs.fen , board.state.movesHistory.getValue());
            
        });

        const boardContainer = this.querySelector("#boardContainer");
        boardContainer.append(board);

        board.addEventListener("makeMove", (e) => {          
            const uci = e.detail.message;
            this.state.move = uci;
        });

        this.addEventListener("playersChanged", (e) => {
            const players = e.detail;
            // keep players (w/b and api urls)
            this.state.players = {
                w: players.w,
                b: players.b,
                wApi: players.wApi,
                bApi: players.bApi
            };
            // update AI settings (simulations, puct) if provided
            if (typeof players.simulations !== 'undefined' || typeof players.puct !== 'undefined') {
                this.state.settings = {
                    simulations: Number(players.simulations) || this.state.settings?.simulations || 400,
                    puct: Number(players.puct) || this.state.settings?.puct || 1.4,
                    suggestOnly: !!players.suggestOnly
                };
            }
        });

        // Start game event includes players + settings
        this.addEventListener('startGame', (e) => {
            const sel = e.detail;
            this.state.players = {
                w: sel.w,
                b: sel.b,
                wApi: sel.wApi,
                bApi: sel.bApi
            };
            this.state.settings = {
                simulations: Number(sel.simulations) || this.state.settings?.simulations || 400,
                puct: Number(sel.puct) || this.state.settings?.puct || 1.4,
                suggestOnly: !!sel.suggestOnly
            };

            const currentState = this.state.state$.getValue();
            if (currentState && !currentState.ended) {
                this.state.decideNextMove(currentState);
            }
        });

        // Start game button http://10.100.22.119:8000/predict
    }


}

customElements.define("chess-play", PlayComponent);