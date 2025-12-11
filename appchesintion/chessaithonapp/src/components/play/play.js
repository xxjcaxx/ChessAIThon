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
            board.dataset.fen = gs.fen;
            board.state.currentFen.next(gs.fen);
            board.state.displayFen.next(gs.fen);
            board.state.currentTurn.next(gs.currentPlayer);
            console.log(gs.fen);
            
        });

        const boardContainer = this.querySelector("#boardContainer");
        boardContainer.append(board);

        board.addEventListener("makeMove", (e) => {          
            const uci = e.detail.message;
            this.state.move = uci;
        });

        this.addEventListener("playersChanged", (e) => {
            const players = e.detail;
            this.state.players = players;
        });
    }


}

customElements.define("chess-play", PlayComponent);