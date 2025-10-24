import { setFen, getFen } from "chessmarro-board";
import template from "./play.html?raw"
import style from "./play.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";


class PlayComponent extends HTMLElement {

    state = {
        currentFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        displayFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    }

    async connectedCallback() {
        // Estilos
        const styleElement = document.createElement("style");
        styleElement.textContent = style;
        this.append(styleElement);

        // Contenido
        this.innerHTML = template;

        //Board
        const board = document.createElement("chess-board");
        board.dataset.fen = this.state.currentFen.getValue();
        // Board observables
        board.state.currentFen = this.state.currentFen;
        board.state.displayFen = this.state.displayFen;

        const boardContainer = this.querySelector("#boardContainer");
        boardContainer.append(board);
    }


}

customElements.define("chess-play", PlayComponent);