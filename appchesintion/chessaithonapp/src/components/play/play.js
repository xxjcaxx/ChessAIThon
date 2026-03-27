import { setFen, getFen } from "chessmarro-board";
import template from "./play.html?raw"
import style from "./play.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage, decodeIdentificator } from "../../chessUtils";
import { initStyle, initTemplate } from '../componentsUtils.js';
import { GameState } from "../../services/chessGameService.js";
import '../request_log/request_log.js';


class PlayComponent extends HTMLElement {

    state = new GameState("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    stateSubscription = null;
    logSubscription = null;
    board = null;
    handleBoardMove = null;
    handlePlayersChanged = null;
    handleStartGame = null;
    handleAiToggleChanged = null;
 
    async connectedCallback() {
        const identificator = this.identificator;
        const fen = decodeIdentificator(identificator);
        if (fen) {
            this.state = new GameState(fen);
        }

        this.state.resume();

        this.append(
            initStyle(style),
            initTemplate(template)
        );

        //Board
        this.board = document.createElement("chess-board");
        const board = this.board;
        const gameStatus = this.querySelector("chess-game-status");

        this.stateSubscription = this.state.state$.subscribe(gs => {
            board.state.movesHistory.next([...board.state.movesHistory.getValue(),
                { fen: board.dataset.fen, move: gs.lastMove}])
            board.dataset.fen = gs.fen;
            //board.makeMove(gs.lastMove);
            
            
            board.state.currentFen.next(gs.fen);
            board.state.displayFen.next(gs.fen);
            board.state.currentTurn.next(gs.currentPlayer);
            board.state.suggestedMoves.next(gs.suggestedMoves || []);

            if (gameStatus) {
                gameStatus.gameState = gs;
            }
            //console.log(gs.fen , board.state.movesHistory.getValue());
            
        });

        const boardContainer = this.querySelector("#boardContainer");
        boardContainer.append(board);

        // Wire request log component
        const logComponent = this.querySelector('chess-request-log');
        if (logComponent) {
            this.logSubscription = this.state.requestLog$.subscribe(entry => {
                logComponent.addEntry(entry);
            });
        }

        this.handleBoardMove = (e) => {
            const uci = e.detail.message;
            this.state.move = uci;
        };
        board.addEventListener("makeMove", this.handleBoardMove);

        this.handlePlayersChanged = (e) => {
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
        };
        this.addEventListener("playersChanged", this.handlePlayersChanged);

        this.handleAiToggleChanged = (e) => {
            const shouldStop = !!e?.detail?.stopped;
            const currentState = this.state.state$.getValue();

            if (shouldStop) {
                this.state.stop();
                return;
            }

            this.state.resume();
            if (currentState && !currentState.ended) {
                this.state.decideNextMove(currentState);
            }
        };
        this.addEventListener('aiToggleChanged', this.handleAiToggleChanged);

        // Start game event includes players + settings
        this.handleStartGame = (e) => {
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
        };
        this.addEventListener('startGame', this.handleStartGame);

        // New game button
        this.querySelector('#new_game').addEventListener('click', () => {
            this.state.reset();
            if (this.board) {
                this.board.state.movesHistory.next([]);
            }
        });

        // Start game button http://10.100.22.119:8000/predict
    }

    disconnectedCallback() {
        this.state.stop();

        if (this.stateSubscription) {
            this.stateSubscription.unsubscribe();
            this.stateSubscription = null;
        }

        if (this.logSubscription) {
            this.logSubscription.unsubscribe();
            this.logSubscription = null;
        }

        if (this.board && this.handleBoardMove) {
            this.board.removeEventListener("makeMove", this.handleBoardMove);
        }

        if (this.handlePlayersChanged) {
            this.removeEventListener("playersChanged", this.handlePlayersChanged);
        }

        if (this.handleStartGame) {
            this.removeEventListener("startGame", this.handleStartGame);
        }

        if (this.handleAiToggleChanged) {
            this.removeEventListener("aiToggleChanged", this.handleAiToggleChanged);
        }

        this.handleBoardMove = null;
        this.handlePlayersChanged = null;
        this.handleStartGame = null;
        this.handleAiToggleChanged = null;
        this.board = null;
    }


}

customElements.define("chess-play", PlayComponent);