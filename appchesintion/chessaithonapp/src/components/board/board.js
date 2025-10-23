import { Chess, validateFen } from 'chess.js'
import style from "./board.css?inline"
import template from "./board.html?raw"
import { setFen, getFen } from "chessmarro-board";
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";


const renderMoves = (moves) => {
    const wrapper = document.createElement('div');
    wrapper.innerHTML = '<div class="tags">' + moves.map((move) => `
  <span data-move="${move.lan}"class="tag is-light is-clickable"><span class="is-size-4">${move.piece}</span>${move.lan}</span>
  `).join('') + '</ul>';

    const moveDiv = document.createElement('div');
    moveDiv.classList.add('tags');
    const moveSpans = moves.map(move => {
        const moveSpan = document.createElement('span');
        moveSpan.classList.add('tag', 'is-light', 'is-clickable');
        moveSpan.dataset.move = move.lan;
        moveSpan.innerHTML = `<span class="is-size-4">${move.piece}</span>${move.lan}`;
        moveSpan.addEventListener('mouseenter', (e) => {
            if (e.target === moveSpan) {
                const customEvent = new CustomEvent('enterMove', {
                    bubbles: true,  // para que se propague
                    detail: { message: move.lan }
                });
                moveSpan.dispatchEvent(customEvent);
            }
        });
        moveSpan.addEventListener('mouseout', (e) => {
            if (e.target === moveSpan) {
                const customEvent = new CustomEvent('outMove', {
                    bubbles: true,  // para que se propague
                    detail: { message: move.lan }
                });
                moveSpan.dispatchEvent(customEvent);
            }
        });
        return moveSpan;
    });

    moveDiv.append(...moveSpans);
    return moveDiv;
}

const renderMovesDiv = (movesList, fen) => {
    if (fen) {
        const chess = new Chess(fen, { skipValidation: true });
        const currentTurn = chess.turn();
        const moves = chess.moves({ verbose: true }).map(m => ({ piece: chessPiecesUnicode[currentTurn === "b" ? m.piece : m.piece.toUpperCase()], lan: m.lan }))
        movesList.replaceChildren(renderMoves(moves))
    } else {
        movesList.replaceChildren();
    }

}

class boardComponent extends HTMLElement {

    state = {
        currentFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        currentBoard: new BehaviorSubject(null),
        currentTurn: new BehaviorSubject(null),
        //storedScenarios: new BehaviorSubject([]),
        //loadedScenarios: new BehaviorSubject([]),
        displayFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        movesHistory: new BehaviorSubject([]),
    }


    async connectedCallback() {

        let initFen = this.dataset.fen;
        if (!validateFen(initFen).ok) {
            initFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
        this.state.currentFen.next(initFen);
        this.state.displayFen.next(initFen);

        // Estilos
        const styleElement = document.createElement("style");
        styleElement.textContent = style;
        this.append(styleElement);


        this.innerHTML = template;


        const movesList = this.querySelector("#moves-list");
        const historyList = this.querySelector('#movesHistoryList');
        const currentFenDisplay = this.querySelector('#currentFen');
        const board = this.querySelector("chessmarro-board");

        const resetDisplayFen = () => {
            const fen = this.state.currentFen.getValue();
            this.state.displayFen.next(fen);
        }

        this.state.displayFen.subscribe((fen) => {
            const boardData = setFen(fen);
            board.board = boardData;
            board.refresh();
            renderMovesDiv(movesList, fen);
        });

        this.state.currentFen.subscribe(fen => {
            currentFenDisplay.innerHTML = `Link: <a href="#scenarios/${encodeURIComponent(fen)}">${fen}</a>`

        });

        const promisifyMovePiece = ([x, y], [X, Y], time) => {
            return new Promise((resolve) => {
                setTimeout(() => {
                    board.movePiece([x, y], [X, Y], 0.3);
                    resolve([x, y, X, Y]);
                }, time * 400);
            });

        }

        const currentMove$ = new BehaviorSubject(null);

        function getPieceUnderMouse(event) {
            const el = document.elementFromPoint(event.clientX, event.clientY);
            if (!el) return null;
            if (el.classList.contains('tag')) {
                const moveStr = el.dataset.move;
                if (!moveStr) return null;

                const move = uciToMove(moveStr);
                return move;
            }
            return null;
        }

        fromEvent(document, 'mousemove')
            .pipe(
                map(event => {
                    const move = getPieceUnderMouse(event);
                    return move
                        ? move
                        : null;
                }),
                distinctUntilChanged((a, b) => JSON.stringify(a) === JSON.stringify(b))
            )
            .subscribe(currentMove$);

        currentMove$
            .pipe(
                //tap(move => console.log(JSON.stringify(move))),
                concatMap(move =>
                    move
                        ? promisifyMovePiece([move[0], move[1]], [move[2], move[3]], 0.3)
                        : of(null).pipe(tap(() => { resetDisplayFen() }))
                ),
            )
            .subscribe();




        const makeMove = (move) => {
            const [x, y, X, Y] = uciToMove(move);
            board.movePiece([x, y], [X, Y], 0.3);
            // board.refresh();
            console.log(move);

            const chess = new Chess(this.state.currentFen.getValue(), { skipValidation: true });
            chess.move(move);
            const fen = chess.fen();
            this.state.currentFen.next(fen);
            this.state.currentBoard.next(setFen(fen));
            this.state.currentTurn.next(chess.turn());
            this.state.movesHistory.next([...this.state.movesHistory.getValue(), { fen, move }]);

            renderMovesDiv(movesList, fen);
            const storedBestMoves = loadLocalStorage();
            storedBestMoves.push({ fen, move });
            localStorage.setItem('best_moves', JSON.stringify(storedBestMoves));
            const customEvent = new CustomEvent('makeMove', {
                bubbles: true,  // para que se propague
                detail: { message: move.lan }
            });
            this.dispatchEvent(customEvent);
        }

        movesList.addEventListener("click", (event) => {
            if (event.target.tagName === "SPAN" && event.target.dataset.move) {
                makeMove(event.target.dataset.move);
            }
        });

        board.addEventListener("chessmarro-move", e => {
            console.log(e);
            makeMove(e.detail.uci);
        });

        this.state.movesHistory.subscribe(history => {
            historyList.innerHTML = '';
            history.forEach(h => {
                const li = document.createElement('li');
                li.textContent = `${h.fen} - ${h.move}`
                historyList.append(li);
            });
        });

    }
}

customElements.define("chess-board", boardComponent);