import { Chess, validateFen } from 'chess.js'
import style from "./board.css?inline"
import template from "./board.html?raw"
import { setFen, getFen } from "chessmarro-board";
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";
import { initStyle, initTemplate } from '../componentsUtils.js';

const renderMoves = (moves) => {

    const moveDiv = document.createElement('div');
    moveDiv.classList.add('tags');
    const moveSpans = moves.map(move => {
        const moveSpan = document.createElement('span');
        moveSpan.classList.add('tag', 'is-light', 'is-clickable');
        
        moveSpan.dataset.move = move.lan;
        moveSpan.innerHTML = `<span class="is-size-4">${move.piece}</span>${move.lan}`;
        moveSpan.querySelector('.is-size-4').style.pointerEvents = 'none';
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

const renderMovesDiv = (movesList, fen, suggestedMoves = []) => {
    if (fen) {
        const chess = new Chess(fen, { skipValidation: true });
        const currentTurn = chess.turn();
        const legalMoveSet = new Set(chess.moves({ verbose: true }).map(m => m.lan));

        const suggested = (Array.isArray(suggestedMoves) ? suggestedMoves : [])
            .filter(move => legalMoveSet.has(move))
            .slice(0, 5);

        const sourceMoves = suggested.length
            ? suggested.map(lan => ({ lan, piece: chess.get(lan.slice(0, 2))?.type }))
            : Array.from(legalMoveSet).map(lan => ({ lan, piece: chess.get(lan.slice(0, 2))?.type }));

        const moves = sourceMoves.map(m => ({
            piece: chessPiecesUnicode[currentTurn === "b" ? m.piece : m.piece?.toUpperCase()],
            lan: m.lan
        }));
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
        displayFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        movesHistory: new BehaviorSubject([]),
        suggestedMoves: new BehaviorSubject([]),
    }


    async connectedCallback() {

        let initFen = this.dataset.fen;
        if (!validateFen(initFen).ok) {
            initFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
        this.state.currentFen.next(initFen);
        this.state.displayFen.next(initFen);

        // Estilos
        this.append(
            initStyle(style),
            initTemplate(template)
        );

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
            renderMovesDiv(movesList, fen, this.state.suggestedMoves.getValue());
        });

        this.state.suggestedMoves.subscribe((moves) => {
            renderMovesDiv(movesList, this.state.displayFen.getValue(), moves);
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

        const openPromotionSelector = (turn) => new Promise((resolve) => {
            const overlay = document.createElement('div');
            overlay.className = 'promotion-overlay';

            const modal = document.createElement('div');
            modal.className = 'promotion-modal';

            const title = document.createElement('h3');
            title.className = 'promotion-title';
            title.textContent = 'Elige promoción';

            const options = document.createElement('div');
            options.className = 'promotion-options';

            const pieces = [
                { code: 'q', white: 'Q', black: 'q' },
                { code: 'r', white: 'R', black: 'r' },
                { code: 'b', white: 'B', black: 'b' },
                { code: 'n', white: 'N', black: 'n' }
            ];

            const cleanup = () => {
                document.removeEventListener('keydown', onEscape);
                overlay.remove();
            };

            const selectPromotion = (pieceCode) => {
                cleanup();
                resolve(pieceCode || 'q');
            };

            const onEscape = (event) => {
                if (event.key === 'Escape') {
                    selectPromotion('q');
                }
            };

            pieces.forEach((piece) => {
                const button = document.createElement('button');
                button.type = 'button';
                button.className = 'promotion-option button';
                button.dataset.promotion = piece.code;
                const unicodeKey = turn === 'w' ? piece.white : piece.black;
                button.textContent = chessPiecesUnicode[unicodeKey];
                if (piece.code === 'q') {
                    button.classList.add('is-primary');
                }
                button.addEventListener('click', () => selectPromotion(piece.code));
                options.append(button);
            });

            overlay.addEventListener('click', (event) => {
                if (event.target === overlay) {
                    selectPromotion('q');
                }
            });

            modal.append(title, options);
            overlay.append(modal);
            this.append(overlay);
            document.addEventListener('keydown', onEscape);
        });

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




        const makeMove = async (move) => {
            const [x, y, X, Y] = uciToMove(move);
            board.movePiece([x, y], [X, Y], 0.3);
            const initialFen = this.state.currentFen.getValue();
            //console.log(initialFen);
            
            const chess = new Chess(initialFen, { skipValidation: true });

            const resolvePromotionMove = async (rawMove) => {
                if (!rawMove || rawMove.length < 4) {
                    return rawMove;
                }

                const from = rawMove.slice(0, 2);
                const to = rawMove.slice(2, 4);
                const promotionMove = chess
                    .moves({ verbose: true })
                    .find(m => m.from === from && m.to === to && m.promotion);

                if (!promotionMove) {
                    return rawMove;
                }

                const currentPromotion = rawMove[4]?.toLowerCase();
                if (["q", "r", "b", "n"].includes(currentPromotion)) {
                    return `${from}${to}${currentPromotion}`;
                }

                const promotion = await openPromotionSelector(chess.turn());
                return `${from}${to}${["q", "r", "b", "n"].includes(promotion) ? promotion : "q"}`;
            };

            const resolvedMove = await resolvePromotionMove(move);

            try {
                // = chess.fen();
                chess.move(resolvedMove, { sloppy: true });
                const fen = chess.fen();
                this.state.currentFen.next(fen);
                this.state.currentBoard.next(setFen(fen));
                this.state.currentTurn.next(chess.turn());
                this.state.movesHistory.next([...this.state.movesHistory.getValue(), { fen:initialFen, move: resolvedMove }]);
                this.state.suggestedMoves.next([]);

                renderMovesDiv(movesList, fen, []);
                const storedBestMoves = loadLocalStorage();
                storedBestMoves.push({ fen:initialFen, move: resolvedMove });
                localStorage.setItem('best_moves', JSON.stringify(storedBestMoves));
                const customEvent = new CustomEvent('makeMove', {
                    bubbles: true,  // para que se propague
                    detail: { message: resolvedMove }
                });
                this.dispatchEvent(customEvent);
            } catch (error) {
                console.error("Error making move:", error);
                resetDisplayFen();
                return;
            }


            ;

        }

        movesList.addEventListener("click", (event) => {
            if (event.target.tagName === "SPAN" && event.target.dataset.move) {
                makeMove(event.target.dataset.move);
            }
        });

        board.addEventListener("chessmarro-move", e => {
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