import { setFen, getFen } from "chessmarro-board";
import template from "./scenariosTemplate.html?raw"
import style from "./style.css?inline"
import { Chess } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';



const fensToRows = (rows) => {
  return rows.map(
    (fen) => {
      const row = document.createElement("tr");
      row.classList.add("is-primary", "is-clickable");
      row.innerHTML = `<td data-fen="${fen.fen}">${fen.fen} </td>${fen.move ? `<td data-move="${fen.move}">${fen.move}</td>` : "<td></td>"}`;
      return row;
    }
  );

}

const renderMoves = (moves) => {
  const wrapper = document.createElement('div');
  wrapper.innerHTML = '<div class="tags">' + moves.map((move) => `
  <span data-move="${move.lan}"class="tag is-light is-clickable"><span class="is-size-4">${move.piece}</span>${move.lan}</span>
  `).join('') + '</ul>';
  //const moveDiv = wrapper.firstChild;

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


const uciToMove = (uci) => {
  const letters = [" ", "a", "b", "c", "d", "e", "f", "g", "h", " "];
  const [oy, ox, dy, dx] = uci.split("");
  return [
    letters.indexOf(oy) - 1,
    8 - parseInt(ox),
    letters.indexOf(dy) - 1,
    8 - parseInt(dx),
  ];
};


const chessPiecesUnicode = {
  'P': '♙', // Peón blanco
  'N': '♘', // Caballo blanco
  'B': '♗', // Alfil blanco
  'R': '♖', // Torre blanca
  'Q': '♕', // Reina blanca
  'K': '♔', // Rey blanco
  'p': '♟', // Peón negro
  'n': '♞', // Caballo negro
  'b': '♝', // Alfil negro
  'r': '♜', // Torre negra
  'q': '♛', // Reina negra
  'k': '♚'  // Rey negro
};

const loadLocalStorage = () => {
  let bestMoves = [];
  const localStorageData = localStorage.getItem('best_moves');
  if (localStorageData) {
    try {
      bestMoves = JSON.parse(localStorageData);
    }
    catch (e) {
    }
  }
  return bestMoves;
}




class ScenariosComponent extends HTMLElement {

  state = {
    currentFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    currentBoard: new BehaviorSubject(null),
    currentTurn: new BehaviorSubject(null),
    storedScenarios: new BehaviorSubject([]),
    loadedScenarios: new BehaviorSubject([]),
    displayFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
  }




  async connectedCallback() {


    // Estilos
    const styleElement = document.createElement("style");
    styleElement.textContent = style;
    this.append(styleElement);


    // Contenido
    const templateWrapper = document.createElement("div");
    templateWrapper.innerHTML = template;
    this.append(templateWrapper.querySelector(".main-content").cloneNode(true));
    const scenariosListDiv = this.querySelector("#scenariosList");
    const storedScenarios = this.querySelector("#storedScenarios");
    const loadedScenarios = this.querySelector("#loadedScenarios");
    const scenariosRepresentation = this.querySelector("#representation");
    const movesList = this.querySelector("#moves-list");

    //Datos
    const board = this.querySelector("chessmarro-board");
    const storedBestMoves = loadLocalStorage();

    this.state.currentBoard.next(board.board);
    this.state.currentTurn.next(board.turn);
    this.state.storedScenarios.next(storedBestMoves);


    // Tabla de escenarios guardados
    const scenariosListTable = templateWrapper.querySelector("#scenariosListTable").content.querySelector("table").cloneNode(true);
    const scenariosListTableTbody = scenariosListTable.querySelector("tbody");
    storedScenarios.append(scenariosListTable);

    this.state.storedScenarios.subscribe((storedBestMoves) => {
      scenariosListTableTbody.replaceChildren(...fensToRows(storedBestMoves));
    });

    // Escenarios cargados
    const loadedScenariosListTable = templateWrapper.querySelector("#scenariosListTable").content.querySelector("table").cloneNode(true);
    const loadedscenariosListTableTbody = loadedScenariosListTable.querySelector("tbody");
    loadedScenarios.append(loadedScenariosListTable);

    this.state.loadedScenarios.subscribe((loadedScenarios) => {
      loadedscenariosListTableTbody.replaceChildren(...fensToRows(loadedScenarios));
    });

    const resetDisplayFen = () => {
      console.log("displa");

      const fen = this.state.currentFen.getValue();
      this.state.displayFen.next(fen);
    }



    this.state.displayFen.subscribe((fen) => {
      const boardData = setFen(fen);
      board.board = boardData;
      board.refresh();
      renderMovesDiv(movesList, fen);
    });

    fromEvent(scenariosListDiv, "mouseover").pipe(
      map(event => event.target),
      filter(target => target.tagName === "TD" && target.dataset.fen),
      map(target => target.dataset.fen)
    ).subscribe(fen => {
      this.state.displayFen.next(fen);
    });

    fromEvent(scenariosListDiv, "mouseout").pipe(
      filter(event => event.target.tagName === "TD")
    ).subscribe(() => {
      /*
      const fen = this.state.currentFen.getValue();
      this.state.displayFen.next(fen);*/
      resetDisplayFen();
    });


    fromEvent(scenariosListDiv, "click").pipe(
      filter(event => event.target.tagName === "TD")
    ).subscribe((event) => {
      const fen = event.target.dataset.fen;
      this.state.currentFen.next(fen);
      this.state.displayFen.next(fen);
      this.state.currentBoard.next(setFen(fen));
      const chess = new Chess(fen, { skipValidation: true });
      this.state.currentTurn.next(chess.turn());
    });

    const promisifyMovePiece = ([x, y], [X, Y], time) => {
      return new Promise((resolve) => {
        setTimeout(() => {
          board.movePiece([x, y], [X, Y], 0.3);
          resolve([x, y, X, Y]);
        }, time * 1000);
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
        tap(move => console.log(JSON.stringify(move))),
        switchMap(move =>
          move
            ? promisifyMovePiece([move[0], move[1]], [move[2], move[3]], 0.3)
            : of(null).pipe(tap(() => {resetDisplayFen()}))
        ),
      )
      .subscribe();




    const makeMove = (move) => {
      const [x, y, X, Y] = uciToMove(move);
      board.movePiece([x, y], [X, Y], 0.3);
      // board.refresh();
      const chess = new Chess(this.state.currentFen, { skipValidation: true });
      chess.move(move);
      const fen = chess.fen();
      this.state.currentFen.next(fen);
      this.state.currentBoard.next(setFen(fen));
      this.state.currentTurn.next(chess.turn());

      renderMovesDiv(movesList, fen);
      storedBestMoves.push({ fen, move });
      localStorage.setItem('best_moves', JSON.stringify(storedBestMoves));
      scenariosListTableTbody.replaceChildren(...fensToRows(storedBestMoves));
    }

    movesList.addEventListener("click", (event) => {
      if (event.target.tagName === "SPAN" && event.target.dataset.move) {
        makeMove(event.target.dataset.move);
      }
    });





    document.querySelector('#load-defaults-button').addEventListener('click', async () => {
      const response = await fetch("chess_endgames.csv");
      const data = await response.text();
      const rows = data.split("\n").map((r) => ({ fen: r, move: null }));
      loadedscenariosListTableTbody.replaceChildren(...fensToRows([...rows]));
    });


    board.addEventListener("chessmarro-move", e => {
      console.log(e);
      makeMove(e.detail.uci);

    });

  }


}

customElements.define("chess-scenarios", ScenariosComponent);