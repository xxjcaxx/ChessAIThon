import { setFen, getFen } from "chessmarro-board";
import template from "./scenariosTemplate.html?raw"
import style from "./style.css?inline"
import { Chess } from 'chess.js'


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
  return wrapper.firstChild;
}

const renderMovesDiv = (movesList, fen) => {
  if(fen){
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
    currentFen: null,
    currentBoard: null,
    currentTurn: null
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
    const scenariosRepresentation = this.querySelector("#representation");
    const board = this.querySelector("chessmarro-board");
    this.state.currentBoard = board.board;
    this.state.currentTurn= "w";
    const movesList = this.querySelector("#moves-list");
    // Tabla de escenarios
    const scenariosListTable = templateWrapper.querySelector("#scenariosListTable").content.querySelector("table");
    const scenariosListTableTbody = scenariosListTable.querySelector("tbody");

    // Escenarios guardados
    const storedBestMoves = loadLocalStorage();
    scenariosListTableTbody.replaceChildren(...fensToRows(storedBestMoves));

    scenariosListDiv.addEventListener("mouseover", (event) => {
      if (event.target.tagName === "TD") {
        const fen = event.target.dataset.fen;
        board.board = setFen(fen);
        board.refresh();
        renderMovesDiv(movesList, fen);
      }
    });

    scenariosListDiv.addEventListener("mouseout", (event) => {
      if (event.target.tagName === "TD") {
        const fen = this.state.currentFen;
        board.board = this.state.currentBoard;
        board.refresh();
        renderMovesDiv(movesList, fen);
      }
    });

    scenariosListDiv.addEventListener("click", (event) => {
      if (event.target.tagName === "TD") {
        this.state.currentFen = event.target.dataset.fen;
        board.board = setFen(this.state.currentFen);
        this.state.currentBoard = board.board;
        board.refresh();
        const chess = new Chess(this.state.currentFen, { skipValidation: true });
        this.state.currentTurn = chess.turn();

        renderMovesDiv(movesList, this.state.currentFen);
      }
    });


    let previewTimeout = null;
    movesList.addEventListener("mouseover", (event) => {
      if (event.target.tagName === "SPAN" && event.target.dataset.move) {
        // 1. Cancelar cualquier reseteo pendiente y restaurar el tablero.
        clearTimeout(previewTimeout);
        board.board = this.state.currentBoard;
        board.refresh();

        // 2. Animar el nuevo movimiento.
        const [x, y, X, Y] = uciToMove(event.target.dataset.move);
        board.movePiece([x, y], [X, Y], 0.3);

        // 3. Programar el reseteo al salir del span.
        event.target.addEventListener("mouseout", () => {
          previewTimeout = setTimeout(() => {
            board.movePiece([X, Y], [x, y], 0.3, () => {
              board.board = this.state.currentBoard;
              board.refresh();
            });
          }, 50); // Un pequeño delay para que la transición a otro span sea fluida.
        }, { once: true });
      }
    });

    movesList.addEventListener("click", (event) => {
      if (event.target.tagName === "SPAN" && event.target.dataset.move) {

        const [x, y, X, Y] = uciToMove(event.target.dataset.move);
        board.movePiece([x, y], [X, Y], 0.3);
        board.refresh();

        //this.state.currentBoard = board.board;
         const chess = new Chess(this.state.currentFen, { skipValidation: true });
         chess.move(event.target.dataset.move);
         this.state.currentFen = chess.fen();
         this.state.currentBoard = setFen(this.state.currentFen);

        this.state.currentTurn = chess.turn();
        const fen = this.state.currentFen;
    
        renderMovesDiv(movesList, fen);
        storedBestMoves.push({fen, move: event.target.dataset.move});
        localStorage.setItem('best_moves', JSON.stringify(storedBestMoves));
        scenariosListTableTbody.replaceChildren(...fensToRows(storedBestMoves));
      }
    });

    
    scenariosListDiv.append(scenariosListTable);
    

    document.querySelector('#load-defaults-button').addEventListener('click', async () => {
      const response = await fetch("chess_endgames.csv");
      const data = await response.text();
      const rows = data.split("\n").map((r)=>({fen: r, move: null}));
      scenariosListTableTbody.replaceChildren(...fensToRows([...storedBestMoves,...rows]));
    });

  }


}

customElements.define("chess-scenarios", ScenariosComponent);