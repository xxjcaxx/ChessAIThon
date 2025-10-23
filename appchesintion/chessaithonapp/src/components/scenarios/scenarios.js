import { setFen, getFen } from "chessmarro-board";
import template from "./scenariosTemplate.html?raw"
import style from "./style.css?inline"
import { Chess , validateFen} from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";


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


class ScenariosComponent extends HTMLElement {

  state = {
    currentFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    storedScenarios: new BehaviorSubject([]),
    loadedScenarios: new BehaviorSubject([]),
    displayFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
  }

  async connectedCallback() {

    const identificator = this.identificator;
    console.log(identificator);
    if (identificator) {
      const fen = decodeURIComponent(identificator)
      if (validateFen(fen).ok) {
        this.state.currentFen.next(fen);
        this.state.displayFen.next(fen);
      }
    }


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
    const historyList = this.querySelector('#movesHistoryList');
    const currentFenDisplay = this.querySelector('#currentFen');

    //Board
    const board = document.createElement("chess-board");
    board.dataset.fen = this.state.currentFen.getValue();
      // Board observables
    board.state.currentFen = this.state.currentFen;
    board.state.displayFen = this.state.displayFen;

    const boardContainer = this.querySelector("#boardContainer");
    boardContainer.append(board);

  


    const storedBestMoves = loadLocalStorage();

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
      const fen = this.state.currentFen.getValue();
      this.state.displayFen.next(fen);
    }


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
      resetDisplayFen();
    });


    fromEvent(scenariosListDiv, "click").pipe(
      filter(event => event.target.tagName === "TD")
    ).subscribe((event) => {
      const fen = event.target.dataset.fen;
      this.state.currentFen.next(fen);
      this.state.displayFen.next(fen);
    });


    fromEvent(this, "makeMove").subscribe((event) => {
      const storedBestMoves = loadLocalStorage();
      this.state.storedScenarios.next(storedBestMoves);
    });


    document.querySelector('#load-defaults-button').addEventListener('click', async () => {
      const response = await fetch("chess_endgames.csv");
      const data = await response.text();
      const rows = data.split("\n").map((r) => ({ fen: r, move: null }));
      loadedscenariosListTableTbody.replaceChildren(...fensToRows([...rows]));
    });


    // fin connected callback
  }


}

customElements.define("chess-scenarios", ScenariosComponent);