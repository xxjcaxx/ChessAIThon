import { setFen, getFen } from "chessmarro-board";
import template from "./scenariosTemplate.html?raw"
import style from "./style.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage, decodeIdentificator } from "../../chessUtils";


const createBestMoveSelector = (fen, bestMove) => {
  const chess = new Chess(fen, { skipValidation: true });
  const moves = chess.moves({ verbose: true });
  return `<div class="select is-small"><select>` + moves.map(move => `<option ${move.lan === bestMove ? "selected" : ""} value="${move.lan}">${move.lan}</option>`).join("") + "</select></div>";
}

const createInputValue = (value) => {
  return `<input type="number" 
  id="probabilidad" 
  name="probabilidad" 
  min="-1" 
  max="1" 
  step="0.01"
   value="${normalizeScenarioValue(value)}" class="input is-small"/>`
}

const normalizeScenarioValue = (value) => {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0.0;
};

const ensureScenarioValue = (scenario = {}) => {
  return {
    ...scenario,
    value: normalizeScenarioValue(scenario.value)
  };
};

const ensureScenariosWithValue = (scenarios = []) => {
  return scenarios.map(ensureScenarioValue);
};

const formatValueForCsv = (value) => {
  const normalized = normalizeScenarioValue(value);
  const text = `${normalized}`;
  return text.includes('.') ? text : `${text}.0`;
};




const fensToRows = (rows) => {
  return rows.map(
    (fen) => {
      const row = document.createElement("tr");
      row.classList.add("is-clickable");
      row.innerHTML = `
      <td data-fen="${fen.fen}">
      <span class="is-block has-text-weight-medium" style="white-space: nowrap;">
      ${fen.fen}
      </span>
      </td>
      <td data-move="${fen.move}">${createBestMoveSelector(fen.fen, fen.move)}</td>
      <td data-value="${fen.value}">${createInputValue(fen.value)}</td>
      <td>
        <span data-action="delete" data-fen="${fen.fen}" title="Delete scenario">
          <img src="/trash.png" alt="Delete" class="scenario-action-icon" />
        </span>
        <span data-action="linktoplaysmall" data-fen="${fen.fen}" title="Open in play">
          <img src="/linktoplay.png" alt="Play" class="scenario-action-icon" />
        </span>
        <span data-action="representationLink" data-fen="${fen.fen}" title="Open representation">
          <img src="/matrixiconsmall.png" alt="Representation" class="scenario-action-icon" />
        </span>
      </td>`;

      row.querySelector("select").addEventListener("change", (event) => {
        const newMove = event.target.value;
        const newFen = fen.fen;
        const storedBestMoves = loadLocalStorage();
        const index = storedBestMoves.findIndex(fen => fen.fen === newFen);
        if (index !== -1) {
          storedBestMoves[index].move = newMove;
          localStorage.setItem('best_moves', JSON.stringify(storedBestMoves));
        }
      });

      row.querySelector("input").addEventListener("change", (event) => {
        const newValue = event.target.value;
        const newFen = fen.fen;
        const storedBestMoves = loadLocalStorage();
        const index = storedBestMoves.findIndex(fen => fen.fen === newFen);
        if (index !== -1) {
          storedBestMoves[index].value = normalizeScenarioValue(newValue);
          localStorage.setItem('best_moves', JSON.stringify(storedBestMoves));

        }
      });


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
    const fen = decodeIdentificator(identificator);
    if (fen) {
      this.state.currentFen.next(fen);
      this.state.displayFen.next(fen);
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




  const storedBestMoves = ensureScenariosWithValue(loadLocalStorage());
  localStorage.setItem('best_moves', JSON.stringify(storedBestMoves));


    this.state.storedScenarios.next(storedBestMoves);
    //console.log(storedBestMoves);



    // Tabla de escenarios guardados
    const scenariosListTable = templateWrapper.querySelector("#scenariosListTable").content.querySelector("table").cloneNode(true);
    const scenariosListTableTbody = scenariosListTable.querySelector("tbody");
    storedScenarios.append(scenariosListTable);

  

    // Escenarios cargados
    const loadedScenariosListTable = templateWrapper.querySelector("#scenariosListTable").content.querySelector("table").cloneNode(true);
    const loadedscenariosListTableTbody = loadedScenariosListTable.querySelector("tbody");
    loadedScenarios.append(loadedScenariosListTable);




    const resetDisplayFen = () => {
      const fen = this.state.currentFen.getValue();
      this.state.displayFen.next(fen);
    }

    const resetEvents = () => {
      fromEvent(scenariosListDiv.querySelectorAll("td"), "mouseover").pipe(
        map(event => event.currentTarget),
        filter(target => target.dataset.fen),
        map(target => target.dataset.fen),
      ).subscribe(fen => {
        this.state.displayFen.next(fen);
      });

      fromEvent(scenariosListDiv.querySelectorAll("td"), "mouseout").pipe(
        //filter(event => event.target.tagName === "TD")
      ).subscribe(() => {
        resetDisplayFen();
      });


      fromEvent(scenariosListDiv.querySelectorAll("td"), "click").pipe(
        filter(event => event.currentTarget.dataset.fen)
      ).subscribe((event) => {
        const fen = event.currentTarget.dataset.fen;
        this.state.currentFen.next(fen);
        this.state.displayFen.next(fen);
      });
    }


    this.state.loadedScenarios.subscribe((loadedScenarios) => {
      loadedscenariosListTableTbody.replaceChildren(...fensToRows(loadedScenarios));
      resetEvents();
    });

      this.state.storedScenarios.subscribe((storedBestMoves) => {
      scenariosListTableTbody.replaceChildren(...fensToRows(storedBestMoves));
      resetEvents();
    });

    fromEvent(scenariosListDiv, "click").pipe(
      map(event => event.target.closest('span[data-action="representationLink"]')),
      filter(Boolean)
    ).subscribe((actionElement) => {
      const fen = actionElement.dataset.fen;
      window.location.href = `#representation/${encodeURIComponent(fen)}`;
    });

    fromEvent(scenariosListDiv, "click").pipe(
      map(event => event.target.closest('span[data-action="linktoplaysmall"]')),
      filter(Boolean)
    ).subscribe((actionElement) => {
      const fen = actionElement.dataset.fen;
      window.location.href = `#play/${encodeURIComponent(fen)}`;
    });

    fromEvent(scenariosListDiv, "click").pipe(
      map(event => event.target.closest('span[data-action="delete"]')),
      filter(Boolean)
    ).subscribe((actionElement) => {
      console.log("Borrar: ", actionElement.dataset.fen);
      const storedScenarios = this.state.storedScenarios.getValue();
      const index = storedScenarios.findIndex(fen => fen.fen === actionElement.dataset.fen);
      if (index !== -1) {
        storedScenarios.splice(index, 1);
        this.state.storedScenarios.next(storedScenarios);
        localStorage.setItem('best_moves', JSON.stringify(storedScenarios));
      }

    });



    fromEvent(this, "makeMove").subscribe((event) => {
      const storedBestMoves = ensureScenariosWithValue(loadLocalStorage());
      localStorage.setItem('best_moves', JSON.stringify(storedBestMoves));
      this.state.storedScenarios.next(storedBestMoves);
      //const ultimoElemento = storedScenarios.lastElementChild;
      storedScenarios.scrollTo({
        top: storedScenarios.scrollHeight,
        behavior: 'smooth'
      });
    });


    const loadDataIntoTable = (data) => {
      const rows = data.split("\n").map((r) => {
        const [fen, move, value] = r.split(",");
        return ({
          fen: fen.trim().replace(/^"(.*)"$/, '$1'),
          move: move ? move.trim().replace(/^"(.*)"$/, '$1') : null,
          value: value ? parseFloat(value.trim().replace(/^"(.*)"$/, '$1')) : 0.0
        });
      }).filter(row => validateFen(row.fen).ok);
      console.log(rows);

      this.state.loadedScenarios.next([...rows]);
    };


    this.querySelector('#load-defaults-button').addEventListener('click', async () => {
      const response = await fetch("chess_endgames.csv");
      const data = await response.text();
      loadDataIntoTable(data);

    });


    this.querySelector('#fileInput').addEventListener('change', async (event) => {
      const file = event.target.files[0];
      const reader = new FileReader();
      reader.onload = async (event) => {
        const data = event.target.result;
        loadDataIntoTable(data);

      };
      reader.readAsText(file);
    });

    this.querySelector('#buttonLink').addEventListener('click', async () => {
      const link = this.querySelector('#linktocsv').value;
      const response = await fetch(link);
      const data = await response.text();
      loadDataIntoTable(data);

    });

    fromEvent(document.querySelector('#addYourFen'), "click").subscribe(() => {
      const storedScenarios = this.state.storedScenarios.getValue();
      const yourFEN = document.querySelector('#yourFen').value;
      storedScenarios.push({ fen: yourFEN, move: null, value: 0.0 });
      this.state.storedScenarios.next(storedScenarios);


      localStorage.setItem('best_moves', JSON.stringify(storedScenarios));
    });

    this.querySelector('#saveCSVButton').addEventListener('click', async () => {
      //loadLocalStorage(); // Asegurarse de cargar los datos más recientes
      const storedBestMoves = ensureScenariosWithValue(loadLocalStorage());
      localStorage.setItem('best_moves', JSON.stringify(storedBestMoves));
      console.log(storedBestMoves);
      const headers = ['fen', 'move', 'value'];

      // 2. Construir las filas del CSV
      const csvContent = [
        headers.join(','), // Cabecera
        ...storedBestMoves.map(row =>
          headers.map(fieldName => {
            if (fieldName === 'value') {
              return JSON.stringify(formatValueForCsv(row.value));
            }

            if (fieldName === 'move') {
              return JSON.stringify(row.move || '');
            }

            return JSON.stringify(row[fieldName] || '');
          }).join(',')
        )
      ].join('\r\n');

      // 3. Crear el archivo y descargarlo
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');

      link.setAttribute('href', url);
      link.setAttribute('download', 'scenarios.csv');
      link.style.visibility = 'hidden';

      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });



    this.querySelector("#show_your_scenarios").addEventListener("click", () => {
      this.querySelector("#yourScenarios").classList.remove("is-hidden");
      this.querySelector("#loadedScenariosDiv").classList.add("is-hidden");
      this.querySelector("#show_your_scenarios").classList.add("is-active");
      this.querySelector("#show_loaded_scenarios").classList.remove("is-active");
    });

    this.querySelector("#show_loaded_scenarios").addEventListener("click", () => {
      this.querySelector("#yourScenarios").classList.add("is-hidden");
      this.querySelector("#loadedScenariosDiv").classList.remove("is-hidden");
      this.querySelector("#show_your_scenarios").classList.remove("is-active");
      this.querySelector("#show_loaded_scenarios").classList.add("is-active");
    });





    // fin connected callback
  }


}

customElements.define("chess-scenarios", ScenariosComponent);