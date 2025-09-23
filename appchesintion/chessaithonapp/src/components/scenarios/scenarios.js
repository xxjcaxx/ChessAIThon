import { setFen } from "chessmarro-board";
import template from "./scenariosTemplate.html?raw"
import style from "./style.css?inline"


const fensToRows = (rows) => {
  return rows.map(
    (fen) => {
      const row = document.createElement("tr");
      row.classList.add("is-primary","is-clickable");
      row.innerHTML = `<td data-fen="${fen}">${fen}</td>`;
      return row;
    }
  );

}

class ScenariosComponent extends HTMLElement {

  async connectedCallback() {
    const styleElement = document.createElement("style");
    styleElement.textContent = style;
    this.append(styleElement);
    const templateWrapper = document.createElement("div");
    templateWrapper.innerHTML = template;
    this.append(templateWrapper.querySelector(".main-content").cloneNode(true));
    const scenariosListDiv = this.querySelector("#scenariosList");
    const scenariosRepresentation = this.querySelector("#representation");
    const board = this.querySelector("chessmarro-board");


    const scenariosListTable = templateWrapper.querySelector("#scenariosListTable").content.querySelector("table");
    const scenariosListTableTbody = scenariosListTable.querySelector("tbody");


    const localstorageData = localStorage.getItem('best_moves');
    if (localstorageData) {
      let bestMoves = [];
      try{
       bestMoves = JSON.parse(localstorageData);
      }
      catch(e){

      }
      const bestMovesFens = bestMoves.map(bmf=> bmf.fen)
      scenariosListTableTbody.append(...fensToRows(bestMovesFens));
    }


    scenariosListDiv.addEventListener("mouseover", (event) => {
      if (event.target.tagName === "TD") {
        const fen = event.target.dataset.fen;
        board.board = setFen(fen);
        board.refresh();
      }
    });




    scenariosListDiv.append(scenariosListTable);



    document.querySelector('#load-defaults-button').addEventListener('click', async () => {
      const response = await fetch("chess_endgames.csv");
      const data = await response.text();
      const rows = data.split("\n");
      scenariosListTableTbody.append(...fensToRows(rows));
    });

  }


}

customElements.define("chess-scenarios", ScenariosComponent);