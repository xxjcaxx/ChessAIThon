import { setFen } from "chessmarro-board";
import template from "./scenariosTemplate.html?raw"
import style from "./style.css?inline"

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
    const response = await fetch("chess_endgames.csv");
    const data = await response.text();
    const rows = data.split("\n");
    for (let fen of rows) {
      const row = document.createElement("tr");

      row.classList.add("is-primary");
      row.innerHTML = `<td data-fen="${fen}">${fen}</td>`;
      scenariosListTableTbody.appendChild(row);
    }

    scenariosListDiv.addEventListener("mouseover", (event) => {
      if (event.target.tagName === "TD") {
        const fen = event.target.dataset.fen;
        board.board = setFen(fen);
        board.render();
      }
    });


    scenariosListDiv.append(scenariosListTable);



  }


}

customElements.define("chess-scenarios", ScenariosComponent);