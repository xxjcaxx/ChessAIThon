import { setFen } from "chessmarro-board";
import template from "./scenariosTemplate.html?raw"
import style from "./style.css?inline"
import { Chess } from 'chess.js'


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

const renderMoves= (moves) =>{
  const wrapper = document.createElement('div');
  wrapper.innerHTML = '<div class="tags">'+moves.map((move)=> `
  <span data-move="${move}"class="tag is-light is-clickable">${move}</span>
  `).join('')+'</ul>';
  return wrapper.firstChild;
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
    const movesList = this.querySelector("#moves-list");

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
        
        const chess = new Chess(fen, { skipValidation : true });
        const moves = chess.moves({verbose:true}).map(m=> m.lan)
        movesList.replaceChildren(renderMoves(moves))
        

      }
    });


    movesList.addEventListener("mouseover", (event)=>{
      if (event.target.tagName === "SPAN") {
        const [x,y,X,Y] = uciToMove(event.target.dataset.move);
        const resetBoard = board.board
        board.movePiece([x,y],[X,Y],0.5);
        event.target.addEventListener("mouseout",(et)=>{
          console.log("mouseout");
          
          board.movePiece([X,Y],[x,y],0.5);
          //board.board = resetBoard;
          setTimeout(()=>{
            board.board = resetBoard;
            board.refresh();
          },500)
          //board.refresh();
        }, { once: true });
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