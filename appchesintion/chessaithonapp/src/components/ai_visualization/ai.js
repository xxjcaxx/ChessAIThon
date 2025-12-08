import { setFen, getFen } from "chessmarro-board";
import template from "./ai.html?raw"
import style from "./ai.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";
import { initStyle, initTemplate } from '../componentsUtils.js';
import { askAIMoveTree, GameState } from "../../services/chessGameService.js";


class aiComponent extends HTMLElement {

   
    async connectedCallback() {

        this.append(
            initStyle(style),
            initTemplate(template)
        );

        const fenInput = this.querySelector("#fenInput");
        const api = this.querySelector("#player-api");
        const analyzeButton = this.querySelector("#analyzeButton");
        console.log(fenInput,api,analyzeButton);
        

        analyzeButton.addEventListener("click", async (e) => {
            e.preventDefault();
            const fen = fenInput.value;
            const apiUrl = api.value;
            console.log(fen,apiUrl);
            

           /* if (!validateFen(fen).valid) {
                alert("Invalid FEN string");
                return;
            }*/
            const {move,tree} = await askAIMoveTree(apiUrl,fen,10);
            console.log(move,tree);
        });

            
        

    }


}

customElements.define("chess-ai-visualization", aiComponent);