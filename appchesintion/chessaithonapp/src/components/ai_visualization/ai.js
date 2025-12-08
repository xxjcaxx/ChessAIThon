import { setFen, getFen } from "chessmarro-board";
import template from "./ai.html?raw"
import style from "./ai.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";
import { initStyle, initTemplate } from '../componentsUtils.js';
import { GameState } from "../../services/chessGameService.js";


class aiComponent extends HTMLElement {

   
    async connectedCallback() {

        this.append(
            initStyle(style),
            initTemplate(template)
        );

        
        

    }


}

customElements.define("chess-ai-visualization", aiComponent);