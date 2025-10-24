import { setFen, getFen } from "chessmarro-board";
import template from "./players.html?raw"
import style from "./players.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";


class PlayersComponent extends HTMLElement {

    state = {
        currentFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        displayFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    }

    async connectedCallback() {
        // Estilos
        const styleElement = document.createElement("style");
        styleElement.textContent = style;
        this.append(styleElement);

        // Contenido
        this.innerHTML = template;

        function toggleAIInput(select, apiField) {

            if (select.value === 'ai') {
                apiField.classList.remove('is-hidden');
            } else {
                apiField.classList.add('is-hidden');
            }
        }

        const players = ["player1", "player2"];

        players.forEach(player => {
           // console.log(this);
            
            const select = this.querySelector(`#${player}-select`);
            const apiField = this.querySelector(`#${player}-api`);

            // Asegurar que el estado inicial esté correcto
            toggleAIInput(select, apiField);

            // Evento de cambio
            select.addEventListener("change", () => {
                toggleAIInput(select, apiField);
            });
        });



    }


}

customElements.define("chess-players", PlayersComponent);