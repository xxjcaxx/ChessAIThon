import { setFen, getFen } from "chessmarro-board";
import template from "./players.html?raw"
import style from "./players.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";


class PlayersComponent extends HTMLElement {


    
    

    async connectedCallback() {
        // Estilos
        const styleElement = document.createElement("style");
        styleElement.textContent = style;
        this.append(styleElement);

        // Contenido
        this.innerHTML = template;

        const  toggleAIInput = (select, apiField) => {

            if (select.value === 'ai') {
                apiField.classList.remove('is-hidden');
            } else {
                apiField.classList.add('is-hidden');
            }
            const selected = {
                w: this.querySelector(`#player1-select`).value,
                b: this.querySelector(`#player2-select`).value,
                wApi: this.querySelector(`#player1-api input`).value,
                bApi: this.querySelector(`#player2-api input`).value
            };
            const event = new CustomEvent('playersChanged', { detail: selected, bubbles: true });
            this.dispatchEvent(event);
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
            apiField.addEventListener("input",()=>{
                toggleAIInput(select, apiField);
            })
        });

        // Botón de inicio de juego
        this.querySelector('#start_game').addEventListener('click', (e) => {
            e.preventDefault();
            const selected = {
                w: this.querySelector(`#player1-select`).value,
                b: this.querySelector(`#player2-select`).value,
                wApi: this.querySelector(`#player1-api input`).value,
                bApi: this.querySelector(`#player2-api input`).value
            };
            const event = new CustomEvent('startGame', { detail: selected, bubbles: true });
            this.dispatchEvent(event);
        });



    }


}

customElements.define("chess-players", PlayersComponent);