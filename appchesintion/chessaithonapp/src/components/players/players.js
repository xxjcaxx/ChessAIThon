import { setFen, getFen } from "chessmarro-board";
import template from "./players.html?raw"
import style from "./players.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";


class PlayersComponent extends HTMLElement {


    
    

    async connectedCallback() {
        const LOCAL_STORAGE_KEY = 'players_config';
        let aiStopped = false;

        const savePlayersConfig = () => {
            const payload = {
                w: this.querySelector(`#player1-select`)?.value || 'human',
                b: this.querySelector(`#player2-select`)?.value || 'human',
                wApi: this.querySelector(`#player1-api input`)?.value || '',
                bApi: this.querySelector(`#player2-api input`)?.value || ''
            };
            localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(payload));
        };

        const loadPlayersConfig = () => {
            try {
                const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
                if (!raw) return;
                const parsed = JSON.parse(raw);

                const p1Select = this.querySelector(`#player1-select`);
                const p2Select = this.querySelector(`#player2-select`);
                const p1ApiInput = this.querySelector(`#player1-api input`);
                const p2ApiInput = this.querySelector(`#player2-api input`);

                if (p1Select && (parsed.w === 'human' || parsed.w === 'ai')) p1Select.value = parsed.w;
                if (p2Select && (parsed.b === 'human' || parsed.b === 'ai')) p2Select.value = parsed.b;
                if (p1ApiInput && typeof parsed.wApi === 'string') p1ApiInput.value = parsed.wApi;
                if (p2ApiInput && typeof parsed.bApi === 'string') p2ApiInput.value = parsed.bApi;
            } catch (error) {
                console.error('Could not load players config from localStorage', error);
            }
        };

        // Estilos
        const styleElement = document.createElement("style");
        styleElement.textContent = style;
        this.append(styleElement);

        // Contenido
        this.innerHTML = template;

        loadPlayersConfig();

        const  toggleAIInput = (select, apiField) => {

            if (select && apiField) {
                if (select.value === 'ai') {
                    apiField.classList.remove('is-hidden');
                } else {
                    apiField.classList.add('is-hidden');
                }
            }

            const selected = {
                w: this.querySelector(`#player1-select`).value,
                b: this.querySelector(`#player2-select`).value,
                wApi: this.querySelector(`#player1-api input`).value,
                bApi: this.querySelector(`#player2-api input`).value,
                simulations: Number(this.querySelector('#ai-simulations').value) || 400,
                puct:  1.4,
                suggestOnly: this.querySelector('#ai-suggest-only')?.checked || false
            };
            const event = new CustomEvent('playersChanged', { detail: selected, bubbles: true });
            this.dispatchEvent(event);
            savePlayersConfig();

            const hasAnyAI = selected.w === 'ai' || selected.b === 'ai';
            const aiToggleBtn = this.querySelector('#ai_toggle');
            if (aiToggleBtn) {
                aiToggleBtn.disabled = !hasAnyAI;
            }
        }

        const renderAiToggleButton = () => {
            const aiToggleBtn = this.querySelector('#ai_toggle');
            if (!aiToggleBtn) {
                return;
            }

            aiToggleBtn.textContent = aiStopped ? 'Restart AI' : 'Stop AI';
            aiToggleBtn.classList.toggle('is-warning', !aiStopped);
            aiToggleBtn.classList.toggle('is-success', aiStopped);
        };

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
            });

            // Botones de URL predefinida
            apiField.querySelectorAll('[data-preset-url]').forEach(btn => {
                btn.addEventListener('click', () => {
                    const urlInput = apiField.querySelector('input[type="url"]');
                    if (urlInput) {
                        urlInput.value = btn.dataset.presetUrl;
                        toggleAIInput(select, apiField);
                    }
                });
            })
        });

        // Settings inputs should also trigger playersChanged
        const simsInput = this.querySelector('#ai-simulations');
        const puctInput = this.querySelector('#ai-puct');
        const suggestOnlyInput = this.querySelector('#ai-suggest-only');
        if (simsInput) simsInput.addEventListener('input', () => toggleAIInput());
        if (puctInput) puctInput.addEventListener('input', () => toggleAIInput());
        if (suggestOnlyInput) suggestOnlyInput.addEventListener('change', () => toggleAIInput());

        const aiToggleBtn = this.querySelector('#ai_toggle');
        renderAiToggleButton();
        if (aiToggleBtn) {
            aiToggleBtn.addEventListener('click', (e) => {
                e.preventDefault();
                aiStopped = !aiStopped;
                renderAiToggleButton();

                const event = new CustomEvent('aiToggleChanged', {
                    detail: { stopped: aiStopped },
                    bubbles: true
                });
                this.dispatchEvent(event);
            });
        }

        // Botón de inicio de juego
        this.querySelector('#start_game').addEventListener('click', (e) => {
            e.preventDefault();
            const selected = {
                w: this.querySelector(`#player1-select`).value,
                b: this.querySelector(`#player2-select`).value,
                wApi: this.querySelector(`#player1-api input`).value,
                bApi: this.querySelector(`#player2-api input`).value,
                simulations: Number(this.querySelector('#ai-simulations').value) || 400,
                puct:  1.4,
                suggestOnly: this.querySelector('#ai-suggest-only')?.checked || false
            };
            const event = new CustomEvent('startGame', { detail: selected, bubbles: true });
            this.dispatchEvent(event);

            if (aiStopped) {
                aiStopped = false;
                renderAiToggleButton();
                const resumeEvent = new CustomEvent('aiToggleChanged', {
                    detail: { stopped: false },
                    bubbles: true
                });
                this.dispatchEvent(resumeEvent);
            }

            savePlayersConfig();
        });



    }


}

customElements.define("chess-players", PlayersComponent);