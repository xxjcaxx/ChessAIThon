import { setFen, getFen } from "chessmarro-board";
import template from "./representation.html?raw"
import style from "./representation.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage, decodeIdentificator } from "../../chessUtils";
import { CodificationService } from "./codification.service";

const codificationService = new CodificationService();

const renderLayer = (layer, id) => {
    const wrapper = document.createElement('div');
    wrapper.innerHTML = `<div class="layer">
  <span>${codificationService.name_layers[id] ? codificationService.name_layers[id] : ''}</span>
  <div class="boardRep">
  ${layer.map((row, idRow) =>
        row.map((cell, idCol) => {
            return `<div class="cellRep ${(idRow + idCol) % 2 === 1 ? 'blackRep' : 'whiteRep'} ${cell != 0 ? 'activeRep' : 'innactiveRep'}">${cell}</div>`

        }).join('')
    ).join('')
        }
    
    </div>
</div>`;
    return wrapper.firstElementChild;

}

const renderLayers = (layers) => {
    const wrapper = document.createElement('div');
    wrapper.classList.add('layers');
    layers.forEach((layer, id) => {
        wrapper.append(renderLayer(layer, id));
    });
    return wrapper;
}



class RepresentationComponent extends HTMLElement {

    state = {
        currentFen: new BehaviorSubject("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    }

    connectedCallback() {


        this.innerHTML = template;
        const container = this.querySelector('#representation');
        const fenInput = this.querySelector('#fenInput');
        fenInput.value = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        const identificator = this.identificator;
        const fen = decodeIdentificator(identificator);
        if (fen) {
            this.state.currentFen.next(fen);

        }


        // Estilos
        const styleElement = document.createElement("style");
        styleElement.textContent = style;
        this.append(styleElement);


        this.state.currentFen.subscribe(fen => {
            const layers = codificationService.concat_fen_legal(fen);
            const boardPieces = codificationService.getBoardPieces(fen);
            console.log(boardPieces);
            const boardPiecesLayer = renderLayer(boardPieces);
            container.replaceChildren(renderLayers(layers));
            container.prepend(boardPiecesLayer);
        });

        fromEvent(fenInput, 'input').pipe(
        ).subscribe((event) => {
            console.log(fenInput.value);

            const fen = fenInput.value;
            if (validateFen(fen).ok) {
                console.log(fenInput.value);
                this.state.currentFen.next(fen);
            }
        });



    }


}

customElements.define("chess-representation", RepresentationComponent);


