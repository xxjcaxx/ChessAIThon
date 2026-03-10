import { setFen, getFen } from "chessmarro-board";
import template from "./ai.html?raw"
import style from "./ai.css?inline"
import { Chess, validateFen } from 'chess.js'
import { BehaviorSubject, Subject, fromEvent, map, filter, tap, merge, switchMap, of, throttleTime, asyncScheduler, concat, take, concatMap, distinctUntilChanged } from 'rxjs';
import { uciToMove, chessPiecesUnicode, loadLocalStorage } from "../../chessUtils";
import { initStyle, initTemplate } from '../componentsUtils.js';
import { askAIMoveTree, GameState } from "../../services/chessGameService.js";
import { generateMiniChessBoard } from "../../chessUtils.js";


const addChessPiecesToTreeNode = (node,fen) => {
    const chess = new Chess(fen);
    const board = chess.board();
    node.miniBoard = generateMiniChessBoard(board);
    if (node.children) {
        node.children.forEach(child => {
            const move = chess.move(child.move);
            addChessPiecesToTreeNode(child,chess.fen());
            chess.undo();
        });
    }
}


class aiComponent extends HTMLElement {

   
    async connectedCallback() {
        const LOCAL_STORAGE_API_KEY = "ai_visualization_api_url";

        this.append(
            initStyle(style),
            initTemplate(template)
        );

        const fenInput = this.querySelector("#fenInput");
        const api = this.querySelector("#player-api");
        const simulationsInput = this.querySelector("#ai-simulations");
        const puctInput = this.querySelector("#ai-puct");
        const analyzeButton = this.querySelector("#analyzeButton");
        const aiSuggestedMove = this.querySelector("#aiSuggestedMove");
        console.log(fenInput,api,analyzeButton);

        const savedApiUrl = localStorage.getItem(LOCAL_STORAGE_API_KEY);
        if (savedApiUrl && api) {
            api.value = savedApiUrl;
        }

        if (api) {
            api.addEventListener("change", () => {
                localStorage.setItem(LOCAL_STORAGE_API_KEY, api.value.trim());
            });
        }
        

        analyzeButton.addEventListener("click", async (e) => {
            e.preventDefault();
            const fen = fenInput.value;
            const apiUrl = api.value;
            const simulations = Number(simulationsInput?.value) || 400;
            const puct = Number(puctInput?.value) || 1.0;
            localStorage.setItem(LOCAL_STORAGE_API_KEY, apiUrl.trim());
            console.log(fen,apiUrl);
            

           /* if (!validateFen(fen).valid) {
                alert("Invalid FEN string");
                return;
            }*/
            const { move, mcts_tree, tree } = await askAIMoveTree(apiUrl, fen, simulations, puct);
            const treeData = mcts_tree || tree;

            if (!treeData) {
                aiSuggestedMove.textContent = `Best move: ${move || 'N/A'}. The server did not return MCTS tree data.`;
                return;
            }

            addChessPiecesToTreeNode(treeData,fen);
            aiSuggestedMove.textContent = `Best move: ${move || 'N/A'}. Tree loaded from live AI response.`;
            console.log(move,treeData);

            const treeGraphContainer = this.querySelector("chess-mcts-visualizer");
            treeGraphContainer.treeData = treeData;
            
        });

            
        

    }


}

customElements.define("chess-ai-visualization", aiComponent);