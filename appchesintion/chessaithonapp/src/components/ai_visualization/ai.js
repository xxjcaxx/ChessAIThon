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
            const {move,tree} = await askAIMoveTree(apiUrl,fen,100);
            addChessPiecesToTreeNode(tree,fen);
            console.log(move,tree);

            const treeGraphContainer = this.querySelector("chess-mcts-visualizer");
            treeGraphContainer.treeData = tree;
            
        });

            
        

    }


}

customElements.define("chess-ai-visualization", aiComponent);