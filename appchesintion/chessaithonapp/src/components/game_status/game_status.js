import template from "./game_status.html?raw";
import style from "./game_status.css?inline";
import { initStyle, initTemplate } from "../componentsUtils.js";

class GameStatusComponent extends HTMLElement {
  _gameState = null;
  statusElement = null;

  connectedCallback() {
    if (!this.hasChildNodes()) {
      this.append(
        initStyle(style),
        initTemplate(template)
      );
    }

    this.statusElement = this.querySelector("#gameStatus");
    this.render();
  }

  set gameState(value) {
    this._gameState = value;
    this.render();
  }

  get gameState() {
    return this._gameState;
  }

  parseFenMeta(fen) {
    if (!fen || typeof fen !== "string") {
      return null;
    }

    const [position, activeColor, castling, enPassant, halfmove, fullmove] = fen.split(" ");
    if (!position || !activeColor) {
      return null;
    }

    const turnPiece = activeColor === "w" ? "♙" : "♟";
    const turnClass = activeColor === "w" ? "turn-white" : "turn-black";
    const castlingLabel = castling && castling !== "-" ? castling : "—";

    return {
      turnPiece,
      turnClass,
      castlingLabel,
      enPassant: enPassant || "-",
      halfmove: halfmove || "0",
      fullmove: fullmove || "1"
    };
  }

  renderStatusHtml(gs) {
    const fenMeta = this.parseFenMeta(gs?.fen);
    if (!fenMeta) {
      return "";
    }

    let stateText = "Playing";
    let stateClass = "status-chip status-running";

    if (gs.ended) {
      if (gs.endReason === "checkmate") {
        const winner = gs.winner === "white" ? "♙" : "♟";
        stateText = `Mate ${winner}`;
        stateClass = "status-chip status-checkmate";
      } else if (gs.endReason === "stalemate") {
        stateText = "Stalemate";
        stateClass = "status-chip status-draw";
      } else {
        stateText = "Draw";
        stateClass = "status-chip status-draw";
      }
    } else if (gs.inCheck) {
      const checkedSide = gs.currentPlayer === "w" ? "♙" : "♟";
      stateText = `Check ${checkedSide}`;
      stateClass = "status-chip status-check";
    }

    return `
      <span class="status-table">
        <span class="status-head">State</span>
        <span class="status-head">Turn</span>
        <span class="status-head">Castling</span>
        <span class="status-head">En Passant</span>
        <span class="status-head">Half</span>
        <span class="status-head">Full</span>

        <span class="${stateClass} status-cell status-state">${stateText}</span>
        <span class="status-chip status-cell status-turn ${fenMeta.turnClass}" title="Turn">${fenMeta.turnPiece}</span>
        <span class="status-chip status-cell status-meta">${fenMeta.castlingLabel}</span>
        <span class="status-chip status-cell status-meta">${fenMeta.enPassant}</span>
        <span class="status-chip status-cell status-meta">${fenMeta.halfmove}</span>
        <span class="status-chip status-cell status-meta">${fenMeta.fullmove}</span>
      </span>
    `;
  }

  render() {
    if (!this.statusElement) {
      return;
    }

    this.statusElement.innerHTML = this.renderStatusHtml(this._gameState);
  }
}

customElements.define("chess-game-status", GameStatusComponent);
