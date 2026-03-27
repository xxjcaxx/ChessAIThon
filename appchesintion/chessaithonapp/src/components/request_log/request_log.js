import style from "./request_log.css?inline";
import { initStyle } from "../componentsUtils.js";

class RequestLogComponent extends HTMLElement {
  _entries = [];

  connectedCallback() {
    this.append(initStyle(style));

    const wrapper = document.createElement("div");
    wrapper.style.cssText = "max-width:600px;margin:0 auto";
    wrapper.innerHTML = `
      <div class="box has-background-dark p-0" style="overflow:hidden">
        <div class="is-flex is-justify-content-space-between is-align-items-center px-4 py-3 log-header">
          <span class="title is-6 has-text-white mb-0">📡 Server Requests</span>
          <button class="button is-small is-dark" id="clear-log">Clear</button>
        </div>
        <div id="log-list" class="log-list">
          <p class="has-text-grey is-size-7 has-text-centered p-4">No requests yet.</p>
        </div>
      </div>
    `;
    this.append(wrapper);

    wrapper.querySelector("#clear-log").addEventListener("click", () => {
      this._entries = [];
      this._render();
    });
  }

  addEntry(entry) {
    this._entries.unshift({ ...entry, _id: Date.now(), _expanded: false });
    this._render();
  }

  _render() {
    const list = this.querySelector("#log-list");
    if (!list) return;

    if (this._entries.length === 0) {
      list.innerHTML = `<p class="has-text-grey is-size-7 has-text-centered p-4">No requests yet.</p>`;
      return;
    }

    list.innerHTML = "";
    this._entries.forEach((entry) => list.appendChild(this._createEl(entry)));
  }

  _createEl(entry) {
    const hasProblems = entry.problems?.length > 0;
    const hasMove = !!entry.response?.move;
    const statusTag = hasProblems
      ? `<span class="tag is-danger is-light is-small">error</span>`
      : hasMove
        ? `<span class="tag is-success is-light is-small">ok</span>`
        : `<span class="tag is-warning is-light is-small">no move</span>`;
    const turnTagClass = entry.turn === "w" ? "is-white" : "is-dark";
    const turnLabel = entry.turn === "w" ? "W" : "B";
    const time = new Date(entry.timestamp).toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
    const fenBoard = entry.fen?.split(" ")[0] ?? "";

    const el = document.createElement("div");
    el.className = `log-entry${entry._expanded ? " is-expanded" : ""}`;

    el.innerHTML = `
      <div class="is-flex is-align-items-center px-4 py-2 log-entry-header is-clickable" style="gap:0.5rem">
        <span class="expand-icon has-text-grey" style="font-size:0.6rem">▶</span>
        <span class="tag ${turnTagClass}">${turnLabel}</span>
        <span class="is-family-monospace is-size-7 has-text-grey-light log-fen-preview" title="${entry.fen}">${fenBoard}</span>
        ${hasMove ? `<span class="tag is-success is-small">${entry.response.move}</span>` : ""}
        ${statusTag}
        <span class="is-size-7 has-text-grey is-family-monospace ml-1">${time}</span>
      </div>
      <div class="log-entry-body px-4 py-3">
        <p class="heading has-text-grey-light mb-1">FEN</p>
        <code class="is-family-monospace is-size-7 has-text-info">${entry.fen}</code>

        <p class="heading has-text-grey-light mt-3 mb-1">Response</p>
        ${
          entry.response
            ? `<pre class="log-pre is-size-7 is-family-monospace">${JSON.stringify(entry.response, null, 2)}</pre>`
            : `<span class="tag is-danger is-light">No response received</span>`
        }

        ${hasProblems ? `
          <p class="heading has-text-danger mt-3 mb-1">⚠ Problems</p>
          <ul>
            ${entry.problems.map(p => `
              <li class="is-flex is-align-items-center is-size-7 has-text-warning" style="gap:0.35rem">
                <span>▸</span><span>${p}</span>
              </li>`).join("")}
          </ul>` : ""}
      </div>
    `;

    el.querySelector(".log-entry-header").addEventListener("click", () => {
      entry._expanded = !entry._expanded;
      el.classList.toggle("is-expanded", entry._expanded);
    });

    return el;
  }
}

customElements.define("chess-request-log", RequestLogComponent);

